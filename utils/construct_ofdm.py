import os
import h5py
import json
import numpy as np
import pandas as pd
from spectrum import pmtm
import matplotlib.pyplot as plt
from utils.evaluate_ofdm import evaluate_ofdm
from utils.channel_models import apply3GPPchannel
from data_loading import unpack_complex, pack_to_complex
from utils.synthetic_ofdm_modem import OFDMA, constellation_symbols, bits_to_symbols


def create_OFDM(num_samples=16384, num_symbols=16, mod_order=16, num_subcarriers=16, symbol_length=128, cyclic_prefix=(0, 0),
                EVM=-25, scfdma=False, bitstream_type="random", channel_3gpp=None, num_frames=1, normalize=True):

    bit_length = int(np.log2(mod_order))
    if bitstream_type != "random" and bitstream_type != "random_fixed":
        num_qam_symbols = (32 * 32 * 3 * 8) // bit_length if bitstream_type == "CIFAR" else (784 * 8) // bit_length
        num_subcarriers = 96 if bitstream_type == "CIFAR" else 56
        num_symbols = num_qam_symbols // num_subcarriers

    ofdm_dict = {"num_subcarriers": num_subcarriers, "num_OFDM_symbols": num_symbols, "symbol_length": symbol_length,
                 "scfdma": scfdma, "cyclic_prefix": cyclic_prefix, "mod_order": mod_order, "EVM": EVM,
                 "bitstream_type": bitstream_type, "channel3gpp": channel_3gpp, "num_frames": num_frames, "normalize": normalize}

    ofdm = OFDMA(num_subcarriers=num_subcarriers, num_OFDM_symbols=num_symbols, symbol_length=symbol_length,
                 cyclic_prefix=cyclic_prefix, channel_3gpp=channel_3gpp, num_frames=num_frames, normalize=normalize)
    dataset, labels = [], []
    symbol_locs = constellation_symbols(mod_order=mod_order)
    sample_labels =  range(num_samples)

    # Load fixed random bitstream to be modulated (only for channel applied datasets)
    if bitstream_type == "random_fixed":
        with open(r'./Datasets/fixed_random_bistreams.json', 'r') as F:
            fixed_random_bistreams = json.loads(F.read())
            full_bitstream_len = int(num_frames * num_symbols * num_subcarriers * bit_length)
            fixed_bitstream = np.array(fixed_random_bistreams["fixed_bitstream"])
            fixed_bitstream = fixed_bitstream[:full_bitstream_len]
            fixed_bitstream = fixed_bitstream.reshape(num_frames * num_symbols * num_subcarriers, bit_length)

    print("Modulating Dataset")
    for i, sample_label in enumerate(sample_labels):

        if i % (len(sample_labels) // 10) == 0:
            print(f"{i}/{len(sample_labels)}")

        # Create bitstream to be encoded to M-QAM symbols
        if bitstream_type == "random":
            bitstream = np.random.randint(0, 2, (num_frames * num_symbols * num_subcarriers, bit_length))
        else:   # bitstream_type == "random_fixed"
            bitstream = fixed_bitstream

        # Get symbol from bitstream
        symbol_stream_OFDM = bits_to_symbols(bitstream, symbol_locs)
        symbol_stream_OFDM = symbol_stream_OFDM.reshape(num_frames * num_symbols, num_subcarriers)

        # Modulate symbols to timeseries waveform and apply sc-FDMA if necessary
        x_OFDM = ofdm.OFDM_to_SCFDMA(symbol_stream_OFDM) if scfdma else symbol_stream_OFDM
        s_OFDM = ofdm.mod(x_OFDM)

        # Add AWGN to modulated waveform with sigma_n noise power level
        sigma_n = np.sqrt(10 ** (EVM / 10)) * np.sqrt(symbol_length / num_subcarriers) if normalize else np.sqrt(10 ** (EVM / 10))
        w = np.random.randn(1, s_OFDM.shape[1]) * sigma_n / np.sqrt(2) + 1j * np.random.randn(1, s_OFDM.shape[1]) * sigma_n / np.sqrt(2)
        waveform = s_OFDM + w

        # Add propagated channel distortion to datasets waveforms individually
        if channel_3gpp is not None:
            sample_rate = {128: 1.92e6, 256: 3.84e6, 512: 7.68e6, 1024: 15.36e6}[symbol_length]
            if channel_3gpp != "AWGN":
                waveform = apply3GPPchannel(waveform, sampling_rate=sample_rate, chan=channel_3gpp).reshape(1, -1)
        dataset.append(waveform)
        labels.append(sample_label)
    dataset = np.concatenate(dataset, axis=0)

    iq_data = unpack_complex(dataset).astype(float)
    channel_scale_coeffs = {}
    channel_max = iq_data.max()
    channel_min = iq_data.min()
    channel_scale_coeffs["max"] = channel_max
    channel_scale_coeffs["min"] = channel_min

    labels = pd.Series(labels)
    dataset = dataset.reshape(num_samples, -1)
    dataset = pd.DataFrame(dataset)
    dataset.insert(0, "labels", labels)
    return dataset, channel_scale_coeffs, ofdm_dict


def save_dataset(num_samples=65536, traffic_setting="low_traffic", mod_order=16, symbol_length=128, EVM=-25,
                 cp_type="long", scfdma=False, data_rep="iq", bitstream_type="random", channel_3gpp=None,
                 num_frames=1, normalize=True, evaluate=False, set_num=None):
    """
    Save Training target distribution as well as supporting metadata and evaluations
    :param num_samples: Int number of samples synthesized for training distribution
    :param traffic_setting: traffic setting for percent of occupied subcarriers used
    :param mod_order: M-QAM modulation order
    :param symbol_length: length of OFDM symbol fft windwo
    :param EVM: error vector magnitude in dB for AWGN
    :param cp_type: cyclic-prefix length setting
    :param scfdma: bool for converting OFDM modulated waveforms to sc-FDMA waveforms
    :param data_rep: save waveform representation as I/Q or Phase-Magnitude
    :param bitstream_type: Random or fixed bitstream type for modulating
    :param channel_3gpp: Apply an optional 3GPP propagated channel model
    :param num_frames: Number of subframes (0.5ms slots) to be modulated
    :param normalize: bool to normalize power of OFDM fft relative to number of subcarriers
    :param evaluate: evaluate target distribution
    :return:
    """

    # Specify coefficients based on the 3GPP standard
    sample_rate = {128: 1.92e6, 256: 3.84e6, 512: 7.68e6, 1024: 15.36e6}[symbol_length]
    cyclic_prefix = {"short": (5.2e-6, 4.69e-6), "long": (16.67e-6, 16.67e-6)}[cp_type]
    cyclic_prefix = [int(np.round(cp_len * sample_rate)) for cp_len in cyclic_prefix]
    num_symbols = {"short": 7, "long": 6}[cp_type]
    occupancy = {"low_traffic": 75 / 512, "mid_traffic": 150 / 512, "high_traffic": 225 / 512}[traffic_setting]
    num_subcarriers = int(symbol_length * occupancy)
    num_subcarriers = num_subcarriers - 1 if num_subcarriers % 2 != 0 else num_subcarriers

    train_dataset, scale_factors, ofdm_dict = create_OFDM(num_samples, num_symbols, mod_order, num_subcarriers, symbol_length,
                                                          cyclic_prefix, EVM, scfdma, bitstream_type, channel_3gpp,
                                                          num_frames, normalize)
    test_dataset, _, ofdm_dict = create_OFDM(num_samples//4, num_symbols, mod_order, num_subcarriers, symbol_length, cyclic_prefix,
                                             EVM, scfdma, bitstream_type, channel_3gpp, num_frames, normalize)

    # Dataset name based on OFDM synthetic parameters
    fdma = "_scfdma" if ofdm_dict["scfdma"] else ""
    chan = f"_{channel_3gpp}" if channel_3gpp is not None else ""
    cp = f"_cp_{cp_type}" if cyclic_prefix[0] > 0 else ""
    normalize = "_normed" if normalize else ""
    spec = data_rep + "_"
    string_3 = f"_subs{str(ofdm_dict['num_subcarriers'])}"
    dataset_name = rf"mod{str(mod_order)}_{spec}{bitstream_type}{cp}_{str(symbol_length)}fft_rblocks{str(ofdm_dict['num_frames'])}{string_3}" \
                   rf"_evm{str(EVM)}{fdma}{chan}{normalize}"
    dir_path = rf"./Datasets/{dataset_name}"

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if set_num is not None:
        h5f = h5py.File(f'{dir_path}/test_{str(set_num)}.h5', 'w')
    else:
        h5f = h5py.File(f'{dir_path}/test.h5', 'w')
    test_dataset = test_dataset.values
    h5f.create_dataset('train', data=test_dataset)
    h5f.close()

    if set_num is not None:
        h5f = h5py.File(f'{dir_path}/train_{str(set_num)}.h5', 'w')
    else:
        h5f = h5py.File(f'{dir_path}/train.h5', 'w')
    train_dataset = train_dataset.values
    h5f.create_dataset('train', data=train_dataset)
    h5f.close()

    # Save scale factors and OFDM parameter dictionary
    with open(rf'{dir_path}/scale_factors.json', 'w') as F:
        F.write(json.dumps(scale_factors))
    with open(rf'{dir_path}/ofdm_cls.json', 'w') as F:
        F.write(json.dumps(ofdm_dict))

    # Evaluate the target training distribution for quality control sake
    if evaluate:
        test_dataset = unpack_complex(test_dataset[:, 1:])
        data_path = "/".join(dir_path.split('/')[2:])
        dir_path = f"{dir_path}/"
        _ = evaluate_ofdm(test_dataset, dir_path, data_path, "Real", True)
        targ_data = pack_to_complex(test_dataset)
        target_spectrums = []
        for i in range(250):
            sample = targ_data[i, :]
            if i % (len(targ_data) // 10) == 0:
                print(i, end=", ")
            Sk, weights, eigenvalues = pmtm(sample, NW=4, method="eigen")
            Sk = np.mean(np.abs(Sk * weights) ** 2, axis=0)
            targ_spectrum = 10 * np.log10(np.fft.fftshift(Sk))
            target_spectrums.append(targ_spectrum)
        median_targ_spec = np.median(target_spectrums, axis=0)
        quant99_targ_spec = np.quantile(target_spectrums, 0.99, axis=0)
        plt.plot(range(len(quant99_targ_spec)), quant99_targ_spec, color="red", alpha=0.6, label="99th quantile spectrum")
        plt.plot(range(len(median_targ_spec)), median_targ_spec, color="blue", alpha=0.6, label="Median spectrum")
        plt.grid()
        plt.savefig(rf"{dir_path}spectrums.png")
        plt.show()


