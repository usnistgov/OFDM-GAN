import os
import json
import h5py
import numpy as np
import data_loading
from spectrum import pmtm
from numpy.linalg import norm
from numpy.fft import fftshift
from matplotlib import pyplot as plt
from data_loading import pack_to_complex
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import euclidean as l2
from utils.synthetic_ofdm_modem import demodulate_ofdm_waveforms, get_ZC_base_seqs, OFDMA


def constellation_density(demod_symbols, dataset_name, output_path):
    """
    Plot demodulated constellation from generated distribution
    :param demod_symbols: array of complex demodulated m-QAM symbols
    :param dataset_name: name of dataset distribution
    :param output_path: path to save figure
    :return:
    """
    print("Plot constellation density diagram")
    plt.figure(figsize=(7, 7))
    I, Q = np.real(demod_symbols[:, ]).flatten(), np.imag(demod_symbols[:, ]).flatten()
    plt.hist2d(I, Q, density=True, bins=150, range=[[-1.5, 1.5], [-1.5, 1.5]], cmap=plt.cm.binary)
    plt.axes().set_aspect('equal', 'box')
    plt.xlabel("I", fontsize=12)
    plt.ylabel("Q", fontsize=12)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig(output_path + f"Constellation_density_{dataset_name}.png", dpi=500)
    plt.close('all')


def plot_waveforms(data_complex, num_waveforms, dataset_name, waveform_type, output_path):
    """
    Plot target or generated OFDM waveforms for example plots/quality control plots
    :param data_complex: complex I/Q waveforms
    :param num_waveforms: number of waveforms to plot
    :param dataset_name: type of waveform distributions
    :param waveform_type: waveform type string identifier
    :param output_path: path to save figures
    :return:
    """
    wavetype_filename = waveform_type.replace(" ", "_")
    if not os.path.isdir(f"{output_path}{wavetype_filename}_waveform_plots/"):
        os.makedirs(f"{output_path}{wavetype_filename}_waveform_plots/")
    for i in range(num_waveforms):
        len_waveform = len(data_complex[i, :])
        plt.plot(range(len_waveform), np.real(data_complex[i, :]), alpha=0.7, label="I Component", color="blue")
        plt.plot(range(len_waveform), np.imag(data_complex[i, :]), alpha=0.7, label="Q Component", color="green")
        plt.xlabel("Time Index", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.legend()
        plt.grid()
        plt.savefig(output_path + f"{wavetype_filename}_waveform_plots/{dataset_name}_example_{i}.png", dpi=500)
        plt.close('all')


def evaluate_equalization(drs_seqs, subcarrier_indices, num_frames):
    """
    Calculate propagated channel coherence bandwidths from equalization coefficients taken
    from the recieved DMRS sequences
    :param drs_seqs: distribution of recieved DMRS sequences
    :param subcarrier_indices: occuplied subcarrier indicies
    :param num_frames: Number of slots being allocated
    :return: array of calculated Coherence bandwidths
    """
    coherence_bandwidths = []
    for received_drs_seqs, subcarrier_ind in zip(drs_seqs, subcarrier_indices):
        received_drs_seqs = received_drs_seqs.reshape(num_frames, -1)
        for frame_dmrs in received_drs_seqs:
            zc_seqs = get_ZC_base_seqs(len(frame_dmrs))
            dmrs_sequence = zc_seqs[0]
            channel_coefs = frame_dmrs / dmrs_sequence
            autocorrelation = np.abs(np.correlate(channel_coefs, channel_coefs, mode="full"))
            peaks, _ = find_peaks(autocorrelation)
            # calculate the half width at half max of the autocorrelation function
            widths, _, _, _ = peak_widths(autocorrelation, [len(channel_coefs) - 1], rel_height=0.5)
            coherence_bandwidths.append((0.5 * widths[0]) / 512)
    return coherence_bandwidths


def evaluate_BER(bitstreams, ofdm_params):
    """
    Compute the BER based on the fixed modulated bitstream from the target distribution
    :param bitstreams: array of Generated Bitstreams
    :param ofdm_params: dictionary of target OFDM parameters
    :return: Total generated BER value
    """
    bit_length = int(np.log2(ofdm_params["mod_order"]))
    num_frames = ofdm_params["num_frames"]
    num_symbols = ofdm_params["num_OFDM_symbols"]
    num_subcarriers = ofdm_params["num_subcarriers"]

    with open(r'./Datasets/fixed_random_bistreams.json', 'r') as F:
        fixed_random_bistreams = json.loads(F.read())
        full_bitstream_len = int(num_frames * num_symbols * num_subcarriers * bit_length)
        fixed_bitstream = np.array(fixed_random_bistreams["fixed_bitstream"])
        fixed_bitstream = fixed_bitstream[:full_bitstream_len]

    fixed_bitstream = fixed_bitstream.reshape(num_frames * num_symbols, -1)
    dmrs_inds = [3 + (num_symbols * i) for i in range(num_frames)]
    fixed_bitstream = np.delete(fixed_bitstream, dmrs_inds, 0)
    fixed_bitstream = fixed_bitstream.reshape(-1, )
    total_errors = 0
    total_bits = 0
    for received_bitstream in bitstreams:
        num_errors = len(np.where(np.not_equal(fixed_bitstream, received_bitstream))[0])
        total_errors += num_errors
        total_bits += len(fixed_bitstream)
    BER = total_errors / total_bits
    print(f"BER: {BER}, total_bits = {total_bits}, total errors = {total_errors}")
    return BER


def Coherence_Bandwidth_plots(targ_dist, gen_dist, output_path, save_bool):
    """
    Plot histograms of target and gerneated Coherence Bandwidth distributions
    :param targ_dist: array of target coherence bandwidth distributions
    :param gen_dist: array of generated coherence bandwidth distributions
    :param output_path: path to save figures
    :param save_bool: save or display figures
    :return:
    """
    min_targ, max_targ = min(targ_dist), max(targ_dist)
    min_gen, max_gen = min(gen_dist), max(gen_dist)
    lower_range, upper_range = min(min_targ, min_gen), max(max_targ, max_gen)
    bins = np.linspace(lower_range, upper_range, 100)
    plt.hist(targ_dist, bins, alpha=0.7, color="red",  label="Target Distribution")
    plt.hist(gen_dist, bins, alpha=0.7, color="blue", label="Generated Distribution")
    plt.ylabel("Count")
    plt.xlabel("Coherence Bandwidth (Digital Frequency)")
    plt.legend()
    plt.grid(True)
    if save_bool:
        plt.savefig(output_path + f"Coherence_bandwidth_hists.png", dpi=500)
    plt.close('all')


def evaluate_cyclic_prefix(gen_waveforms, targ_waveforms, cyclic_prefix, num_frames, output_path, save_bool):
    """
    Evaluate generated and target cyclic prefixes by looking at cross-correlation between cps and OFDM symbols
    :param gen_waveforms: distribution of generated waveforms
    :param targ_waveforms: distribution of target waveforms
    :param cyclic_prefix: tuple of length in timesteps of first and remaining cyclic prefix lengths
    :param num_frames: number of slots being modulated
    :param output_path: path to save directory
    :param save_bool: save plots or display plots
    :return: relative difference of max-cp cross correlation between target and generated distributions
    """
    plt.close('all')
    cp_type = "long" if cyclic_prefix[0] == cyclic_prefix[1] else "short"
    num_symbols = {"short": 7, "long": 6}[cp_type]
    gen_corr_vectors = []
    targ_corr_vectors = []
    for gen_waveform, targ_waveform in zip(gen_waveforms, targ_waveforms):
        gen_waveform = gen_waveform.reshape(num_frames, -1)
        targ_waveform = targ_waveform.reshape(num_frames, -1)
        # loop over each slot of each waveform (6 cps + OFDM symbols)
        for gen_frame, targ_frame in zip(gen_waveform, targ_waveform):
            if cyclic_prefix[1] != cyclic_prefix[0]:
                extra_timesteps = cyclic_prefix[0] - cyclic_prefix[1]
                gen_frame = gen_frame[:, extra_timesteps:]
                targ_frame = gen_frame[:, extra_timesteps:]
            gen_frame = gen_frame.reshape(num_symbols, -1)
            targ_frame = targ_frame.reshape(num_symbols, -1)
            # get generated and target cyclic prefixes
            gen_cps = gen_frame[:, :cyclic_prefix[1]]
            targ_cps = targ_frame[:, :cyclic_prefix[1]]
            # get generated and target OFDM symbols
            gen_ofdm_signal = gen_frame[:, cyclic_prefix[1]:]
            targ_ofdm_signal = targ_frame[:, cyclic_prefix[1]:]
            gen_ofdm_signal = gen_ofdm_signal.flatten()
            targ_ofdm_signal = targ_ofdm_signal.flatten()
            # Get the cross correlation between each cyclic prefix and the remaining OFDM symbols as one waveform
            for num, (gen_cp, targ_cp) in enumerate(zip(gen_cps, targ_cps)):
                # roll the cross correlation function to normalize lag coefficients across each of the cyclic
                # prefix locations in the full waveform for both generated and target
                gen_shift_val = len(gen_ofdm_signal) - num * (len(gen_ofdm_signal) // 6)
                gen_corr = np.correlate(gen_ofdm_signal, gen_cp)
                gen_corr = np.roll(gen_corr, gen_shift_val)
                gen_corr_vectors.append(gen_corr)

                targ_shift_val = len(targ_ofdm_signal) - num * (len(targ_ofdm_signal) // 6)
                targ_corr = np.correlate(targ_ofdm_signal, targ_cp)
                targ_corr = np.roll(targ_corr, targ_shift_val)
                targ_corr_vectors.append(targ_corr)

    avg_gen_corr = np.median(gen_corr_vectors, axis=0)
    avg_targ_corr = np.median(targ_corr_vectors, axis=0)
    # plot the median cyclic-prefix - OFDM symbol cross correlation functions for generated and target distributions
    plt.plot(range(len(avg_targ_corr)), np.abs(avg_targ_corr), alpha=0.5, color="red", label="Target")
    plt.plot(range(len(avg_gen_corr)), np.abs(avg_gen_corr), alpha=0.5, color="blue", label="Generated")
    plt.ylabel("Cross-Correlation Magnitude")
    plt.xlabel("Lag")
    plt.grid()
    plt.legend()
    if save_bool:
        plt.savefig(output_path + f"cyclic_prefix_correlations.png", dpi=500)
    plt.close('all')
    # return the relative difference maximum absolute cross correlation values between target and generated
    max_gen = np.abs(avg_gen_corr).max()
    max_targ = np.abs(avg_targ_corr).max()
    relative_cp_dist = np.abs(max_targ - max_gen) / max_targ
    return relative_cp_dist


def calculate_EVMs(demod_data_symbols, demod_closest_data_symbols):
    """
    Calculate the median EVM from the generated distribution waveforms
    :param demod_data_symbols: Demodulated M-QAM symbols
    :param demod_closest_data_symbols: Closest ideal M-QAM symbol to the received symbols
    :return: median EVM and intervals for 95% confidence interval
    """
    mean_EVMs = []
    for i, (closest_symbols, demod_symbols) in enumerate(zip(demod_closest_data_symbols, demod_data_symbols)):
        mean_evm = 10 * np.log10(np.mean(np.abs(closest_symbols - demod_symbols) ** 2) / np.mean(np.abs(closest_symbols) ** 2))
        mean_EVMs.append(mean_evm)
    evm = np.median(mean_EVMs)
    evm_quantile_025 = np.quantile(mean_EVMs, 0.025)
    evm_quantile_975 = np.quantile(mean_EVMs, 0.975)
    print(f"Median mean-EVM: {evm}")
    return evm, evm_quantile_025, evm_quantile_975


def get_spectrum_inds(subinds, fft, spectrum_len):
    """
    Get the in-band occupied spectrum indicies
    :param subinds: number of subcarriers occupied
    :param fft: fft window length
    :param spectrum_len: spectrum length
    :return: occupied sub-carrier indicies
    """
    resolution_scale = spectrum_len // fft
    subinds = subinds + [0]
    spectrum_inds = []
    for sub_ind in subinds:
        sub_ind_scaled = sub_ind * resolution_scale
        sub_spec_inds = list(range(sub_ind_scaled, sub_ind_scaled + resolution_scale))
        spectrum_inds = spectrum_inds + sub_spec_inds
    return spectrum_inds


def calculate_PSDs(iq_data, psd_method="eigen"):
    """
    Calculate Power spectral density (dB) of the generated waveforms
    :param iq_data: distribution of complex waveforms
    :param psd_method: which method of evaluation to use
    :return: array of PSD estimates in dB
    """
    spectrums = []
    if psd_method == "adapt":
        iq_data = iq_data[:500]
    print("Estimate PSD", end=": ")
    for i, sample in enumerate(iq_data):
        if i % (len(iq_data) // 10) == 0:
            print(i, end=", ")
        # use the multitaper method to compute spectral power
        Sk, weights, eigenvalues = pmtm(sample, NW=4, k=7, method=psd_method)
        if psd_method == "adapt":
            weights = weights.transpose()
        spectrum = 10 * np.log10(np.abs(np.mean(Sk * weights, axis=0) ** 2))
        spectrums.append(spectrum)
    return spectrums


def plot_psd_distributions(targ_spectrums, gen_spectrums, output_path, psd_method="eigen"):
    """
    plot median PSD of target and generated distributions
    :param targ_spectrums: array of target psds
    :param gen_spectrums: array of generated psds
    :param output_path: path to save directory
    :param psd_method: method used to estimate psds
    :return:
    """
    median_gen_spec, median_targ_spec = np.median(gen_spectrums, axis=0), np.median(targ_spectrums, axis=0)
    plt.close('all')
    w = np.linspace(-0.5, 0.5, len(median_gen_spec))
    plt.plot(w, np.fft.fftshift(median_gen_spec), color="blue", alpha=0.6, label="Median Generated spectrum")
    plt.plot(w, np.fft.fftshift(median_targ_spec), color="red", alpha=0.6, label="Median Target spectrum")
    plt.grid()
    plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=12)
    plt.ylabel("PSD (dB)", fontsize=12)
    plt.savefig(rf"{output_path}{psd_method}_spectrums.png")
    plt.close('all')

    plt.plot(w, np.fft.fftshift(median_targ_spec), color="black", alpha=0.85, label="Median Target spectrum")
    plt.grid()
    plt.xlabel("Digital Frequency", fontsize=12)
    plt.ylabel("PSD (dB)", fontsize=12)
    plt.savefig(rf"{output_path}Target_spectrums.png")
    plt.close('all')


def evaluate_spectrum_distance(targ_spectrums, gen_spectrums, subinds, fft):
    """
    Evaluate the relative difference between median target and generated PSD distributions
    :param targ_spectrums: array of target psds
    :param gen_spectrums: array of generated psds
    :param subinds: occupied subcarrier indicies
    :param fft: fft window length
    :return: relative distance for PSDs (whole, in-band, and out of band)
    """
    spectrum_len = len(gen_spectrums[0])
    spectrum_inds = get_spectrum_inds(subinds, fft, spectrum_len) if subinds is not None else None
    non_spectrum_inds = np.array([ind for ind in range(spectrum_len) if ind not in spectrum_inds]) if spectrum_inds is not None else None
    gen_spectrums = np.array(gen_spectrums)
    targ_spectrums = np.array(targ_spectrums)

    median_gen = np.median(gen_spectrums, axis=0)
    median_targ = np.median(targ_spectrums, axis=0)
    median_dist = l2(median_targ, median_gen) / norm(median_targ, 2)
    inband_median_dist, outband_median_dist = None, None
    # compute the inband and out-band portion of the PSD relative error if the subcarrier indicies are known
    if subinds is not None and non_spectrum_inds is not None:
        inband_median_gen, inband_median_targ = median_gen[subinds], median_targ[subinds]
        inband_median_dist = l2(inband_median_targ, inband_median_gen) / norm(inband_median_targ, 2)
        outband_median_gen, outband_median_targ = median_gen[non_spectrum_inds], median_targ[non_spectrum_inds]
        outband_median_dist = l2(outband_median_targ, outband_median_gen) / norm(outband_median_targ, 2)
    return median_dist, inband_median_dist, outband_median_dist


def calulate_empirical_SNR(demod_symbols, waveform_set, ofdm_params):
    """
    Compute SNR based on Transmitted and Recieved Generated signals
    :param demod_symbols: array of recieved demodulated QAM symbols
    :param waveform_set: array of generated OFDM waveforms
    :param ofdm_params: dictionary of OFDM parameters
    :return:
    """
    ofdm = OFDMA(num_subcarriers=ofdm_params["num_subcarriers"], num_OFDM_symbols=ofdm_params["num_OFDM_symbols"],
                 symbol_length=ofdm_params["symbol_length"], cyclic_prefix=ofdm_params["cyclic_prefix"], channel_3gpp=None,
                 num_frames=ofdm_params["num_frames"], normalize=ofdm_params["normalize"])
    noiseless_waveform_set = []
    for symbols in demod_symbols:
        waveform = ofdm.mod(symbols).reshape(-1, )
        # modulate the ideal QAM symbol stream to a noiseless waveform
        noiseless_waveform_set.append(waveform)
    noiseless_waveform_set = np.array(noiseless_waveform_set)
    # compute the generated noise as the difference of the generated distribution from the noiseless distribution
    noise_set = waveform_set - noiseless_waveform_set
    # Compute the Signal-to-Noise Ratio (SNR) in dB
    noise_power = np.sum(np.power(np.abs(noise_set), 2), axis=1)
    signal_power = np.sum(np.power(np.abs(noiseless_waveform_set), 2), axis=1)
    SNR = 10 * np.log10(signal_power / noise_power)
    median_SNR, quant025_SNR, quan975_SNR = np.median(SNR), np.quantile(SNR, 0.025), np.quantile(SNR, 0.975)
    # return the Median SNR across the generated distribution and the 95% confidence interval
    return median_SNR, quant025_SNR, quan975_SNR


def evaluate_ofdm(iq_data, output_path, data_path, dataset_name, save_bool):
    """
    Full OFDM signal evalutation method that computes all evaluation metrics
    :param iq_data: Generated Distribution of waveforms
    :param output_path: Path to save directory
    :param data_path: path to target distribution files
    :param dataset_name: Name of generated dataset
    :param save_bool:
    :return: tuple of computed metric distributions
    """
    print("Beginning OFDM evaluation: ")
    with open(rf'./Datasets/{data_path}/ofdm_cls.json', 'r') as F:
        ofdm_params = json.loads(F.read())
    if len(iq_data.shape) > 2:
        iq_data = pack_to_complex(iq_data)
    channel_model = True if ofdm_params["channel3gpp"] is not None else False

    # demodulate waveforms to retrieve QAM symbol streams, bitstreams, etc.
    demod_symbols, demod_closest_symbols, demod_drs_seqs, bit_streams, subcarrier_indices = \
        demodulate_ofdm_waveforms(iq_data, ofdm_params)

    psd_estimates = calculate_PSDs(iq_data, psd_method="eigen")
    SNR_data = [None, None, None]    # Signal-to-Noise Ratio Estimation:
    if not channel_model:
        SNR_data = calulate_empirical_SNR(demod_closest_symbols, iq_data, ofdm_params)
    evm_data = calculate_EVMs(demod_symbols, demod_closest_symbols)

    # Evaluate Channel model propagated waveforms
    BER = None
    coher_bws = None
    if channel_model:
        coher_bws = evaluate_equalization(demod_drs_seqs, subcarrier_indices, ofdm_params["num_frames"])
        eq_demod_symbols, eq_demod_closest_symbols, _, bit_streams, _ = \
            demodulate_ofdm_waveforms(iq_data, ofdm_params, equalize=True)
        if ofdm_params["bitstream_type"] == "random_fixed":
            BER = evaluate_BER(bit_streams, ofdm_params)
        evm_data = calculate_EVMs(eq_demod_symbols, eq_demod_closest_symbols)

    if save_bool:
        # plot demodulated constellations as a 2d Histogram
        constellation_density(np.concatenate(demod_symbols), dataset_name, output_path)
        # plot example waveforms
        plot_waveforms(iq_data, 5, dataset_name, "OFDM", output_path)
        # evaluate equalized constelation for channel propagated distribution
        if channel_model:
            constellation_density(np.concatenate(eq_demod_symbols), f"Equalized {dataset_name}", output_path)
            symbolstreams = data_loading.unpack_complex(eq_demod_symbols).astype(float)
            h5f = h5py.File(f'{output_path}/demodulated_equalized_gen_symbols.h5', 'w')
            h5f.create_dataset('equalized_demod_symbols', data=symbolstreams)
            h5f.close()
        # save the demodulated symbols for later plotting across multiple GAN runs
        symbolstreams = data_loading.unpack_complex(demod_symbols).astype(float)
        h5f = h5py.File(f'{output_path}/demodulated_gen_symbols.h5', 'w')
        h5f.create_dataset('demod_symbols', data=symbolstreams)
        h5f.close()
    return bit_streams, demod_symbols, demod_drs_seqs, psd_estimates, SNR_data, evm_data, BER, coher_bws, subcarrier_indices
