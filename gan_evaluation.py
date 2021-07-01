"""
This module holds high level generated distribution evaluation methods. Wrapper
methods are also present for evaluating target distributions


This file can also be imported as a module and contains the following functions:
    * MyEncoder - Custom Json encoder
    * plot_losses - plot GAN function losses
    * inverse_transform_dataset - inverse scale the generated distribution
    * load_generated - load and save a generated distribution for evaluation
    * load_target - load in target test distribution used for relative evaluations
    * load_test_distributions - load both test train and generated distributions
    * test_gan - main evaluation method
    * test_target_distribution - evaluate target distribution
    * retest_gan - re-evaluate gan distribution
"""

import json
import h5py
import numpy as np
import data_loading
import torch.utils.data
from pickle import dump
import matplotlib.pyplot as plt
from sklearn import preprocessing
import utils.evaluate_ofdm as evalofdm


class MyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle numpy objects and arrays
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(MyEncoder, self).default(obj)


def plot_losses(train_hist_df, model_name, save, output_path):
    """
    Plot the losses of D and G recorded in the train_hist dictionary during training
    :param train_hist_df: Dataframe containing batch-level training metrics
    :param model_name: String Name of Model
    :param save: Boolean whether to save plots, or plot to console
    :param output_path: Path to save plots to
    :return: None
    """
    D_loss, G_loss = train_hist_df["Loss_D"], train_hist_df["Loss_G"]
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(f"GAN Training Loss {model_name}")
    plt.plot(range(len(D_loss)), D_loss, color="red", alpha=0.4, label="Discriminator Loss")
    plt.plot(range(len(G_loss)), G_loss, color="blue", alpha=0.4, label="Generator Loss")
    D_loss_ma = D_loss.rolling(window=int(len(D_loss) / 100)).mean()
    G_loss_ma = G_loss.rolling(window=int(len(G_loss) / 100)).mean()
    plt.plot(range(len(D_loss_ma)), D_loss_ma, color="red")
    plt.plot(range(len(G_loss_ma)), G_loss_ma, color="blue")
    plt.ylabel("Loss")
    plt.xlabel("Batch number")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(output_path + "Gan_training_loss.png", dpi=500)
    plt.close('all')

    D_x, D_G_z, GP = train_hist_df["D(x)"], train_hist_df["D(G(z2))"], train_hist_df["GP"]
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(f"Critic Output: {model_name}")
    plt.plot(range(len(D_x)), D_x, color="red", alpha=0.4, label="D(x)")
    plt.plot(range(len(D_G_z)), D_G_z, color="blue", alpha=0.4, label="D(G(z))")
    plt.plot(range(len(GP)), GP, color="green", alpha=0.4, label="GP")
    D_x_ma = D_x.rolling(window=int(len(D_x) / 100)).mean()
    D_G_z_ma = D_G_z.rolling(window=int(len(D_G_z) / 100)).mean()
    GP_ma = GP.rolling(window=int(len(GP) / 100)).mean()
    plt.plot(range(len(D_x_ma)), D_x_ma, color="red")
    plt.plot(range(len(D_G_z_ma)), D_G_z_ma, color="blue")
    plt.plot(range(len(GP_ma)), GP_ma, color="green")
    plt.ylabel("Discriminator Output")
    plt.xlabel("Batch number")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(output_path + "Gan_D_output.png", dpi=500)
    plt.close('all')


def inverse_transform_dataset(gen_data, transformer, dataset, data_scaler):
    """
    Inverse transform Generated data relative to scaling of target distribution prior to training
    :param gen_data: Generated distribution
    :param transformer: Data scalingg transformer object
    :param dataset: Dataset Name
    :param data_scaler: Data scaling transformer object
    :return: inverse scaled generated distribution
    """
    # apply feature based min-max or max-absolute scaling
    if data_scaler.find("feature") != -1:
        gen_data_shape = gen_data.shape
        gen_data = gen_data.reshape(gen_data_shape[0], -1)
        gen_data = transformer.inverse_transform(gen_data)
        gen_data = gen_data.reshape(gen_data_shape)
    # Apply Global min-max scaling
    if data_scaler.find("global") != -1:
        with open(rf'./Datasets/{dataset}/scale_factors.json', 'r') as F:
            channel_scale_factors = json.loads(F.read())
            if data_scaler.find("iq") != -1:
                for i in range(gen_data.shape[1]):
                    channel_max, channel_min = channel_scale_factors[f"max_{str(i)}"],  channel_scale_factors[f"min_{str(i)}"]
                    feature_max, feature_min = 1, -1
                    gen_data[:, i, :] = (gen_data[:, i, :] - feature_min) / (feature_max - feature_min)
                    gen_data[:, i, :] = gen_data[:, i, :] * (channel_max - channel_min) + channel_min
            else:
                channel_max = channel_scale_factors[f"max"]
                channel_min = channel_scale_factors[f"min"]
                abs_max = np.max(np.abs([channel_max, channel_min]))
                if data_scaler == "global_min_max":
                    feature_max, feature_min = 1, -1
                    gen_data = (gen_data - feature_min) / (feature_max - feature_min)
                    gen_data = gen_data * (channel_max - channel_min) + channel_min
                else:
                    gen_data = gen_data * abs_max
    return gen_data


def load_generated(G, n_samples, device):
    """
    create generated distribution, sampled from Generator for GAN evaluation
    :param G: Generator model
    :param n_samples: Number of samples to be generated
    :param device: GPU device ID number
    :return: Generated data
    """
    gen_data = []
    num_generated_samples = 0
    batch_size = 1024
    num_batches = n_samples // batch_size
    print(f"Generating {n_samples} fake samples: ")
    for i in range(num_batches):
        z = data_loading.get_latent_vectors(batch_size, 100, False, device)
        fake = G(z)
        fake = fake.detach().cpu()
        num_generated_samples += len(fake)
        gen_data.append(fake)
    gen_data = torch.cat(gen_data, dim=0).numpy()
    gen_data = gen_data[:n_samples] if len(gen_data) > n_samples else gen_data
    return gen_data


def load_target(dataset, d_type, dist_name):
    """
    Load target distribution from h5 file and process it into the proper format
    :param dataset: Dataset name
    :param d_type: Data-type (Complex/float)
    :param dist_name: Distribution name (Test/validation)
    :return: Target distribution, and supporting info
    """
    h5f = h5py.File(f"./Datasets/{dataset}/{dist_name}.h5", 'r')
    targ_dataset = h5f['train'][:]
    h5f.close()

    targ_data = np.array(targ_dataset[:, 1:]).astype(d_type)
    targ_labels = np.array(np.real(targ_dataset[:, 0])).astype(np.int)

    targ_data = data_loading.unpack_complex(targ_data)
    targ_data = np.expand_dims(targ_data, axis=1) if len(targ_data.shape) < 3 else targ_data
    n_samples = len(targ_data)
    return targ_data, targ_labels, n_samples


def load_test_distributions(train_specs_dict, G, transformer, device):
    """
    Load in Generated and Target test distributions used for evaluation of GAN performance
    :param train_specs_dict: GAN configuration dictionary
    :param G: Generator model
    :param transformer: data scaler-transformer model
    :param device: GPU device ID number
    :return: Test distributions (Target/Generated)
    """
    dataset = train_specs_dict["dataloader_specs"]["dataset_specs"]["data_set"]
    data_scaler = train_specs_dict["dataloader_specs"]["dataset_specs"]["data_scaler"]
    try:
        pad_length = train_specs_dict['dataloader_specs']['dataset_specs']["pad_length"]
    except KeyError:
        pad_length = 0
    try:
        stftransform = train_specs_dict['dataloader_specs']['dataset_specs']["stft"]
        nperseg = train_specs_dict['dataloader_specs']['dataset_specs']["nperseg"]
    except KeyError:
        stftransform = False
        nperseg = 0
    try:
        fft_shift = train_specs_dict["dataloader_specs"]["dataset_specs"]["fft_shift"]
    except KeyError:
        fft_shift = False

    d_type = complex
    targ_data, targ_labels, n_samples = load_target(dataset, d_type, "test")
    gen_data = load_generated(G, n_samples, device)

    if transformer is None:
        transformer = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        train_data, train_labels, n_samples = load_target(dataset, d_type, "train")
        data_shape = train_data.shape
        if stftransform:
            train_data = data_loading.pack_to_complex(train_data)
            train_data, pad_length = data_loading.pad_signal_to_power_of_2(train_data)
            train_data, f, t = data_loading.waveform_to_stft(train_data, 2, nperseg)
            if fft_shift:
                train_data = np.fft.fftshift(train_data, axes=(1,))
            train_data = train_data.reshape(train_data.shape[0], nperseg, -1)
            train_data = train_data.view(complex)
            train_data = data_loading.unpack_complex(train_data).view(float)

        train_data = train_data.reshape(data_shape[0], -1)
        transformer = transformer.fit(train_data)
    # inverse scale the dataset back to its unscaled representation
    gen_data = inverse_transform_dataset(gen_data, transformer, dataset, data_scaler)

    if stftransform:
        gen_data = data_loading.pack_to_complex(gen_data)
        if fft_shift:
            gen_data = np.fft.ifftshift(gen_data, axes=(1,))
        gen_data = data_loading.stft_to_waveform(gen_data, nperseg=nperseg)
        gen_data = data_loading.unpack_complex(gen_data).view(float)
    if pad_length is not None and pad_length > 0:
        gen_data = data_loading.unpad_signal(gen_data, pad_length)
    assert gen_data.shape == targ_data.shape, f"Generated and Target test distributions " \
                                              f"are not the same: Gen {gen_data.shape} =/= " \
                                              f"Targ {targ_data.shape}"
    return targ_data, targ_labels, gen_data, n_samples


def test_gan(G_net, train_hist_df, output_path, device, gan_name, specs, data_scaler=None):
    """
    Evaluate performance of Generator and save performance metrics plots/ tables
    :param G_net: Generator model
    :param train_hist_df: Dataframe of batch-level training KPIs
    :param output_path: Path to save directory
    :param device: GPU device ID number
    :param gan_name: String name of GAN model
    :param specs: GAN configuraion dictionary
    :param data_scaler: Data scaler info used for inverse scaling
    :return: None
    """
    try:
        plot_losses(train_hist_df, gan_name, specs["save_model"], output_path)

        metric_dict = {}
        metric_dict["config"] = output_path
        dataset = specs["dataloader_specs"]["dataset_specs"]["data_set"]
        save = specs["save_model"]

        targ_data, targ_labels, gen_data, n_samples = load_test_distributions(specs, G_net, data_scaler, device)
        # Save generated distribution
        h5f = h5py.File(f'{output_path}/generated_distribution.h5', 'w')
        h5f.create_dataset('generates', data=gen_data)
        h5f.close()

        # save sklearn data scaling transformer to pkl file
        if data_scaler is not None:
            dump(data_scaler, open(f'{output_path}/target_data_scaler.pkl', 'wb'))

        with open(rf'./Datasets/{dataset}/ofdm_cls.json', 'r') as F:
            ofdm_params = json.loads(F.read())

        cp_corr_ratio = evalofdm.evaluate_cyclic_prefix(data_loading.pack_to_complex(gen_data),
                                                        data_loading.pack_to_complex(targ_data),
                                                        ofdm_params["cyclic_prefix"], ofdm_params["num_frames"],
                                                        output_path, True)
        metric_dict["cyclic_prefix_ratio"] = cp_corr_ratio
        _, _, _, gen_psd, gen_SNR_data, gen_EVM_data, gen_BER, gen_coher_BWs, gen_subinds = \
            evalofdm.evaluate_ofdm(gen_data, output_path, dataset, "Generated", save)
        fft = ofdm_params["symbol_length"]
        subcarrier_inds = gen_subinds[0]

        metric_dict["evm"] = gen_EVM_data[0]
        metric_dict["evm_quant_025"] = gen_EVM_data[1]
        metric_dict["evm_quant_975"] = gen_EVM_data[2]
        metric_dict["median_snr"] = gen_SNR_data[0]
        metric_dict["median_snr_quant_025"] = gen_SNR_data[1]
        metric_dict["median_snr_quant_975"] = gen_SNR_data[2]
        metric_dict["BER"] = gen_BER

        targ_psd = np.genfromtxt(rf'./Datasets/{dataset}/psd_vectors.csv', delimiter=",")
        if ofdm_params["channel3gpp"] is not None:
            targ_coher_BWs = np.genfromtxt(rf'./Datasets/{dataset}/coherence_bandwidths.csv', delimiter=",")

        gen_coher_BWs_mean, gen_coher_BWs_std = None, None
        targ_coher_BWs_mean, targ_coher_BWs_std = None, None
        if ofdm_params["channel3gpp"] is not None:
            evalofdm.Coherence_Bandwidth_plots(targ_coher_BWs, gen_coher_BWs, output_path, save)
            gen_coher_BWs_mean, gen_coher_BWs_std = np.mean(gen_coher_BWs), np.std(gen_coher_BWs)
            targ_coher_BWs_mean, targ_coher_BWs_std = np.mean(targ_coher_BWs), np.std(targ_coher_BWs)
            gen_coher_BWs = np.array(gen_coher_BWs)
            np.savetxt(rf'{output_path}/coherence_bandwidths.csv', gen_coher_BWs, delimiter=",")
        metric_dict["coher_bandw_mean"] = gen_coher_BWs_mean
        metric_dict["coher_bandw_std"] = gen_coher_BWs_std
        metric_dict["target_coher_bandw_mean"] = targ_coher_BWs_mean
        metric_dict["target_coher_bandw_std"] = targ_coher_BWs_std

        # PSD distance metrics and plotting
        evalofdm.plot_psd_distributions(targ_psd, gen_psd, output_path, psd_method="eigen")
        wholeband_dist, inband_dist, outband_dist, wholeband_dist_linear = \
            evalofdm.evaluate_spectrum_distance(targ_psd, gen_psd, subcarrier_inds, fft)
        metric_dict["relative_median_l2"] = wholeband_dist
        metric_dict["relative_median_l2_linear"] = wholeband_dist_linear
        metric_dict["inband_relative_median_l2"] = inband_dist
        metric_dict["outband_relative_median_l2"] = outband_dist

        with open(rf'{output_path}/distance_metrics.json', 'w') as F:
            F.write(json.dumps(metric_dict))

    except TypeError:
        print("eval broken")
        return True


def test_target_dist(dataset):
    """
    Evaluate target distribution and save various distributions for comparison to the generated distribution runs
    :param dataset: path to target distribution
    :return:
    """
    metric_dict = {}
    targ_data, targ_labels, _ = load_target(dataset, complex, "test")
    output_path = rf'./Datasets/{dataset}/'
    with open(rf'./Datasets/{dataset}/ofdm_cls.json', 'r') as F:
        ofdm_params = json.loads(F.read())
    print("Evaluating OFDM")
    _, _, _, targ_psd, targ_SNR_data, targ_EVM_data, targ_BER, targ_coher_BWs, targ_subinds = \
        evalofdm.evaluate_ofdm(targ_data, output_path, dataset, "Target", True)
    print("Finised eval")
    targ_psd = np.array(targ_psd)
    np.savetxt(rf'./Datasets/{dataset}/psd_vectors.csv', targ_psd, delimiter=",")
    if ofdm_params["channel3gpp"] is not None:
        targ_coher_BWs = np.array(targ_coher_BWs)
        np.savetxt(rf'./Datasets/{dataset}/coherence_bandwidths.csv', targ_coher_BWs, delimiter=",")

    targ_coher_BWs_mean, targ_coher_BWs_std = None, None
    if ofdm_params["channel3gpp"] is not None:
        targ_coher_BWs_mean, targ_coher_BWs_std = np.mean(targ_coher_BWs), np.std(targ_coher_BWs)
    metric_dict["target_coher_bandw_mean"] = targ_coher_BWs_mean
    metric_dict["target_coher_bandw_std"] = targ_coher_BWs_std
    metric_dict["target_evm"] = targ_EVM_data[0]
    metric_dict["target_evm_quant_025"] = targ_EVM_data[1]
    metric_dict["target_evm_quant_975"] = targ_EVM_data[2]
    metric_dict["target_median_snr"] = targ_SNR_data[0]
    metric_dict["target_median_snr_quant_025"] = targ_SNR_data[1]
    metric_dict["target_median_snr_quant_975"] = targ_SNR_data[2]
    metric_dict["target_BER"] = targ_BER
    with open(rf'{output_path}/target_metrics.json', 'w') as F:
        F.write(json.dumps(metric_dict))

    median_targ_spec = np.median(targ_psd, axis=0)
    spectrum_len = len(median_targ_spec)
    w = np.linspace(-0.5, 0.5, spectrum_len)
    plt.plot(w, np.fft.fftshift(median_targ_spec), color="black", alpha=0.85)
    plt.grid()
    plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=14)
    plt.ylabel("PSD (dB)", fontsize=14)
    plt.savefig(rf"{output_path}target_spectrums.png")
    plt.close('all')

    len_spec = len(w)
    inband_w = w[int(len_spec * 0.33): -int(len_spec * 0.33)]
    inband_median_targ = np.fft.fftshift(median_targ_spec)[int(len_spec * 0.33): -int(len_spec * 0.33)]
    plt.plot(inband_w, inband_median_targ, color="black", alpha=0.85)
    plt.grid()
    plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=14)
    plt.ylabel("PSD (dB)", fontsize=14)
    plt.savefig(rf"{output_path}inband_target_spectrums.png")
    plt.close('all')


def retest_gan(dir_path):
    """
    Re-test the evaluation of a trained GAN run (used when evaluation was
    updated after a model was already trained)
    :param dir_path: Path to saved GAN run
    :return:
    """
    save_bool = True
    with open(rf'{dir_path}distance_metrics.json', 'r') as F:
        metric_dict = json.loads(F.read())
    with open(dir_path + 'gan_train_config.json', 'r') as fp:
        train_specs_dict = json.loads(fp.read())
    dataset = train_specs_dict["dataloader_specs"]["dataset_specs"]["data_set"]
    print(f"Dataset: {dataset}")
    with open(rf'./Datasets/{dataset}/ofdm_cls.json', 'r') as F:
        ofdm_params = json.loads(F.read())
    targ_data, targ_labels, _ = load_target(dataset, complex, "test")
    h5f = h5py.File(f'{dir_path}generated_distribution.h5', 'r')
    gen_data = h5f['generates'][:]
    h5f.close()

    cp_corr_ratio = evalofdm.evaluate_cyclic_prefix(data_loading.pack_to_complex(gen_data),
                                                    data_loading.pack_to_complex(targ_data),
                                                    ofdm_params["cyclic_prefix"], ofdm_params["num_frames"],
                                                    dir_path, save_bool)
    metric_dict["cyclic_prefix_ratio"] = cp_corr_ratio

    _, _, _, gen_psd, gen_SNR_data, gen_EVM_data, gen_BER, gen_coher_BWs, gen_subinds = \
        evalofdm.evaluate_ofdm(gen_data, dir_path, dataset, "Generated", save_bool)

    fft = ofdm_params["symbol_length"]
    subcarrier_inds = gen_subinds[0]

    metric_dict["evm"] = gen_EVM_data[0]
    metric_dict["evm_quant_025"] = gen_EVM_data[1]
    metric_dict["evm_quant_975"] = gen_EVM_data[2]
    metric_dict["median_snr"] = gen_SNR_data[0]
    metric_dict["median_snr_quant_025"] = gen_SNR_data[1]
    metric_dict["median_snr_quant_975"] = gen_SNR_data[2]
    metric_dict["BER"] = gen_BER
    targ_psd = np.genfromtxt(rf'./Datasets/{dataset}/psd_vectors.csv', delimiter=",")
    gen_coher_BWs_mean, gen_coher_BWs_std = None, None
    targ_coher_BWs_mean, targ_coher_BWs_std = None, None
    if ofdm_params["channel3gpp"] is not None:
        targ_coher_BWs = np.genfromtxt(rf'./Datasets/{dataset}/coherence_bandwidths.csv', delimiter=",")
        evalofdm.Coherence_Bandwidth_plots(targ_coher_BWs, gen_coher_BWs, dir_path, save_bool)
        gen_coher_BWs_mean, gen_coher_BWs_std = np.mean(gen_coher_BWs), np.std(gen_coher_BWs)
        targ_coher_BWs_mean, targ_coher_BWs_std = np.mean(targ_coher_BWs), np.std(targ_coher_BWs)

        gen_coher_BWs = np.array(gen_coher_BWs)
        np.savetxt(rf'{dir_path}coherence_bandwidths.csv', gen_coher_BWs, delimiter=",")
    metric_dict["coher_bandw_mean"] = gen_coher_BWs_mean
    metric_dict["coher_bandw_std"] = gen_coher_BWs_std
    metric_dict["target_coher_bandw_mean"] = targ_coher_BWs_mean
    metric_dict["target_coher_bandw_std"] = targ_coher_BWs_std
    # PSD distance metrics and plotting
    evalofdm.plot_psd_distributions(targ_psd, gen_psd, dir_path, psd_method="eigen")
    wholeband_dist, inband_dist, outband_dist, wholeband_dist_linear = \
        evalofdm.evaluate_spectrum_distance(targ_psd, gen_psd, subcarrier_inds, fft)
    metric_dict["relative_median_l2"] = wholeband_dist
    metric_dict["relative_median_l2_linear"] = wholeband_dist_linear
    metric_dict["inband_relative_median_l2"] = inband_dist
    metric_dict["outband_relative_median_l2"] = outband_dist
    if save_bool:
        with open(rf'{dir_path}distance_metrics.json', 'w') as F:
            F.write(json.dumps(metric_dict))
