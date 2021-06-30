"""
This module holds pytorch related data loading methods and data
preprocessing methods related to target data representation.

This file can also be imported as a module and contains the following functions:
    * unpack_complex - unpack complex waveform to two-channel real-valued waveform
    * pack_to_complex - pack two-channel real-valued waveform to complex waveform
    * scale_dataset - scale target distribution to range [-1, 1]
    * load_target_distribution - load target training distribution from files
    * TargetDataset - wrapper for dataset for ease of loading into pytorch framework
    * build_DataLoader - create PyTorch DataLoader object
    * get_latent_vectors - load batch of latent vectors for input to generator
    * stft_to_waveform - convert complex STFT representation to complex waveform
    * waveform_to_stft - convert complex waveform to complex STFT
    * pad_signal_to_power_of_2 - zero-pad waveform to next power of 2
    * unpad_signal - remove zero-padding from waveform
"""

import h5py
import json
import torch
import numpy as np
from scipy import signal
from numpy.fft import fftshift
from sklearn import preprocessing
from scipy.stats import truncnorm
from torch.utils.data import Dataset


def unpack_complex(iq_data):
    """
    Convert complex 2D matrix to 3D matrix with 2 channels for real and imaginary dimensions
    :param iq_data: numpy complex matrix (2D)
    :return: numpy floating point matrix (3D)
    """
    iq_real = iq_data.real
    iq_imaginary = iq_data.imag
    iq_real = np.expand_dims(iq_real, axis=1)    # Make dataset 3-dimensional to work with framework
    iq_imaginary = np.expand_dims(iq_imaginary, axis=1)    # Make dataset 3-dimensional to work with framework
    unpacked_data = np.concatenate((iq_real, iq_imaginary), 1)
    return unpacked_data


def pack_to_complex(iq_data):
    """
     convert 3D matrix with 2 channels for real and imaginary dimensions to 2D complex representation
    :param iq_data: numpy floating point matrix (3D)
    :return:  numpy complex matrix (2D)
    """
    num_dims = len(iq_data.shape)
    if num_dims == 2:
        complex_data = 1j * iq_data[:, 1] + iq_data[:, 0]
    elif num_dims == 3:
        complex_data = 1j * iq_data[:, 1, :] + iq_data[:, 0, :]
    else:
        complex_data = 1j * iq_data[:, 1, :, :] + iq_data[:, 0, :, :]
    return complex_data


def scale_dataset(data, data_set, data_scaler):
    """
    Scale target distribution's range to [-1, 1] with multiple scaling options
    :param data: Target distribution
    :param data_set: dataset name
    :param data_scaler: data-scaler setting
    :return: scaled target distribution
    """
    if data_scaler == "activation_scaler":
        return data, None

    # Feature Based data scaling:
    if data_scaler.find("feature") != -1:
        print(f"feature Based Scaling: {data_scaler}")
        data_shape = data.shape
        data = data.reshape(data_shape[0], -1)
        transformer = preprocessing.MaxAbsScaler() if data_scaler == "feature_max_abs" \
            else preprocessing.MinMaxScaler(feature_range=(-1, 1))
        transformer = transformer.fit(data)
        data = transformer.transform(data)
        data = data.reshape(data_shape)
        return data, transformer

    # Global Dataset scaling:
    elif data_scaler.find("global") != -1:
        transformer = None
        with open(rf'./Datasets/{data_set}/scale_factors.json', 'r') as F:
            channel_scale_factors = json.loads(F.read())
        channel_max = channel_scale_factors["max"]
        channel_min = channel_scale_factors["min"]
        if data_scaler == "global_min_max":
            feature_max, feature_min = 1, -1
            data = (data - channel_min) / (channel_max - channel_min)
            data = data * (feature_max - feature_min) + feature_min
        else:
            data = data / np.max(np.abs([channel_max, channel_min]))
        return data, transformer


def load_target_distribution(data_set, data_scaler, pad_signal, num_samples, stft, nperseg, fft_shift):
    """
    Load in target distribution, scale data to [-1, 1], and unpack any labels from the data
    :param fft_shift: Shift STFT to be zero-frequency centered
    :param nperseg: STFT FFT window length
    :param stft: Convert complex waveform to STFT
    :param num_samples: Number of samples to load from the target distribution
    :param pad_signal: Length of zero padding target distribution waveforms
    :param data_set: Name of dataset
    :param data_scaler: Name of scaling function option
    :return: PyTorch tensors
    """
    d_type = complex
    h5f = h5py.File(rf"./Datasets/{data_set}/train.h5", 'r')
    real_dataset = h5f['train'][:]
    print("Dataset_length: ", len(real_dataset))
    h5f.close()
    data = np.array(real_dataset[:, 1:]).astype(d_type)
    class_labels = np.real(real_dataset[:, 0]).astype(np.int)

    if int(num_samples) > 64:
        data = data[:num_samples]
        class_labels = class_labels[:num_samples]

    input_length = len(data[0, :])
    pad_length = None
    if pad_signal and not stft:
        # WaveGAN uses strides of 4 so waveforms are padded to be powers of 2
        data, pad_length = pad_signal_to_power_of_2(data)
        input_length = pad_length + input_length
    if stft:
        data, pad_length = pad_signal_to_power_of_2(data)
        data, f, t = waveform_to_stft(data, 2, nperseg)
        if fft_shift:
            data = np.fft.fftshift(data, axes=(1,))

        input_length = (nperseg, data.shape[-1])
        data = data.reshape(data.shape[0], nperseg, -1)
    data = data.view(complex)
    data = unpack_complex(data).view(float)   # Unpacking complex-representation to 2-channel representation

    data = np.expand_dims(data, axis=1) if len(data.shape) < 3 else data
    data, transformer = scale_dataset(data, data_set, data_scaler)
    data = torch.from_numpy(data).float()
    class_labels = torch.from_numpy(class_labels).float()
    return data, class_labels, input_length, pad_length, transformer


class TargetDataset(Dataset):
    """
    Wrapper for dataset that can be easily loaded and used for training through PyTorch's framework.
    Pairs a training example with its label in the format (training example, label)
    """
    def __init__(self, data_set, data_scaler, pad_signal, num_samples, stft=False, nperseg=0, fft_shift=False, **kwargs):
        self.dataset, self.labels, self.input_length, self.pad_length, self.transformer = \
            load_target_distribution(data_set, data_scaler, pad_signal, num_samples, stft, nperseg, fft_shift)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


def build_DataLoader(dataset_specs):
    """
    Creates new Dataset, Sampler, and DataLoader using train_specs_dict. data-factors are
    specified by dataset-specs dictionary
    :param dataset_specs: dictionary defining data-specific
    :return: DataLoader
    """
    dataset = TargetDataset(**dataset_specs)
    sampler = None
    return dataset, sampler


def get_latent_vectors(batch_size, latent_size, latent_type="gaussian", device="cuda:0"):
    """
    Load latent space variables and fake labels used for Generator
    :param latent_type: Uniform or Gaussian latent distribution
    :param batch_size: length of batch
    :param latent_size: lantent space random seed variable dimension
    :param device: nvidia-device object
    :return: latent variable pytorch-tensor and fake class labels
    """
    if latent_type == "gaussian":
        z = torch.randn(batch_size, latent_size, 1, device=device)
    elif latent_type == "uniform":
        z = torch.from_numpy(np.random.uniform(low=-1.0, high=1.0, size=(batch_size, latent_size, 1))).float().to(device)
    else:
        truncate = 1.0
        lower_trunc_val = -1 * truncate
        z = []  # assume no correlation between multivariate dimensions
        for dim in range(latent_size):
            z.append(truncnorm.rvs(lower_trunc_val, truncate, size=batch_size))
        z = np.transpose(z)
        z = torch.from_numpy(z).unsqueeze(2).float().to(device)
    return z


def stft_to_waveform(dataset, fs=2, nperseg=64):
    """
    Transform Short-Time-Fourier-Transform (STFT) representation to complex waveform
    :param dataset: STFT Dataset
    :param fs: Sampling frequency (Hz)
    :param nperseg: N-Per-Segment Window length
    :return: Complex waveform dataset
    """
    waveform_dataset = []
    print("Mapping STFT dataset to timeseries:", end=" ")
    for i, spectrogram in enumerate(dataset):
        if i % 10000 == 0:
            print(i)
        t, x = signal.istft(spectrogram, fs, nperseg=nperseg, noverlap=int(nperseg * 0.75), input_onesided=False)
        waveform_dataset.append(x)
    waveform_dataset = np.array(waveform_dataset, dtype=complex)
    return waveform_dataset


def waveform_to_stft(dataset, fs=2, nperseg=64):
    """
    Convert complex waveform representation to Transform Short-Time-Fourier-Transform (STFT) representation
    :param dataset: Complex waveform dataset
    :param fs: sampling frequency (Hz)
    :param nperseg: N-per-segment window length
    :return: STFT Dataset
    """
    stft_dataset = []
    print("Mapping timeseries dataset to stft")
    for i, x in enumerate(dataset):
        if i % 10000 == 0:
            print(i)
        f, t, spectrogram = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=int(nperseg * 0.75),
                                        return_onesided=False, boundary="even")
        stft_dataset.append(spectrogram)
    stft_dataset = np.array(stft_dataset, dtype=complex)
    return stft_dataset, f, t


def pad_signal_to_power_of_2(waveform_dataset):
    """
    Add zero padding to signal to nearest power of 2
    :param waveform_dataset: Target Distribution
    :return: zero-padded target distribution, zero-padding length
    """
    waveform_length = waveform_dataset.shape[-1]
    d_type = complex
    found = False
    test_int = waveform_length
    next_power_of_2 = None
    while found is False:
        if test_int & (test_int - 1) == 0:
            found = True
            next_power_of_2 = test_int
        else:
            test_int += 1
    pad_length = next_power_of_2 - waveform_length
    padding_array_1 = np.zeros((len(waveform_dataset), pad_length // 2)).astype(d_type)
    padding_array_2 = np.zeros((len(waveform_dataset), pad_length // 2)).astype(d_type)
    padding_array_1, padding_array_2 = padding_array_1 + 1e-8, padding_array_2 + 1e-8
    waveform_dataset = np.hstack((padding_array_1, waveform_dataset, padding_array_2))
    return waveform_dataset, pad_length


def unpad_signal(waveform_dataset, pad_length):
    """
    Remove zero-padding of signal
    :param waveform_dataset: zero-padded dataset
    :param pad_length: length of zero-padding
    :return: waveform dataset
    """
    if pad_length > 0:
        waveform_dataset = waveform_dataset[:, :, pad_length // 2: - pad_length // 2]
        return waveform_dataset
    else:
        return waveform_dataset
