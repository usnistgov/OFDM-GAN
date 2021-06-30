"""
Module contains 3GPP Rayleigh Fading-Channel Models taken from MATLAB implementation
"""

import numpy as np
from numpy import pi as pi


def apply3GPPchannel(x, sampling_rate, chan):
    """ 
    Apply 3GPP channel model to complex-valued input signal
    (author: Adam Wunderlich 9/5/2018)

    :param x: (complex numpy.ndarray) input signal
    :param sampling_rate: (float) sampling rate in Hz
    :param chan: (string) one of the strings 'EPA5Hz','EVA5Hz','EVA70Hz','ETU70Hz','ETU300Hz'
    :returns y: (complex numpy.ndarray) output signal
    """
    num_samples = np.max(x.shape)
    x = x.reshape(1, num_samples)
    sampling_period = 1 / sampling_rate
    if chan == 'EPA5Hz':
        chanType = 'EPA'
        doppler_freq_max = 5  # max doppler freq (Hz)
    elif chan == 'EVA5Hz':
        chanType = 'EVA'
        doppler_freq_max = 5
    elif chan == 'EVA70Hz':
        chanType = 'EVA'
        doppler_freq_max = 70
    elif chan == 'ETU70Hz':
        chanType = 'ETU'
        doppler_freq_max = 70
    elif chan == 'ETU300Hz':
        chanType = 'ETU'
        doppler_freq_max = 300
    else:
        raise ValueError('Unknown chan type.')
    tapDelays, pathGains = get3GPPchanParams(chanType, sampling_period)  # get channel parameters
    numTaps = len(tapDelays)
    H = np.zeros((numTaps, num_samples), dtype=complex)
    q = np.around(tapDelays / sampling_period).astype(int)  # tapDelay in samples
    # get filter coefficients
    for k in np.arange(numTaps):
        H[k, :] = np.sqrt(pathGains[k]) * RayleighProcess(num_samples, sampling_rate, doppler_freq_max)
    H = H / np.sqrt(np.sum(np.square(np.linalg.norm(H, axis=1, ord=2))) / num_samples)
    # apply channel to signal
    X = np.zeros((numTaps, num_samples), dtype=complex)
    for k in np.arange(numTaps):
        X[k, q[k]:] = x[0, :(num_samples - q[k])]
    y = np.sum(H * X, 0)  # Sum over each column. Result is 1 x num_samples
    return y


def get3GPPchanParams(chanType, Ts=10e-9):
    """
    Return 3GPP channel parameters from 3GPP TS 36.101 Annex B
    (author: Adam Wunderlich 9/5/2018)
    :param chanType: (string) one of 'EPA', 'EVA', or 'ETU')
    :param Ts: (float) desired sampling period
    returns td: (numpy.ndarray) tap delays in seconds
    returns pg: (numpy.ndarray) path gains in linear units
    """
    if chanType == 'EPA':
        tapDelays = np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9
        pathGains_dB = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
    elif chanType == 'EVA':
        tapDelays = np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]) * 1e-9
        pathGains_dB = np.array([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
    elif chanType == 'ETU':
        tapDelays = np.array([0, 50, 120, 200, 230, 500, 1600, 2300, 5000]) * 1e-9
        pathGains_dB = np.array([-1, -1, -1, 0, 0, 0, -3, -5, -7])
    else:
        raise ValueError('chanType incorrectly specified.')
    pathGains = 10 ** (pathGains_dB / 10)
    # To adjust for different sampling period,
    # use method described in 3GPP TR 25.943 Annex B
    td = np.empty(0)
    pg = np.empty(0)
    for k in np.arange(np.floor(tapDelays[-1] / Ts) + 1):
        ta = (k - .5) * Ts
        tb = (k + .5) * Ts
        IA = np.argwhere(tapDelays > ta)
        IB = np.argwhere(tapDelays <= tb)
        I = np.intersect1d(IA, IB)
        if I.size > 0:
            td = np.append(td, k * Ts)
            pg = np.append(pg, np.mean(pathGains[I]))
    pg = pg / np.amax(pg)  # normalize path gains
    return td, pg


def RayleighProcess(N, fs, fd):
    """
    Simulate a complex Rayleigh random process with classical Doppler spectrum
     using IDFT method of Young & Beaulieu, "The Generation of Correlated Rayleigh Random
     Variates by Inverse Discrete Fourier Transform", IEEE Trans. Comm., July 2000, pp. 1114-1127.
     Author: Adam Wunderlich 9/4/2018
    :param N: number of samples
    :param fs: sampling rate
    :param fd: max_doppler frequency (Hz)
    :return: length N complex-valued Rayleigh process
    """

    fm = fd / fs  # max doppler freq normalized by sampling rate
    km = np.floor(fm * N)  # freq index corresponding to max Doppler freq
    M = N  # fft length
    if km < 10:  # if km is too small, then increase fft length
        while km < 10:
            M = 2 * M
            km = np.floor(fm * M)  # freq index corresponding to max Doppler freq
    km = int(km)
    F_M = np.zeros(M)
    k1 = np.arange(1, km)
    F_M[k1] = np.sqrt(1 / (2 * np.sqrt(1 - (k1 / (M * fm)) ** 2)))
    F_M[km] = np.sqrt((km / 2) * (pi / 2 - np.arctan((km - 1) / np.sqrt(2 * km - 1))))
    F_M[M - km] = np.sqrt((km / 2) * (pi / 2 - np.arctan((km - 1) / np.sqrt(2 * km - 1))))
    k2 = np.arange(M - km + 1, M)
    F_M[k2] = np.sqrt(1. / (2 * np.sqrt(1 - ((M - k2) / (M * fm)) ** 2)))

    A = np.random.randn(M) * 0.7071067811865476
    B = np.random.randn(M) * 0.7071067811865476

    X = F_M * A - 1j * F_M * B
    x = np.fft.ifft(X)
    x = x[: N]  # return signal of desired length
    x = x.reshape(1, N)

    # scale to have unit mean power
    P = np.vdot(x, x) / N
    x = x / np.sqrt(P)
    return x
