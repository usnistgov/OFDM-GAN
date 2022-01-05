# -*- coding: utf-8 -*-

import numpy as np
from data_loading import unpack_complex


def apply_cfo(iq_data, cfo, sample_rate):
    """
    Apply carrier frequency offset (CFO) impairment as described in Sec 2.5
    of (Smaini,2012).
    
    Parameters
    ----------
    iq_data : numpy arrary of complex iq data (samples x waveform length)
    cfo : carrier frequency offset (Hertz)
    sample_rate : digital sampling rate (samples/sec)

    Returns
    -------
    iq_cfo
    """        
    delta = cfo / sample_rate
    t = np.arange(iq_data.shape[1])
    iq_cfo  = iq_data * np.exp(1j * 2 * np.pi * delta * t)
    return iq_cfo

def apply_IQimbalance(iq_data, delta_G, delta_phi):
    """
    Apply IQ imblance impairement for transmitter 
    using equation (2.129) in (Smaini, 2012).
    
    Parameters
    ----------
    iq_data : numpy arrary of complex iq data (samples x waveform length)
    delta_G : gain imbalance
    delta_phi : phase imbalance
    

    Returns
    -------
    y : complex numpy array of impaired baseband signals (samples x waveform length)
    """
    
    # create transformation matrix
    a = 1 - delta_G / 2
    b = 1 + delta_G / 2
    cp = np.cos(delta_phi / 2)
    sp = np.sin(delta_phi / 2)
    A = np.array([1, 1j])
    B = np.array([[a * cp, b * sp], [a * sp, b * cp]])
    T = np.matmul(A, B)

    x = unpack_complex(iq_data)
    y = np.tensordot(T, x, (0, 1))
    return y
