import scipy
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from numpy.fft import fftshift


grey_codes_2psk = {(0,): 0, (1,): 1}
grey_codes_4psk = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
grey_codes_8psk = {(0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 1): 2, (0, 1, 0): 3,
                   (1, 1, 0): 4, (1, 1, 1): 5, (1, 0, 1): 6, (1, 0, 0): 7}
grey_codes_16qam = {(0, 0, 0, 0): 0, (0, 0, 0, 1): 1, (0, 0, 1, 1): 2, (0, 0, 1, 0): 3,
                    (0, 1, 0, 0): 4, (0, 1, 0, 1): 5, (0, 1, 1, 1): 6, (0, 1, 1, 0): 7,
                    (1, 1, 0, 0): 8, (1, 1, 0, 1): 9, (1, 1, 1, 1): 10, (1, 1, 1, 0): 11,
                    (1, 0, 0, 0): 12, (1, 0, 0, 1): 13, (1, 0, 1, 1): 14, (1, 0, 1, 0): 15}
grey_codes_32qam = {(1, 0, 0, 0, 0): 0, (1, 0, 0, 1, 0): 1, (1, 1, 0, 1, 0): 2, (1, 1, 0, 0, 0): 3, (1, 0, 0, 1, 1): 4,
                    (0, 0, 0, 1, 1): 5, (0, 0, 0, 1, 0): 6, (0, 1, 0, 1, 0): 7, (0, 1, 0, 1, 1): 8, (1, 1, 0, 1, 1): 9,
                    (1, 0, 0, 0, 1): 10, (0, 0, 0, 0, 1): 11, (0, 0, 0, 0, 0): 12, (0, 1, 0, 0, 0): 13, (0, 1, 0, 0, 1): 14,
                    (1, 1, 0, 0, 1): 15, (1, 0, 1, 0, 1): 16, (0, 0, 1, 0, 1): 17, (0, 0, 1, 0, 0): 18, (0, 1, 1, 0, 0): 19,
                    (0, 1, 1, 0, 1): 20, (1, 1, 1, 0, 1): 21, (1, 0, 1, 1, 1): 22, (0, 0, 1, 1, 1): 23, (0, 0, 1, 1, 0): 24,
                    (0, 1, 1, 1, 0): 25, (0, 1, 1, 1, 1): 26, (1, 1, 1, 1, 1): 27, (1, 0, 1, 0, 0): 28, (1, 0, 1, 1, 0): 29,
                    (1, 1, 1, 1, 0): 30, (1, 1, 1, 0, 0): 31}
grey_codes_64qam = {(0, 0, 0, 0, 0, 0): 0, (0, 0, 1, 0, 0, 0): 1, (0, 1, 1, 0, 0, 0): 2, (0, 1, 0, 0, 0, 0): 3,
                    (1, 1, 0, 0, 0, 0): 4, (1, 1, 1, 0, 0, 0): 5, (1, 0, 1, 0, 0, 0): 6, (1, 0, 0, 0, 0, 0): 7,
                    (0, 0, 0, 0, 0, 1): 8, (0, 0, 1, 0, 0, 1): 9, (0, 1, 1, 0, 0, 1): 10, (0, 1, 0, 0, 0, 1): 11,
                    (1, 1, 0, 0, 0, 1): 12, (1, 1, 1, 0, 0, 1): 13, (1, 0, 1, 0, 0, 1): 14, (1, 0, 0, 0, 0, 1): 15,
                    (0, 0, 0, 0, 1, 1): 16, (0, 0, 1, 0, 1, 1): 17, (0, 1, 1, 0, 1, 1): 18, (0, 1, 0, 0, 1, 1): 19,
                    (1, 1, 0, 0, 1, 1): 20, (1, 1, 1, 0, 1, 1): 21, (1, 0, 1, 0, 1, 1): 22, (1, 0, 0, 0, 1, 1): 23,
                    (0, 0, 0, 0, 1, 0): 24, (0, 0, 1, 0, 1, 0): 25, (0, 1, 1, 0, 1, 0): 26, (0, 1, 0, 0, 1, 0): 27,
                    (1, 1, 0, 0, 1, 0): 28, (1, 1, 1, 0, 1, 0): 29, (1, 0, 1, 0, 1, 0): 30, (1, 0, 0, 0, 1, 0): 31,
                    (0, 0, 0, 1, 1, 0): 32, (0, 0, 1, 1, 1, 0): 33, (0, 1, 1, 1, 1, 0): 34, (0, 1, 0, 1, 1, 0): 35,
                    (1, 1, 0, 1, 1, 0): 36, (1, 1, 1, 1, 1, 0): 37, (1, 0, 1, 1, 1, 0): 38, (1, 0, 0, 1, 1, 0): 39,
                    (0, 0, 0, 1, 1, 1): 40, (0, 0, 1, 1, 1, 1): 41, (0, 1, 1, 1, 1, 1): 42, (0, 1, 0, 1, 1, 1): 43,
                    (1, 1, 0, 1, 1, 1): 44, (1, 1, 1, 1, 1, 1): 45, (1, 0, 1, 1, 1, 1): 46, (1, 0, 0, 1, 1, 1): 47,
                    (0, 0, 0, 1, 0, 1): 48, (0, 0, 1, 1, 0, 1): 49, (0, 1, 1, 1, 0, 1): 50, (0, 1, 0, 1, 0, 1): 51,
                    (1, 1, 0, 1, 0, 1): 52, (1, 1, 1, 1, 0, 1): 53, (1, 0, 1, 1, 0, 1): 54, (1, 0, 0, 1, 0, 1): 55,
                    (0, 0, 0, 1, 0, 0): 56, (0, 0, 1, 1, 0, 0): 57, (0, 1, 1, 1, 0, 0): 58, (0, 1, 0, 1, 0, 0): 59,
                    (1, 1, 0, 1, 0, 0): 60, (1, 1, 1, 1, 0, 0): 61, (1, 0, 1, 1, 0, 0): 62, (1, 0, 0, 1, 0, 0): 63}
modulation_grey_codes = {2: grey_codes_2psk, 4: grey_codes_4psk, 8: grey_codes_8psk,
                         16: grey_codes_16qam, 32: grey_codes_32qam, 64: grey_codes_64qam}


def constellation_symbols(mod_order=16):
    """
    Returns Modulation scheme corresponding to a set of standard modulation orders.
    Modulation orders 2, 4, and 8 are Phase-Shift Key modulated and 16, 32, and 64
    are Quadrature-Amplitude modulated

    :param  mod_order: constellation modulation order
    :returns s: IQ constellation symbol locations
    """
    # Modulation orders that are gridlike
    if mod_order in [4, 16, 32, 64]:
        mod_order = 36 if mod_order == 32 else mod_order
        base_grid = 2 * np.arange(np.sqrt(mod_order)) - np.sqrt(mod_order) + 1
        igrid, qgrid = np.meshgrid(base_grid, base_grid)
        symbols = igrid.reshape(mod_order) + 1j * qgrid.reshape(mod_order)  # noiseless symbols
        if mod_order == 36:
            max_amp = np.max(np.abs(symbols))
            symbols = [symbol for symbol in symbols if np.abs(symbol) < max_amp]
        s = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))  # RMS constellation normalized to unit power
    # Modulation orders that lie on the unit-circle (Phase-shift keying)
    if mod_order in [2, 8]:
        m = np.arange(0, mod_order)
        igrid = 1 / np.sqrt(2) * np.cos(m / mod_order * 2 * np.pi)
        qgrid = 1 / np.sqrt(2) * np.sin(m / mod_order * 2 * np.pi)
        symbols = igrid.reshape(mod_order) + 1j * qgrid.reshape(mod_order)  # noiseless symbols
        s = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))  # RMS constellation normalized to unit power
    return s


def bits_to_symbols(bitstream, symbol_locs):
    """
    Returns QAM symbol stream from bitstream

    :params  bitstream: data bitstream set to be modulated
    :params  symbol_locs: Constellation complex symbol locations

    :returns symbol_stream: data QAM symbol stream
    """
    mod_order = len(symbol_locs)
    grey_codes = modulation_grey_codes[mod_order]
    symbol_stream = np.array([symbol_locs[grey_codes[tuple(bit_set)]] for bit_set in bitstream])
    return symbol_stream


def get_subcarrier_inds(start_rb, num_rbs, fft_len, total_subs):
    """
    Get FFT indicies for active subcarriers given channel badnwidth, number of resource blocks and placement
    :param start_rb:  start resource block index
    :param num_rbs: numnber of contiguous resource blocks
    :param fft_len: length of FFT used to modulate data subcarriers
    :param total_subs: number of active subcarriers
    :returns active_inds IQ constellation symbol locations
    """
    total_rbs = total_subs // 12
    fft_inds = list(range(fft_len))  # total fft indices
    guard_len = (fft_len - total_subs) // 2

    band_inds = fft_inds[guard_len: -guard_len]  # band subcariers where data can be modulated
    band_inds = np.array_split(band_inds, total_rbs)
    active_rbs = list(range(start_rb, start_rb + num_rbs))

    active_inds = [band_inds[i] for i in active_rbs]
    active_inds_list = [ind for sublist in active_inds for ind in sublist]
    active_inds = np.zeros(fft_len)
    active_inds[active_inds_list] = 1

    # index subcarriers in fft indices while dc-frequency centered
    # inverse fft shift to get to default fft indexing
    active_inds = scipy.fftpack.ifftshift(active_inds)
    active_inds = np.argwhere(active_inds).flatten()
    return active_inds


class OFDMA:
    """
    A class to modulate/demodulate OFDM (Orthogonal Frequency Division Multiplexing) synthetic
    waveforms based on LTE 3GPP standards

    Methods
    -------
        mod(symbols)
            modulate bitstream to complex OFDM time-series signal
        demod(time_signal, equalize=True)
            Demodulate complex OFDM time-series signal and apply channel equalization
        OFDM_to_SCFDMA(symbols)
            Apply DFT to OFDM symbols to transform to SC-FDMA symbols
        SCFDMA_to_OFDM(symbols)
            Apply iDFT to SC-FDMA symbols to transform to OFDM symbols
    """

    def __init__(self, num_subcarriers=16, num_OFDM_symbols=1, symbol_length=128, cyclic_prefix=[0, 0],
                 subcarrier_inds=None, scfdma=False, channel_3gpp=None, num_frames=1, normalize=True):
        """
        Constructs all the necessary attributes for the OFDM object.
        :param num_subcarriers: int number of subcarriers to allocate data symbols
        :param num_OFDM_symbols: int number of OFDM symbols in a resource block
        :param symbol_length:int fft length corresponding to channel bandwidth
        :param cyclic_prefix: tuple time-length of cyclic prefix in seconds
        :param subcarrier_inds: array of occupied subcarrier indicies
        :param scfdma: bool convert OFDM symbols into Single-carrier FDMA
        :param channel_3gpp: str name of 3GPP channel model applied to modulated waveforms
        :param num_frames: int number of LTE Subframes [Must be a multiple of 2].
        :param normalize: bool normalize OFDM fft
        """
        self.num_subcarriers = num_subcarriers
        self.num_OFDM_symbols = num_OFDM_symbols
        self.fft_length = symbol_length
        self.scfdma = scfdma
        self.channel_3gpp = channel_3gpp
        self.num_frames = num_frames
        self.normalize = normalize
        self.normalization = np.sqrt(self.fft_length / num_subcarriers)
        self.first_cyclic_prefix_length = cyclic_prefix[0]
        self.cyclic_prefix_length = cyclic_prefix[1]
        self.waveform_length = self.num_frames * (self.fft_length + cyclic_prefix[0] +
                                                  ((self.num_OFDM_symbols - 1) * (self.fft_length + cyclic_prefix[1])))
        if subcarrier_inds is None:
            block_1 = [i + 1 for i in range(num_subcarriers // 2)]
            block_2 = [self.fft_length - i - 1 for i in range(num_subcarriers // 2)]
            self.subcarrier_indices = block_1 + block_2
            self.subcarrier_indices = [0] if num_subcarriers == 1 else self.subcarrier_indices
        else:
            self.subcarrier_indices = subcarrier_inds
            self.num_subcarriers = len(subcarrier_inds)

    def mod(self, symbols):
        """
        modulate bitstream to complex OFDM time-series signal
        :param symbols: numpy array of complex IQ symbols (shape: [num_OFDM_symbols, num_subcarriers])
        :return: OFDM complex time-series
        """

        symbols = symbols.reshape(self.num_OFDM_symbols * self.num_frames, self.num_subcarriers)
        frequency_domain = np.zeros((self.num_OFDM_symbols * self.num_frames, self.fft_length), dtype=np.complex128)

        frequency_domain[:, self.subcarrier_indices] = symbols
        # Add DRS Zadoff-Chu base sequence if a channel model is going to be applied to the signal
        if self.channel_3gpp is not None:
            # Add DRS sequence to each 4th OFDM symbol of every subframe
            for frame in range(self.num_frames):
                zc_seqs = get_ZC_base_seqs(num_subcarriers=self.num_subcarriers)
                # seq_ind = np.random.randint(0, len(zc_seqs) - 1)
                seq_ind = 0
                base_sequence = zc_seqs[seq_ind]
                frequency_domain[int(frame * self.num_OFDM_symbols) + 3, self.subcarrier_indices] = base_sequence

        # Apply OFDM FFT to complex subcarrier values to get time-domain waveform
        if self.normalize:
            time_signal = np.fft.ifft(frequency_domain, axis=1, norm="ortho") * self.normalization
        else:
            time_signal = np.fft.ifft(frequency_domain, axis=1, norm="ortho")

        # Add Cyclic prefix to each OFDM symbol, taking subframes into account seperately
        signal_list = []
        for subframe_time_signal in np.split(time_signal, self.num_frames):
            # Add cyclic prefixes seperately if "short" cyclic prefix setting (first cp is slightly longer)
            if self.cyclic_prefix_length != self.first_cyclic_prefix_length:
                first_symbol = np.hstack((subframe_time_signal[0, -self.first_cyclic_prefix_length::], subframe_time_signal[0, :]))
                remaining_symbols = np.hstack((subframe_time_signal[1:, -self.cyclic_prefix_length::], subframe_time_signal[1:, :]))
                remaining_symbols, first_symbol = remaining_symbols.reshape(-1, 1), first_symbol.reshape(-1, 1)
                subframe_time_signal = np.concatenate((first_symbol, remaining_symbols))
            # Add cyclic prefixes for "long" cyclic prefix setting
            else:
                subframe_time_signal = np.hstack((subframe_time_signal[:, -self.cyclic_prefix_length::], subframe_time_signal))
            subframe_time_signal = subframe_time_signal.reshape(1, -1)
            signal_list.append(subframe_time_signal)
        time_signal = np.hstack(signal_list)
        new_time_signal = time_signal.reshape(1, -1)
        return new_time_signal

    def demod(self, time_signal, equalize=True):
        """
        Demodulate complex OFDM time-series signal and apply channel equalization
        :param time_signal: OFDM complex time-series
        :param equalize: Equalize channel effect relative to DRS sequence
        :return: numpy array of complex IQ symbols (shape: [num_OFDM_symbols, num_subcarriers])
        """
        # Remove extra time-steps from first cyclic prefix if "short" cp setting is on
        if self.cyclic_prefix_length != self.first_cyclic_prefix_length:
            extra_timesteps = self.first_cyclic_prefix_length - self.cyclic_prefix_length
            if self.num_frames > 1:
                time_signal = time_signal.reshape(self.num_frames, -1)
                time_signal = time_signal[:, extra_timesteps:]
                time_signal = time_signal.reshape(-1, )
            else:
                time_signal = time_signal[extra_timesteps:]
        time_signal = time_signal.reshape(self.num_frames * self.num_OFDM_symbols, self.fft_length + self.cyclic_prefix_length)
        if self.normalize:
            frequency_domain = np.fft.fft(time_signal[:, -self.fft_length::], axis=1, norm="ortho") / self.normalization
        else:
            frequency_domain = np.fft.fft(time_signal[:, -self.fft_length::], axis=1, norm="ortho")

        frequency_domain = frequency_domain[:, self.subcarrier_indices]
        # If a channel model is applied to the waveforms: Get DRS Sequences and equalize sub-carriers if equalization is turned on
        if self.channel_3gpp is not None:
            frame_frequency_domains = np.split(frequency_domain, self.num_frames)
            frequency_domain = []
            drs_seqs = []
            for frame_frequency_domain in frame_frequency_domains:
                received_zc_seq = frame_frequency_domain[3, :]
                drs_seqs.append(received_zc_seq)
                frame_frequency_domain = frame_frequency_domain[np.arange(len(frame_frequency_domain)) != 3, :]
                if equalize:
                    channel_est = get_channel_effect(received_zc_seq)
                    frame_frequency_domain = frame_frequency_domain / channel_est
                frequency_domain.append(frame_frequency_domain)
            frequency_domain = np.concatenate(frequency_domain)
            drs_seqs = np.array(drs_seqs).reshape(1, -1)
            return frequency_domain, drs_seqs
        return frequency_domain, None

    def OFDM_to_SCFDMA(self, symbols):
        """
        Apply DFT to OFDM symbols to transform to SC-FDMA symbols
        :param symbols: numpy array of complex IQ symbols (shape: [num_OFDM_symbols, num_subcarriers])
        :return: numpy array of complex IQ symbols (shape: [num_OFDM_symbols, num_subcarriers])
        """
        symbols = symbols.reshape(self.num_OFDM_symbols * self.num_frames, self.num_subcarriers)
        scfdma_symbols = np.fft.fft(symbols, axis=1, norm="ortho")
        return scfdma_symbols

    def SCFDMA_to_OFDM(self, scfmda_symbols):
        """
        Apply iDFT to SC-FDMA symbols to transform to OFDM symbols
        :param scfmda_symbols: numpy array of complex SCFDMA symbols (shape: [num_OFDM_symbols, num_subcarriers])
        :return: numpy array of complex OFDM symbols (shape: [num_OFDM_symbols, num_subcarriers])
        """
        scfmda_symbols = scfmda_symbols.reshape(self.num_OFDM_symbols * self.num_frames, self.num_subcarriers)
        symbols = np.fft.ifft(scfmda_symbols, axis=1, norm="ortho")
        return symbols


def demodulate_ofdm_waveforms(data_complex, ofdm_params, equalize=False):
    """
    Demodulate distribution of OFDM waveforms based on OFDM set parameters
    :param data_complex: distribution of complex valued OFDM waveforms
    :param ofdm_params: dictionary of OFDM set parameters
    :param equalize: equalize distortion of subcarriers if waveforms were passed through a channel model
    :return: tuple of demodulated OFDM symbols and related metadata
    """
    mod_order = ofdm_params["mod_order"]
    num_frames = ofdm_params["num_frames"]
    channel_model = ofdm_params["channel3gpp"]

    symbol_locs = constellation_symbols(mod_order=mod_order)
    grey_codes = modulation_grey_codes[mod_order]
    grey_codes = {v: k for k, v in grey_codes.items()}

    # OFDMA modulator-demodulator object
    ofdm = OFDMA(num_subcarriers=ofdm_params["num_subcarriers"], num_OFDM_symbols=ofdm_params["num_OFDM_symbols"],
                 symbol_length=ofdm_params["symbol_length"], cyclic_prefix=ofdm_params["cyclic_prefix"],
                 channel_3gpp=channel_model, num_frames=num_frames, normalize=True)

    demod_data_symbols, demod_closest_data_symbols, demod_drs_seqs, subcarrier_indices, bitstreams = [], [], [], [], []

    # demodulate each recieved OFDM signal separately
    for i, waveform in enumerate(data_complex):
        if i % (len(data_complex) // 10) == 0:
            print(i, end=", ")

        # demodulate recieved OFDM signal
        demod_ofdm_symbol, received_drs_sequence = ofdm.demod(waveform, equalize)
        subcarrier_indices.append(ofdm.subcarrier_indices)

        # Save DMRS sequences if the waveform was propagated through a channel
        if channel_model:
            demod_drs_seqs.append(received_drs_sequence)

        # Save demodulated OFDM Symbols with data subcarriers (i.e. remove the DRS OFDM Symbol)
        demod_ofdm_data_symbol = demod_ofdm_symbol.reshape(-1, )
        if ofdm_params["scfdma"]:
            demod_ofdm_data_symbol = ofdm.SCFDMA_to_OFDM(demod_ofdm_data_symbol).reshape(-1, )

        # compute closest symbol, symbol stream, and bitstream
        closest_symbol_nums = np.vectorize(lambda demod_sym: np.argmin([np.abs(demod_sym - symbol) for
                                                                        symbol in symbol_locs]))(demod_ofdm_data_symbol)
        closest_symbols = np.vectorize(lambda symbol: symbol_locs[symbol])(closest_symbol_nums)
        bitstream = np.concatenate([grey_codes[symbol] for symbol in closest_symbol_nums])
        demod_data_symbols.append(demod_ofdm_data_symbol)
        demod_closest_data_symbols.append(closest_symbols)
        bitstreams.append(bitstream)

    demod_data_symbols = np.array(demod_data_symbols)
    demod_closest_data_symbols = np.array(demod_closest_data_symbols)
    bitstreams = np.array(bitstreams)
    return demod_data_symbols, demod_closest_data_symbols, demod_drs_seqs, bitstreams, subcarrier_indices


def get_ZC_base_seqs(num_subcarriers):
    """
    Compute the Zadoff-Chu base sequences based on the q-root values
    :param num_subcarriers: length of ZC sequence
    :return: ZC base sequences
    """
    N_zc = next_prime_bellow(num_subcarriers)
    rem = num_subcarriers - N_zc
    q_roots = get_q_roots(N_zc)
    base_seqs = []
    for q in q_roots:
        seq = [np.exp(-1j * np.pi * q * m * (m + 1) / N_zc) for m in range(N_zc)]
        seq = seq + seq[:rem]
        base_seqs.append(seq)
    return base_seqs


def get_channel_effect(received_drs, fixed_dmrs=True):
    """
    Calcukate the estimated channel effect based on the Received and known DMRS sequences
    :param received_drs: distribution of received DMRS sequences
    :param fixed_dmrs: bool for whether to use the 0th base sequence or to estimate from all possible DMRS base sequences
    :return:
    """
    num_subcarriers = len(received_drs)
    zc_seqs = get_ZC_base_seqs(num_subcarriers)
    if fixed_dmrs:
        max_ind = 0
    # Estimate the DMRS base-sequence (no longer done do to inaccurate estimates when the channel is extreme)
    else:
        seq_corrs = []
        for i, zc_seq in enumerate(zc_seqs):
            corr = np.abs(signal.correlate(received_drs, zc_seq))
            max_phase_corr = np.max(np.abs(corr))
            seq_corrs.append(max_phase_corr)
        max_ind = np.argmax(seq_corrs)
    best_zc_seq = zc_seqs[max_ind]
    # Get estimated phase rotation:
    channel_est = received_drs / best_zc_seq
    return channel_est


def next_prime_bellow(n):
    """
    Compute factorials and apply Wilson's theorem:  Any number n is a prime number if,
    and only if, (n − 1)! + 1 is divisible by n.
    :param n: integer number
    :return: next lower prime bellow n
    """
    fact = 1
    primes = []
    for k in range(2, n):
        fact = fact * (k - 1)
        if (fact + 1) % k == 0:
            primes.append(k)
    max_prime = np.max(primes)
    return max_prime


def get_q_roots(N_zc):
    """
    Get the Q-roots that define the Zadoff-Chu Base sequences
    :param N_zc:
    :return:
    """
    groups = range(30)
    bases = range(2)
    if N_zc < 60:
        bases = [0]
    q_roots = []
    for u in groups:
        for v in bases:
            q_bar = N_zc * (u + 1) / 31
            q = np.floor(q_bar + 0.5) + v * -1 ** np.floor(2 * q_bar)
            q_roots.append(q)
    return q_roots


def create_fixed_bistream(qam_order=16, traffic_setting="high_traffic", fft=512, cp_type="long", num_resource_blocks=2):
    """
    Create fixed random bitstream used for all channel model-applied datasets
    :param qam_order: QAM Modulation order
    :param traffic_setting: Allocated subcarrier traffic setting
    :param fft: length of FFT used to modulate subcarriers
    :param cp_type: Cyclic-prefix type
    :param num_resource_blocks: Number of RBs allocated sequentially in time
    :returns fixed_bitstreams: dictionary with single fixed random bitstream
    """
    grey_codes = modulation_grey_codes[qam_order]
    grey_codes = {v: k for k, v in grey_codes.items()}
    num_ofdm_symbols = {"short": 7, "long": 6}[cp_type]
    occupancy = {"low_traffic": 75 / 512, "mid_traffic": 150 / 512, "high_traffic": 225 / 512}[traffic_setting]
    num_subcarriers = int(occupancy * fft)
    num_symbols = num_subcarriers * num_ofdm_symbols * num_resource_blocks
    symbol_stream = np.random.randint(0, qam_order, num_symbols)
    bit_stream = [grey_codes[symbol_num] for symbol_num in symbol_stream]
    bitstream = np.concatenate(bit_stream)
    fixed_bistreams = {"fixed": list(bitstream)}
    return fixed_bistreams
