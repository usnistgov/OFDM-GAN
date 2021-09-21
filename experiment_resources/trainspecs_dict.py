specs_dict = {'save_model': True,
              'eval_model': True,
              "checkpoint": None,
              "num_gpus": 4,
              "start_gpu": 0,
              'epochs': 500,
              "model_type": "stftgan",  # "wavegan", "stftgan", "pskgan"
              'D_updates': 1,  # 1, 5
              'weight_init': 'kaiming_normal',  # "kaiming_normal", "orthogonal"
              'stride': 2,  # 2, 4
              "kernel_size": 4,
              "latent_type": "uniform",  # "gaussian", "uniform
              "progressive_kernels": False,  # "standard, kernel, dilation"
              "model_levels": 4,
              'num_channels': 2,  # 1, 2
              "phase_shuffle": False,
              'dataloader_specs': {
                    'dataset_specs': {
                        'data_scaler': "feature_min_max",  # feature_min_max, feature_max_ab, global_min_max, global_max_ab
                        'data_set': "/512fft_rblocks2_subs150_evm-50_ETU300Hz",
                        'num_samples': 0,
                        "pad_signal": True,
                        'stft':  True,
                        'nperseg': 512,
                        'fft_shift': True},
                    'batch_size': 128},
              'optim_params': {
                    'D': {'lr': 0.0001, 'betas': (0.0, 0.9)},
                    'G': {'lr': 0.0001, 'betas': (0.0, 0.9)}},
              }


#%%

"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

gen_path = r"/home/jgs3/PycharmProjects/iqgan/experiment_results/data_throughput_experiment/STFT-GAN/256fft_subs74/2021-04-30_10-12-36/generated_distribution.h5"
h5f = h5py.File(gen_path, 'r')
gen_dataset = h5f['generates'][:]
h5f.close()
d_type = complex
data = np.array(gen_dataset[:, :]).astype(d_type)


waveform = data[0, :, :]
fig = plt.figure(figsize=(5, 6))
real, imag = waveform[0, :], waveform[1, :]
plt.plot(range(len(real)), real, color="blue", alpha=0.7)
plt.plot(range(len(imag)), imag, color="green", alpha=0.7)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.savefig("/home/jgs3/PycharmProjects/iqgan/experiment_results/waveform.png", bbox_inches='tight')
plt.show()

waveform = real + 1j * imag
f, t, stft = signal.stft(waveform, nperseg=256, noverlap=int(256 * 0.75), return_onesided=False)
periodogram = np.fft.fftshift(20 * np.log10(np.abs(stft)))
fig = plt.figure(figsize=(5, 6))
plt.imshow(periodogram, aspect='auto', extent=[0, 1920, -0.5, 0.5])
plt.xlabel("Time")
plt.ylabel("Frequency")
cbar = plt.colorbar()
cbar.set_label('Power (dB)')
plt.savefig("/home/jgs3/PycharmProjects/iqgan/experiment_results/spectrogram.png", bbox_inches='tight')
plt.show()
"""


#%%

# find ./ -name "" -exec rm -rf {} \;

# tar -cf experiment_results.tar.gz --use-compress-program=pigz experiment_results

