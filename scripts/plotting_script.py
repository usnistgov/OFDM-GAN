import glob
import json
import h5py
import numpy as np
import pandas as pd
import data_loading
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.synthetic_ofdm_modem import constellation_symbols


#%%

df_temp = []
dir_path = "./experiment_results/data_throughput_experiment/*/*/*/distance_metrics.json"
dataset_fft_dict = {'256fft_subs36': 256,
                    '128fft_subs56': 128,
                    '256fft_subs112': 256,
                    '512fft_subs224': 512,
                    '128fft_subs36': 128,
                    '128fft_subs18': 128,
                    '256fft_subs74': 256,
                    '512fft_subs150': 512,
                    '512fft_subs74': 512}
dataset_rank_dict = {'256fft_subs36': 4,
                    '128fft_subs56': 3,
                    '256fft_subs112': 6,
                    '512fft_subs224': 9,
                    '128fft_subs36': 2,
                    '128fft_subs18': 1,
                    '256fft_subs74': 5,
                    '512fft_subs150': 8,
                    '512fft_subs74': 7}

dist_paths = glob.glob(dir_path)
print(dist_paths)
for path in dist_paths:
    print(path)
    with open(path) as f:
        data = json.load(f)
        data["config"] = path
        data["dataset"] = data["config"].split("/")[4]
        data["model"] = data["config"].split("/")[3]
        data["timestamp"] = data["config"].split("/")[5]
        df_temp.append(data)
df = pd.DataFrame(df_temp)
df["fft"] = df["dataset"].map(dataset_fft_dict)
df["rank"] = df["dataset"].map(dataset_rank_dict)
print(df["model"].unique())
df.to_csv("./experiment_results/data_throughput_experiment/full_results.csv", index=False)
#%%

metrics_df = pd.read_csv("./experiment_results/data_throughput_experiment/full_results.csv")
models = ['WaveGAN', "PSK-GAN", 'STFT-GAN']
colors = ["blue", "green", "red"]

fig = plt.figure(figsize=(5, 10))
gs = gridspec.GridSpec(ncols=1, nrows=3, wspace=0.0, hspace=0.02)
ax = plt.subplot(gs[0, 0])
for model, color in zip(models, colors):
    model_metrics_df = metrics_df[metrics_df["model"] == model]
    num_samples = len(model_metrics_df["rank"])
    jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
              -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
              -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1]
    plt.plot(model_metrics_df["rank"] + jitter, model_metrics_df["geodesic_psd_distance"], color=color,
             alpha=0.5, linestyle="", marker="o")
    print(model_metrics_df["geodesic_psd_distance"])
    model_metrics_avg = model_metrics_df.groupby(["fft", "rank"]).mean().reset_index()

    metrics_128 = model_metrics_avg[model_metrics_avg["fft"] == 128]
    metrics_256 = model_metrics_avg[model_metrics_avg["fft"] == 256]
    metrics_512 = model_metrics_avg[model_metrics_avg["fft"] == 512]

    plt.plot(metrics_128["rank"], metrics_128["geodesic_psd_distance"], color=color, linestyle="-", alpha=0.75)
    plt.plot(metrics_256["rank"], metrics_256["geodesic_psd_distance"], color=color, linestyle="-", alpha=0.75)
    plt.plot(metrics_512["rank"], metrics_512["geodesic_psd_distance"], color=color, linestyle="-", alpha=0.75, label=model)

plt.xticks(range(1, 10), [r"128-Small", "128-Medium", "128-Large", "256-Small", "256-Medium", "256-Large",
                      "512-Small", "512-Medium", "512-Large"], rotation=45)

plt.ylabel(r"PSD Geodesic Distance ($d_g$)")
plt.grid(True)
ax.set_xticklabels([])


ax = plt.subplot(gs[1, 0])
for model, color in zip(models, colors):
    model_metrics_df = metrics_df[metrics_df["model"] == model]
    num_samples = len(model_metrics_df["rank"])
    jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
              -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
              -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1]
    plt.plot(model_metrics_df["rank"]+jitter, model_metrics_df["evm"], color=color, linestyle="",
             alpha=0.5, marker="o")
    model_metrics_avg = model_metrics_df.groupby(["fft", "rank"]).mean().reset_index()
    metrics_128 = model_metrics_avg[model_metrics_avg["fft"] == 128]
    metrics_256 = model_metrics_avg[model_metrics_avg["fft"] == 256]
    metrics_512 = model_metrics_avg[model_metrics_avg["fft"] == 512]
    plt.plot(metrics_128["rank"], metrics_128["evm"], color=color, linestyle="-", alpha=0.75)
    plt.plot(metrics_256["rank"], metrics_256["evm"], color=color, linestyle="-", alpha=0.75)
    plt.plot(metrics_512["rank"], metrics_512["evm"], color=color, linestyle="-", alpha=0.75, label=model)
plt.xticks(range(1, 10), [r"128-Small", "128-Medium", "128-Large", "256-Small", "256-Medium", "256-Large",
                      "512-Small", "512-Medium", "512-Large"], rotation=45)
plt.ylabel(r"EVM (dB)")
plt.grid(True)
ax.set_xticklabels([])


ax = plt.subplot(gs[2, 0])
for model, color in zip(models, colors):
    model_metrics_df = metrics_df[metrics_df["model"] == model]
    num_samples = len(model_metrics_df["rank"])
    jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
              -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
              -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1]
    plt.plot(model_metrics_df["rank"]+jitter, 100 * model_metrics_df["cyclic_prefix_ratio"], color=color, linestyle="",
             alpha=0.5, marker="o")
    model_metrics_avg = model_metrics_df.groupby(["fft", "rank"]).mean().reset_index()
    metrics_128 = model_metrics_avg[model_metrics_avg["fft"] == 128]
    metrics_256 = model_metrics_avg[model_metrics_avg["fft"] == 256]
    metrics_512 = model_metrics_avg[model_metrics_avg["fft"] == 512]
    plt.plot(metrics_128["rank"], 100 * metrics_128["cyclic_prefix_ratio"], color=color, linestyle="-", alpha=0.75)
    plt.plot(metrics_256["rank"], 100 * metrics_256["cyclic_prefix_ratio"], color=color, linestyle="-", alpha=0.75)
    plt.plot(metrics_512["rank"], 100 * metrics_512["cyclic_prefix_ratio"], color=color, linestyle="-", alpha=0.75, label=model)
plt.xticks(range(1, 10), [r"128-Small", "128-Medium", "128-Large", "256-Small", "256-Medium", "256-Large",
                      "512-Small", "512-Medium", "512-Large"], rotation=45)
plt.ylabel(r"$\rm RelErr_{CP}$ (%)") # cyclic_prefix_ratio
plt.grid(True)
plt.legend()
# plt.tight_layout()
plt.savefig("./experiment_results/main_experiment_plot_geodesic.png", bbox_inches='tight')
plt.show()

#%%  Combined main experiment Constellation plots

metrics_df = pd.read_csv("/experiment_results/data_throughput_experiment/full_results.csv")
models = ["PSK-GAN", 'WaveGAN', 'STFT-GAN']
for model in models:
    model_metrics_df = metrics_df[metrics_df["model"] == model]
    model_metrics_df = model_metrics_df.groupby(by=['dataset']).first()
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], wspace=0.01, hspace=0.01)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(gs[i,j])
            rank = i * 3 + j + 1
            model_run = model_metrics_df[model_metrics_df["rank"] == rank]
            json_path = model_run["config"][0]
            data_path = "/".join(json_path.split("/")[:-1])
            h5f = h5py.File(rf"{data_path}/demodulated_gen_symbols.h5", 'r')
            symbolstreams = h5f['demod_symbols'][:]
            h5f.close()
            demod_symbols = np.array(symbolstreams).astype(float)
            demod_symbols = data_loading.pack_to_complex(demod_symbols)
            mod_order = 16
            symbol_locs = constellation_symbols(mod_order=mod_order)
            I, Q = np.real(demod_symbols[:, ]).flatten(), np.imag(demod_symbols[:, ]).flatten()
            Z, xedges, yedges = np.histogram2d(I, Q, bins=150, range=[[-1.5, 1.5], [-1.5, 1.5]], density=True)
            plt.pcolormesh(xedges, yedges, Z, cmap=plt.cm.binary)  # , norm=mcolors.PowerNorm(0.75))  # , vmax=density_max))
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal', 'box')
            if rank not in [1, 4, 7]:
                ax.set_yticklabels([])
            else:
                ax.set_yticks([-1, 0, 1])
            if rank not in [7, 8, 9]:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([-1, 0, 1])
            if rank == 7:
                ax.set_xlabel("Small", fontsize=14)
            if rank == 8:
                ax.set_xlabel("Medium", fontsize=14)
            if rank == 9:
                ax.set_xlabel("Large", fontsize=14)
            if rank == 1:
                ax.set_ylabel("128", fontsize=14)
            if rank == 4:
                ax.set_ylabel("256", fontsize=14)
            if rank == 7:
                ax.set_ylabel("512", fontsize=14)
    fig.text(0.5, 0.01, "Allocation Size", ha='center', fontsize=14)
    fig.text(0.01, 0.5, "OFDM Symbol Length", va='center', fontsize=14, rotation='vertical')
    plt.savefig(f"./experiment_results/main_experiment_constellation_plot_{model}.png", dpi=500)
    plt.show()

#%% Modulation Experiment

df_temp = []
dir_path = "./experiment_results/modulation_experiment/*/*/distance_metrics.json"
dataset_fft_dict = {'256fft_subs36': 256,
                    '128fft_subs56': 128,
                    '256fft_subs112': 256,
                    '512fft_subs224': 512,
                    '128fft_subs36': 128,
                    '128fft_subs18': 128,
                    '256fft_subs74': 256,
                    '512fft_subs150': 512,
                    '512fft_subs74': 512}
dataset_rank_dict = {'256fft_subs36': 4,
                    '128fft_subs56': 3,
                    '256fft_subs112': 6,
                    '512fft_subs224': 9,
                    '128fft_subs36': 2,
                    '128fft_subs18': 1,
                    '256fft_subs74': 5,
                    '512fft_subs150': 8,
                    '512fft_subs74': 7}
dist_paths = glob.glob(dir_path)
for path in dist_paths:
    with open(path) as f:
        print(path)
        data = json.load(f)
        data["config"] = path
        data["dataset"] = data["config"].split("/")[3]
        data["timestamp"] = data["config"].split("/")[4]
        df_temp.append(data)
df = pd.DataFrame(df_temp)
df["fft"] = df["dataset"].map(dataset_fft_dict)
df["rank"] = df["dataset"].map(dataset_rank_dict)
df.to_csv("/home/jgs3/PycharmProjects/iqgan/experiment_results/modulation_experiment/full_results.csv", index=False)


#%% Modulation plots

metrics_df = pd.read_csv("/experiment_results/modulation_experiment/full_results.csv")
metrics_df = metrics_df.groupby(by=['dataset']).first().reset_index()

fig = plt.figure(figsize=(5,5))
gs = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.0, hspace=0.0)
count = 0
mods = ['QPSK_modulation', '16QAM_modulation', '32QAM_modulation', '64QAM_modulation']
for i in range(2):
    for j in range(2):
        ax = plt.subplot(gs[i, j])
        dataset = mods[count]
        model_run = metrics_df[metrics_df["dataset"] == dataset]
        json_path = model_run["config"].values[0]
        data_path = "/".join(json_path.split("/")[:-1])
        h5f = h5py.File(rf"{data_path}/demodulated_gen_symbols.h5", 'r')
        symbolstreams = h5f['demod_symbols'][:]
        h5f.close()
        demod_symbols = np.array(symbolstreams).astype(float)
        demod_symbols = data_loading.pack_to_complex(demod_symbols)
        mod_order = {'QPSK_modulation': 4, '16QAM_modulation': 16,
                     '32QAM_modulation': 32, '64QAM_modulation': 64}[dataset]
        symbol_locs = constellation_symbols(mod_order=mod_order)
        I, Q = np.real(demod_symbols[:, ]).flatten(), np.imag(demod_symbols[:, ]).flatten()
        Z, xedges, yedges = np.histogram2d(I, Q, bins=150, range=[[-1.5, 1.5], [-1.5, 1.5]], density=True)
        ax.pcolormesh(xedges, yedges, Z, cmap=plt.cm.binary)
        # ax.scatter(np.real(symbol_locs), np.imag(symbol_locs), color="red", s=3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal', 'box')
        if count in [1, 3]:
            ax.set_yticklabels([])
        else:
            ax.set_yticks([-1, 0, 1])
        if count in [0, 1]:
            ax.set_xticklabels([])
        else:
            ax.set_xticks([-1, 0, 1])
        count += 1
fig.text(0.5, 0.01, 'I', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'Q', va='center', rotation='vertical', fontsize=12)
plt.savefig(f"./experiment_results/modulation_constellation.png", dpi=500)
plt.show()


targ_paths = [r"/home/jgs3/PycharmProjects/iqgan/Datasets/mod_order_sets/mod4_iq_random_cp_long_128fft_symbol6_subs36_evm-25/",
              r"/home/jgs3/PycharmProjects/iqgan/Datasets/allocation_sets/mod16_iq_random_cp_long_128fft_rblocks1_subs36_evm-25/",
              r"/home/jgs3/PycharmProjects/iqgan/Datasets/mod_order_sets/mod32_iq_random_cp_long_128fft_symbol6_subs36_evm-25/",
              r"/home/jgs3/PycharmProjects/iqgan/Datasets/mod_order_sets/mod64_iq_random_cp_long_128fft_symbol6_subs36_evm-25/"]


fig = plt.figure(figsize=(5, 5))

gs = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.0, hspace=0.0)
count = 0
mods = ['QPSK_modulation', '16QAM_modulation', '32QAM_modulation', '64QAM_modulation']
for i in range(2):
    for j in range(2):
        ax = plt.subplot(gs[i, j])
        dataset = mods[count]
        data_path = targ_paths[count]
        h5f = h5py.File(rf"{data_path}demodulated_gen_symbols.h5", 'r')
        symbolstreams = h5f['demod_symbols'][:]
        h5f.close()
        demod_symbols = np.array(symbolstreams).astype(float)
        demod_symbols = data_loading.pack_to_complex(demod_symbols)
        mod_order = {'QPSK_modulation': 4, '16QAM_modulation': 16,
                     '32QAM_modulation': 32, '64QAM_modulation': 64}[dataset]
        symbol_locs = constellation_symbols(mod_order=mod_order)
        I, Q = np.real(demod_symbols[:, ]).flatten(), np.imag(demod_symbols[:, ]).flatten()
        Z, xedges, yedges = np.histogram2d(I, Q, bins=150, range=[[-1.5, 1.5], [-1.5, 1.5]], density=True)
        ax.pcolormesh(xedges, yedges, Z, cmap=plt.cm.binary)
        # ax.scatter(np.real(symbol_locs), np.imag(symbol_locs), color="red", s=3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal', 'box')
        if count in [1, 3]:
            ax.set_yticklabels([])
        else:
            ax.set_yticks([-1, 0, 1])
        if count in [0, 1]:
            ax.set_xticklabels([])
        else:
            ax.set_xticks([-1, 0, 1])
        count += 1
fig.text(0.5, 0.01, 'I', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'Q', va='center', rotation='vertical', fontsize=12)
plt.savefig(f"./experiment_results/target_modulation_constellation.png", dpi=500)
plt.show()

#%%

import matplotlib
font = {'size': 12}
matplotlib.rc('font', **font)

metrics_df = pd.read_csv("./experiment_results/modulation_experiment/full_results.csv")
dataset_rank_dict = {'QPSK_modulation': 1, '16QAM_modulation': 2,
             '32QAM_modulation': 3, '64QAM_modulation': 4}
datasets = ['QPSK_modulation', '16QAM_modulation', '32QAM_modulation', '64QAM_modulation']
metrics_df = metrics_df[metrics_df["dataset"].isin(datasets)]
metrics_df["rank"] = metrics_df["dataset"].map(dataset_rank_dict)
metrics_df = metrics_df.sort_values(by=["rank"])

fig = plt.figure(figsize=(5, 10))
gs = gridspec.GridSpec(ncols=1, nrows=3, wspace=0.0, hspace=0.02)
ax = plt.subplot(gs[0, 0])
jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
          -0.1, 0, 0.1]
plt.plot(metrics_df["rank"] + jitter, metrics_df["geodesic_psd_distance"], color="black", linestyle="", marker="o")
# plt.xticks(range(1, 5), ['QPSK', '16-QAM', '32-QAM', '64-QAM'])
plt.ylabel("Geodesic PSD Distance ($d_G$)")
plt.xlabel("Modulation Order")
plt.grid(True)

ax = plt.subplot(gs[1, 0])
jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
          -0.1, 0, 0.1]
plt.plot(metrics_df["rank"] + jitter, metrics_df["evm"], color="black", linestyle="", marker="o")
#plt.xticks(range(1,5), ['QPSK', '16-QAM', '32-QAM', '64-QAM'])
ax.set_xticklabels([])
plt.ylabel("EVM (dB)")
plt.xlabel("Modulation Order")
plt.grid(True)

ax = plt.subplot(gs[2, 0])
jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1,
          -0.1, 0, 0.1]
plt.plot(metrics_df["rank"] + jitter, metrics_df["cyclic_prefix_ratio"], color="black", linestyle="", marker="o")
plt.xticks(range(1, 5), ['QPSK', '16-QAM', '32-QAM', '64-QAM'])
plt.ylabel("Geodesic PSD Distance ($d_G$)")
plt.ylabel(r"$\rm RelErr_{CP}$ (%)") # cyclic_prefix_ratio
plt.xlabel("Modulation Order")
plt.grid(True)
plt.savefig("./experiment_results/modulation_metrics.png", bbox_inches="tight")
plt.show()

#%% Channel model experiment

df_temp = []
dir_path = "/experiment_results/channel_model_experiment/*/*/distance_metrics.json"

dist_paths = glob.glob(dir_path)
for path in dist_paths:
    with open(path) as f:
        data = json.load(f)
        data["config"] = path
        data["dataset"] = data["config"].split("/")[7]
        data["channel_model"] = data["dataset"].split("_")[1]
        data["timestamp"] = data["config"].split("/")[8]
        df_temp.append(data)
df = pd.DataFrame(df_temp)
df.to_csv("/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/full_results.csv", index=False)


metrics_df = pd.read_csv("/experiment_results/channel_model_experiment/full_results.csv")
datasets = ['EPA5Hz', 'EVA5Hz', 'EVA70Hz', "ETU70Hz", 'ETU300Hz']
dataset_rank_dict = {'EPA5Hz': 1, 'EVA5Hz': 2, 'EVA70Hz': 3, 'ETU70Hz': 4, 'ETU300Hz': 5}
target_BERs = [2.290950744558992e-06, 0.00022394043528064147, 0.01490893470790378, 0.013868843069873998, 0.08295074455899198]
target_evm = [-21.85929496995903, -21.457001554728894, -10.444520179396108, -8.36730205420144, -0.8361522550332884]
metrics_df = metrics_df[metrics_df["channel_model"].isin(datasets)]
metrics_df["rank"] = metrics_df["channel_model"].map(dataset_rank_dict)
metrics_df = metrics_df.sort_values(by=["rank"])

jitter = [-0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1]
model_metrics_avg = metrics_df.groupby(["channel_model"]).mean().reset_index()
model_metrics_avg = model_metrics_avg.sort_values(by=["rank"])

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(ncols=1, nrows=2, wspace=0.0, hspace=0.02)
ax = plt.subplot(gs[0, 0])
plt.plot(metrics_df["rank"] + jitter, metrics_df["evm"], color="blue", linestyle="", alpha=0.75, marker="o", label="Generated")
plt.plot(model_metrics_avg["rank"], target_evm, color="red", marker="o", linestyle="", alpha=0.75, label="Target")
ax.set_xticklabels([])
plt.legend()
plt.ylabel("EVM (dB)")  # , fontsize=12)
plt.grid(True)

ax = plt.subplot(gs[1, 0])
gen_BERs = model_metrics_avg["BER"]
plt.plot(metrics_df["rank"] + jitter, metrics_df["BER"], color="blue", linestyle="", alpha=0.75, marker="o", label="Generated")
plt.plot(model_metrics_avg["rank"], target_BERs, color="red", marker="o", linestyle="", alpha=0.75, label="Target")

plt.xticks(range(1, 6), ['EPA-5Hz', 'EVA-5Hz', 'EVA-70Hz', 'ETU-70Hz', 'ETU-300Hz'])
plt.ylabel("BER") # , fontsize=12)
plt.xlabel("Fading Channel Model", fontsize=12)
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.savefig("./experiment_results/channel_ber_evm_plot.png")
plt.show()


#%%
target_paths = [#r"/home/jgs3/PycharmProjects/iqgan/Datasets/OFDM/channel_sets/512fft_AWGN/",
                r"/home/jgs3/PycharmProjects/iqgan/Datasets/OFDM/channel_sets/512fft_EPA5Hz/",
                r"/home/jgs3/PycharmProjects/iqgan/Datasets/OFDM/channel_sets/512fft_EVA5Hz/",
                r"/home/jgs3/PycharmProjects/iqgan/Datasets/OFDM/channel_sets/512fft_EVA70Hz/",
                r"/home/jgs3/PycharmProjects/iqgan/Datasets/OFDM/channel_sets/512fft_ETU70Hz/",
                r"/home/jgs3/PycharmProjects/iqgan/Datasets/OFDM/channel_sets/512fft_ETU300Hz/"]
gen_paths = [#r"/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/512fft_AWGN/2021-05-14_15-45-22/",
             r"/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/512fft_EPA5Hz/2021-05-14_22-53-00/",
             r"/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/512fft_EVA5Hz/2021-06-09_18-38-10/",
             r"/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/512fft_EVA70Hz/2021-05-15_13-33-01/",
             r"/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/512fft_ETU70Hz/2021-06-09_11-58-29/",
             r"/home/jgs3/PycharmProjects/iqgan/experiment_results/channel_model_experiment/512fft_ETU300Hz/2021-05-15_06-10-43/"]

min_ranges = []
max_ranges = []
titles = [r"EPA-5Hz", r"EVA-5Hz", r"EVA-70Hz", r"ETU-70Hz", r"ETU-300Hz"]
for gen_path, targ_path in zip(gen_paths, target_paths):
    targ_dist = np.genfromtxt(rf'{targ_path}coherence_bandwidths.csv', delimiter=",")
    gen_dist = np.genfromtxt(rf'{gen_path}coherence_bandwidths.csv', delimiter=",")
    min_targ, max_targ = min(targ_dist), max(targ_dist)
    min_gen, max_gen = min(gen_dist), max(gen_dist)
    lower_range, upper_range = min(min_targ, min_gen), max(max_targ, max_gen)
    min_ranges.append(lower_range)
    max_ranges.append(upper_range)
lower_bound = np.min(min_ranges)
upper_bound = np.max(max_ranges)

fig = plt.figure(figsize=(5, 10))
gs = gridspec.GridSpec(ncols=1, nrows=5, wspace=0.0, hspace=0.05)
count = 0
for j in range(5):
    ax = plt.subplot(gs[j, 0])
    targ_path = target_paths[count]
    gen_path = gen_paths[count]
    targ_dist = np.genfromtxt(rf'{targ_path}coherence_bandwidths.csv', delimiter=",")
    gen_dist = np.genfromtxt(rf'{gen_path}coherence_bandwidths.csv', delimiter=",")
    bins = np.linspace(lower_bound, upper_bound, 100)
    plt.hist(targ_dist, bins, alpha=0.5, color="red", label="Target")
    plt.hist(gen_dist, bins, alpha=0.5, color="blue", label="Generated")
    plt.xlim(0, 0.15)
    ax.text(.5, .85, titles[j], fontsize=12, horizontalalignment='center', transform=ax.transAxes)
    plt.ylabel("Count")
    if count == 4:
        plt.xlabel("Coherence Bandwidth (cycles/sample)")
    if count in [0, 1, 2, 3]:
        ax.set_xticklabels([])
    plt.grid(True)
    count += 1
plt.legend()
plt.tight_layout()
plt.savefig("./experiment_results/channel_coherence.png", dpi=500)
plt.close('all')


#%%
