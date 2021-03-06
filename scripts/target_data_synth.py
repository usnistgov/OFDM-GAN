from utils.construct_ofdm import save_dataset

# save_dataset(traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, channel_3gpp="EPA5Hz")
# save_dataset(traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-40, num_frames=2, channel_3gpp="EVA70Hz")
save_dataset(traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-50, num_frames=2, channel_3gpp="ETU300Hz")


#%% Allocation Datasets
save_dataset(traffic_setting="low_traffic", mod_order=16, symbol_length=128, EVM=-25, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=16, symbol_length=128, EVM=-25, num_frames=1)
save_dataset(traffic_setting="high_traffic", mod_order=16, symbol_length=128, EVM=-25,  num_frames=1)
save_dataset(traffic_setting="low_traffic", mod_order=16, symbol_length=256, EVM=-25, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=16, symbol_length=256, EVM=-25, num_frames=1)
save_dataset(traffic_setting="high_traffic", mod_order=16, symbol_length=256, EVM=-25, num_frames=1)
save_dataset(traffic_setting="low_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=1)
save_dataset(traffic_setting="high_traffic", mod_order=16, symbol_length=512, EVM=-25,  num_frames=1)

# QAM Modulation order Datasets
save_dataset(traffic_setting="mid_traffic", mod_order=2, symbol_length=128, EVM=-25, scfdma=False, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=4, symbol_length=128, EVM=-25, scfdma=False, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=8, symbol_length=128, EVM=-25, scfdma=False, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=32, symbol_length=128, EVM=-25, scfdma=False, num_frames=1)
save_dataset(traffic_setting="mid_traffic", mod_order=64, symbol_length=128, EVM=-25, scfdma=False, num_frames=1)

# propogated channel model Datasets
save_dataset(num_samples=16384, traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, bitstream_type="random_fixed", channel_3gpp="AWGN")
save_dataset(num_samples=16384, traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, bitstream_type="random_fixed", channel_3gpp="EPA5Hz")
save_dataset(num_samples=16384, traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, bitstream_type="random_fixed", channel_3gpp="EVA5Hz")
save_dataset(num_samples=16384, traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, bitstream_type="random_fixed", channel_3gpp="EVA70Hz")
save_dataset(num_samples=16384, traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, bitstream_type="random_fixed", channel_3gpp="ETU70Hz")
save_dataset(num_samples=16384, traffic_setting="mid_traffic", mod_order=16, symbol_length=512, EVM=-25, num_frames=2, bitstream_type="random_fixed", channel_3gpp="ETU300Hz")

#%%
from utils.construct_ofdm import save_dataset
import numpy as np

save_dataset(traffic_setting="high_traffic", symbol_length=256, cfo=50)
save_dataset(traffic_setting="high_traffic", symbol_length=256, cfo=150)
save_dataset(traffic_setting="high_traffic", symbol_length=256, cfo=300)
save_dataset(traffic_setting="high_traffic", symbol_length=256, delta_G=0.1, delta_phi=np.pi * (5/180))
save_dataset(traffic_setting="high_traffic", symbol_length=256, delta_G=0.2, delta_phi=np.pi * (10/180))
save_dataset(traffic_setting="high_traffic", symbol_length=256, delta_G=0.3, delta_phi=np.pi * (15/180))

