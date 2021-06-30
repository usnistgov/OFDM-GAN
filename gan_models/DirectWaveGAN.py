"""
Direct Waveform GAN creation module

This module implements the PSK-GAN and WaveGAN in an flexible manner that allows for
testing of different model architecture choices in an automated fashion.
"""


import torch.utils.data
import torch.nn as nn
import torch
import torch.nn.functional as F


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor by a random integer
    in {-n, n} and performing reflection padding where necessary.

    Taken from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    """
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):

        if self.shift_factor == 0:
            return x
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)
        # Combine sample indices into lists so that less shuffle operations need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)
        x_shuffle = x.clone()    # Make a copy of x for our output
        for k, idxs in k_map.items():   # Apply shuffle to each sample
            x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect') if k > 0 else \
                F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')
        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


weight_init_dispatcher = {'uniform': nn.init.uniform_, 'normal': nn.init.normal_,
                          'kaiming_uniform': nn.init.kaiming_uniform_, 'kaiming_normal': nn.init.kaiming_normal_}


class G(nn.Module):
    """
    WaveGAN or PSK-GAN Generator model
    """
    def __init__(self, channel_list, network_levels, z_dim, num_channels=1, lower_resolution=16,
                 progressive_kernels=True, weight_init="kaiming_normal", activation_scale=None, **kwargs):
        super(G, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.num_channels = num_channels
        self.lower_resolution = lower_resolution
        self.min_kernel_size = 4
        self.progressive_kernels = progressive_kernels
        self.weight_init = weight_init
        self.channel_list = channel_list
        self.network_levels = network_levels
        self.z_dim = z_dim
        self.activation_scale = activation_scale
        self.input_channel_dim = self.channel_list[0]
        self.conv_layers = nn.ModuleList()
        self.build_model()
        input_dim = self.z_dim
        self.input_layer = nn.Linear(input_dim, self.lower_resolution * self.channel_list[0], bias=True)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        """
        Method builds model architecture based on GAN configuration parameters and dataset dimensions
        :return:
        """
        for layer_num in range(self.network_levels):
            input_channels = self.input_channel_dim if layer_num == 0 else self.channel_list[layer_num]
            output_channels = self.channel_list[layer_num + 1]
            if self.progressive_kernels:
                kernel_size = self.kernel_size // self.scale_factor ** (self.network_levels - layer_num - 1)
                kernel_size = self.min_kernel_size if kernel_size < self.min_kernel_size else kernel_size
                dilation = 1
            #elif self.progressive_kernels == "dilation":
            #     kernel_size = 8
            #    dilation = 2 ** layer_num
            else:
                kernel_size, dilation = self.kernel_size, 1
            output_padding = 1 if dilation > 1 or self.scale_factor == 1 else 0
            padding = ((kernel_size - 1) * dilation + output_padding - 1) // 2 - (self.scale_factor - 1) // 2
            if kernel_size % 2 != 0:
                padding += 1
                output_padding = 1
            conv_block = [nn.ConvTranspose1d(input_channels, output_channels, kernel_size, self.scale_factor,
                                             padding, output_padding, dilation=dilation, bias=True)]
            activation_function = nn.ReLU() if layer_num != self.network_levels - 1 else nn.Tanh()
            conv_block.append(activation_function)
            if self.gen_batch_norm and layer_num != self.network_levels - 1:
                bn = nn.BatchNorm1d(num_features=output_channels)
                conv_block.append(bn)
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, z):
        """
        Method applies forward pass of the Generator
        :param z: batch of latent variables
        :return: batch of generated samples
        """
        z = z.view(z.size(0), -1)
        x = F.relu(self.input_layer(z))
        x = x.view(-1, self.channel_list[0], self.lower_resolution)
        for layer_number in range(self.network_levels):
            x = self.conv_layers[layer_number](x)
        return x


class D(nn.Module):
    """
    WaveGAN or PSK-GAN Discriminator model
    """
    def __init__(self, channel_list, network_levels, num_channels=2, lower_resolution=16, scale_factor=2,
                 weight_init="kaiming_normal", progressive_kernels=True, phase_shuffle=False, **kwargs):
        super(D, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.num_channels = num_channels
        self.lower_resolution = lower_resolution
        self.scale_factor = scale_factor
        self.progressive_kernels = progressive_kernels
        self.channel_list = channel_list
        self.network_levels = network_levels
        self.weight_init = weight_init
        self.scale_factor = scale_factor
        self.phase_shuffle = phase_shuffle
        self.output_channel_dim = self.channel_list[-1]
        self.output_layer = nn.Linear(self.lower_resolution * self.output_channel_dim, 1, bias=True)
        self.min_kernel_size = 4
        self.conv_layers = nn.ModuleList()
        self.build_model()
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        """
        Method builds model architecture based on GAN configuration parameters and dataset dimensions
        :return:
        """
        for layer_num in range(self.network_levels):
            layer_out_dim = self.lower_resolution * self.scale_factor ** (self.network_levels - layer_num - 1)
            input_channels = self.channel_list[layer_num]
            output_channels = self.output_channel_dim if layer_num == self.network_levels - 1 \
                else self.channel_list[layer_num + 1]
            if self.progressive_kernels:
                kernel_size = self.kernel_size // self.scale_factor ** layer_num
                kernel_size = self.min_kernel_size if kernel_size < self.min_kernel_size else kernel_size
                dilation = 1
            # elif self.progressive_kernels == "dilation":
            #    kernel_size = 8
            #    dilation = 2 ** (self.network_levels - layer_num - 1)
            else:
                kernel_size, dilation = self.kernel_size, 1
            padding = ((kernel_size - 1) * dilation - 1) // 2 - (self.scale_factor - 1) // 2
            if kernel_size % 2 != 0:
                padding += 1
            conv_block = [nn.Conv1d(input_channels, output_channels, kernel_size, stride=self.scale_factor,
                                    padding=padding, dilation=dilation, bias=True), nn.LeakyReLU(negative_slope=0.2)]
            if self.phase_shuffle:
                if layer_out_dim > 4:
                    phase_shuffle_layer = PhaseShuffle(shift_factor=2)
                    conv_block.append(phase_shuffle_layer)
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, x):
        """
        Method applies foward pass of Discriminator model
        :param x: batch of target or generated samples
        :return: discriminator output probability
        """
        for layer in range(self.network_levels):
            x = self.conv_layers[layer](x)
        x = torch.flatten(x, start_dim=1)
        out = self.output_layer(x)
        return out
