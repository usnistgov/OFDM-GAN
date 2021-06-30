"""
Time-Frequency representation STFT-GAN model creation

This module automates the creation of STFT-based GAN gan_models with flexible model building to
allow for automated testing of variable model architecture choices.
"""

import torch.utils.data
import torch.nn as nn
import torch
import torch.nn.functional as F


weight_init_dispatcher = {'uniform': nn.init.uniform_, 'normal': nn.init.normal_,
                          'kaiming_uniform': nn.init.kaiming_uniform_, 'kaiming_normal': nn.init.kaiming_normal_}


class G(nn.Module):
    """
    STFT-GAN Generator Model
    """
    def __init__(self, channel_list, network_levels, z_dim, num_channels=2, lower_resolution=16,
                 weight_init="kaiming_normal", **kwargs):
        super(G, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.num_channels = num_channels
        self.lower_resolution = lower_resolution
        self.scale_factor = 2
        self.weight_init = weight_init
        self.channel_list = channel_list
        self.network_levels = network_levels

        input_size = self.lower_resolution[0] * self.lower_resolution[1] * self.channel_list[0]
        self.input_layer = nn.Linear(z_dim, input_size, bias=True)
        self.input_channel_dim = self.channel_list[0]

        self.conv_layers = nn.ModuleList()
        self.build_model()
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
            kernel_size, dilation = self.kernel_size, 1
            output_padding = 1 if dilation > 1 or self.scale_factor == 1 else 0
            padding = [((kernel_dim - 1) * dilation + output_padding - 1) // 2 - (self.scale_factor - 1) // 2 for kernel_dim in kernel_size]
            output_padding = (0, 1) if layer_num == self.network_levels - 1 else 0
            stride = self.scale_factor
            conv_block = [nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride,
                                             padding, output_padding, bias=True)]
            if layer_num != self.network_levels - 1:
                conv_block.append(nn.ReLU())
            else:
                conv_block.append(nn.Tanh())
            self.conv_layers.append(nn.Sequential(*conv_block))

    def forward(self, z):
        """
        Method applies forward pass of the Generator
        :param z: batch of latent variables
        :return: batch of generated samples
        """
        z = z.view(z.size(0), -1)
        x = F.relu(self.input_layer(z))
        x = x.view(x.size(0), -1, self.lower_resolution[0], self.lower_resolution[1])
        for layer_number in range(self.network_levels):
            x = self.conv_layers[layer_number](x)
        return x


class D(nn.Module):
    """
    STFT-GAN Discriminator model
    """
    def __init__(self, channel_list, network_levels, num_channels=2, lower_resolution=16, scale_factor=2,
                 weight_init="kaiming_normal", **kwargs):
        super(D, self).__init__()
        self.__dict__.update(kwargs)  # unpack the dictionary such that each key becomes self.key
        self.num_channels = num_channels
        self.lower_resolution = lower_resolution
        self.scale_factor = scale_factor
        self.channel_list = channel_list
        self.network_levels = network_levels
        self.weight_init = weight_init
        self.scale_factor = 2
        output_size = self.lower_resolution[0] * self.lower_resolution[1] * self.channel_list[-1]
        self.output_layer = nn.Linear(output_size, 1, bias=True)
        self.conv_layers = nn.ModuleList()
        self.build_model()
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                weight_init_dispatcher[self.weight_init](m.weight.data)

    def build_model(self):
        """
        Method builds discriminator model architecture based on GAN configuration parameters and dataset dimensions
        :return:
        """
        for layer_num in range(self.network_levels):
            input_channels = self.channel_list[layer_num]
            output_channels = self.channel_list[layer_num + 1]
            kernel_size, dilation = self.kernel_size, 1
            padding = [((kernel_dim - 1) * dilation) // 2 - (self.scale_factor - 1) // 2 for kernel_dim in kernel_size]
            stride = self.scale_factor
            conv_block = [nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride,
                                    padding=padding, bias=True), nn.LeakyReLU(negative_slope=0.2)]
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
        x = self.output_layer(x)
        return x
