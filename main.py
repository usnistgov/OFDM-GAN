"""Main OFDM-generating GAN experiment module

This module starts the training of a single GAN run, or a batch of GAN runs defined by a default
training configuration dictionary as well as an optional csv file containing a list of updated
configuration parameters (for ease of experimentation)

This module requires that it is run on a machine with NVIDIA GPU machines.

This file can also be imported as a module and contains the following functions:
    * parse_configurations - parse command line arguments
    * update specs_params - update GAN configuration dictionary based on requirements
    * config_name - define unique GAN directory name
    * run_config - run single GAN configuration training
    * main - main method that runs GAN training or batch of GAN trainings
"""


import os
import copy
import torch
import logging
import argparse
import datetime
import pandas as pd
from gan_train import gan_train
from experiment_resources.trainspecs_dict import specs_dict
from utils.gan_configurator import Automated_config_setup


def parse_configurations():
    """
    Parse options from terminal call
    Options:
        --configs: Specify path to csv containing configuration settings used to overwrite default GAN config settings
        --repeats: Specify number of times to repeat model training
    :return: configurations DataFrame, Number of repeats, Number of GPU devices, Main GPU device ID
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, help='CSV of configuration settings')
    parser.add_argument('--repeats', type=int, help='number of configuration repeats')
    args = parser.parse_args()
    configs, repeat_number = args.configs, args.repeats
    if configs is not None:
        configurations_df = pd.read_csv(configs)
        return configurations_df, repeat_number
    else:
        return None, repeat_number


def update_specs_params(specs_dict, num_gpus):
    """
    Update Training specs dictionary to avoid confilcts
    :param specs_dict: GAN Configuration dictionary
    :param num_gpus: Number of GPU devices being used
    :return: Updated GAN Configuration dictionary
    """
    dataset = specs_dict['dataloader_specs']['dataset_specs']["data_set"]
    if specs_dict["model_type"] == "wavegan":
        specs_dict["stride"] = 4
        specs_dict["D_updates"] = 5
        specs_dict["kernel_size"] = 25
        specs_dict['optim_params']['D']['betas'] = (0.5, 0.9)
        specs_dict['optim_params']['G']['betas'] = (0.5, 0.9)
        specs_dict["dataloader_specs"]['dataset_specs']["pad_signal"] = True
    elif specs_dict["model_type"] == "pskgan":
        specs_dict["stride"] = 4
        specs_dict["model_levels"] = 5
        specs_dict["progressive_kernels"] = True
        specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] = "global_min_max"
    else:  # STFT-GAN
        specs_dict['dataloader_specs']['dataset_specs']["pad_signal"] = True
        specs_dict['dataloader_specs']['dataset_specs']['stft'] = True
        specs_dict['dataloader_specs']['dataset_specs']['fft_shift'] = True
    if specs_dict["num_gpus"] > torch.cuda.device_count():
        specs_dict["num_gpus"] = torch.cuda.device_count()
    return specs_dict


def config_name(config):
    """
    Create GAN configuration name used in output-directory based config
    :param config: GAN configuration dictionary
    :return: GAN name string
    """
    config_string = ""
    for factor, value in config.items():
        if factor in ["data_scaler", "data_set"]:
            config_string += value + "_"
    print(config_string[:-1])
    return config_string[:-1]


def run_config(specs_dict, world_size, rank=0, gan_name=None):
    """
    Update configuration dictionary and runs GAN model
    :param rank: Main GPU ID number
    :param specs_dict: GAM Configuration dictionary
    :param world_size: Number of available GPUs
    :param gan_name: Name of GAN configuration used for saving purposes
    :return: None
    """
    specs_dict_updated = update_specs_params(specs_dict, world_size)

    # Define the results directory and make one if it doesnt already exist
    output_dir = "experiment_results/"
    gan_name = specs_dict_updated["dataloader_specs"]["dataset_specs"]["data_set"] if gan_name is None else gan_name
    gan_name = gan_name.replace("/", "_")
    output_path = f"{output_dir}{gan_name}/{str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-')}/"
    if specs_dict_updated["checkpoint"] is not None:
        output_path = os.path.dirname(specs_dict_updated["checkpoint"]) + "/"
    if specs_dict_updated["save_model"] and rank == specs_dict_updated["start_gpu"]:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    # Define Logging File used for saving training history info
    level = logging.INFO
    format = '  %(message)s'
    handlers = [logging.StreamHandler()]
    if specs_dict_updated["save_model"]:
        handlers = [logging.FileHandler(f'{output_path}config_training.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers)

    logging.info(f'Beginning Training of config: {gan_name}')
    gan_train(rank, world_size, specs_dict_updated, gan_name, output_path)

    logging.info("Training finished!")


if __name__ == '__main__':
    print("Main GAN Framework Run: ")
    configs_df, num_repeats = parse_configurations()
    num_repeats = 1 if num_repeats is None else num_repeats
    rank = 0
    for repeat in range(num_repeats):
        if configs_df is not None:
            print(f"Automated GAN Factor Screening: Repeat #{num_repeats}:")
            init_configs = Automated_config_setup()
            for ind in configs_df.index:
                config = configs_df.iloc[ind, :]
                config_str = config_name(config)

                # copy must be passed so changes dont accumulate to config dictionary
                specs_dict_updated = init_configs.map_params(config, copy.deepcopy(specs_dict))
                world_size = int(specs_dict_updated["num_gpus"])
                rank = int(specs_dict_updated["start_gpu"])
                if specs_dict_updated is None:
                    continue
                name_suffix = "_".join([str(val) for val in config.values])
                print(f"GAN Configuration: {config_str}, Repeat # {repeat}")
                run_config(specs_dict_updated, world_size, rank, config_str)
        else:
            print(f"Single GAN Configuration: Repeat # {repeat}")
            world_size = int(specs_dict["num_gpus"])
            rank = int(specs_dict["start_gpu"])
            run_config(specs_dict, world_size, rank)
