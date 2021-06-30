specs_dict = {'save_model': True,
              'eval_model': True,
              "checkpoint": None,
              "num_gpus": 2,
              "start_gpu": 0,
              'epochs': 1,
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
                        'data_set': "/allocation_sets/mod16_iq_random_cp_long_128fft_rblocks1_subs36_evm-25",
                        'num_samples': 0,
                        "pad_signal": False,
                        'stft':  False,
                        'nperseg': 128,
                        'fft_shift': False},
                    'batch_size': 128},
              'optim_params': {
                    'D': {'lr': 0.0001, 'betas': (0.0, 0.9)},
                    'G': {'lr': 0.0001, 'betas': (0.0, 0.9)}},
              }
