# <u> **Software for Modeling OFDM Communication Signals with Generative Adversarial Networks** </u>

## Overview 
This repository contains Python code to generate results for experiments on generative modeling of radio frequency (RF) communication signals, specifically synthetic Orthogonal-Frequency Division Multiplexing (OFDM) signals. This code implements two novel Generative adversarial network (GAN) 
models, a 1D and a 2D convolutional model, named **PSK-GAN** and **STFT-GAN**, respectively, as well as the **WaveGAN** model architecture as 
a baseline for comparison.  For experiment details and results, see

J. Sklar, A. Wunderlich, "Feasibility of Modeling Orthogonal Frequency-Division Multiplexing Communication Signals with Unsupervised Generative Adversarial Networks", Journal of Research of the National Institute of Standards and Technology, Volume 126, Article No. 126046 (2021) https://doi.org/10.6028/jres.126.046.  

## Software Implementation
The software enables automated testing of many model configurations across different datasets. Model creation and training is implemented
using the Pytorch library. This repository contains files for initializing the experiment test runs (`main.py`), training of GAN models(`gan_train.py`), loading target distributions (`data_loading.py`), and evaluation(`gan_evaluation.py`) of generated distributions. The `/utils` directory contains 
supporting modules for target dataset creation, and model evaluation. The `models/` directory contains modules that create **PSK-GAN**, **STFT-GAN**,
and **WaveGAN** architectures.

Running `main.py` runs the default GAN configuration specified by the configuration dictionary `./experiment_resources/training_specs_dict.py`.
Descriptions for the fields specified in `./experiment_resources/training_specs_dict.py` are located in 
`./experiment_resources/configuration_dictionary_description.csv`. Additionally, a set of model configurations can be run in an automated fashion 
by passing a configuration table (csv file) as an argument to the main python module (ex. `main.py --configs path_to_config_table.csv`). Column labels
of a configuration table should correspond to desired keys in the GAN configuration dictionary that are to be changed across runs. 

The training and test target datasets used in this study were synthesized using the script `scripts/target_data_synth.py`.  To execute experiments, first run this script and place its contents in a subdirectory named `Data/`.  When running the models, experimental results are saved in `experiment_results/`.

## <u>Requirements</u>
We use a `conda` virtual environment to manage the project library dependencies.
Run the following commands to install requirements to a new conda environment:
```setup
conda create --name <env> --file .experiment_resources/requirements.txt
conda activate <env>
pip install -r .experiment_resources/pip_requirements.txt
```

## <u>Running Experiments</u>
This code executes three experiments: (1) a data complexity experiment, (2) a modulation order experiment, and (3) a fading channel experiment.  In order to reproduce results from each of the three experiments, run
```angular2html
main.py --configs ./experiment_resources/test_configs_complexity_PSKGAN.csv 
main.py --configs ./experiment_resources/test_configs_complexity_WaveGAN.csv
main.py --configs ./experiment_resources/test_configs_complexity_STFTGAN.csv
main.py --configs ./experiment_resources/test_configs_modulation_STFTGAN.csv
main.py --configs ./experiment_resources/test_configs_channel_STFTGAN.csv
```
Aggregated plots across model runs are created using the script `./scripts/plotting_script.py`.

## <u>Implementation Notes</u>
Single process multi-GPU training is done using Pytorch's DataParallel method, in order to increase training speed.  During code development, we found that multi-process multi-GPU training using the DistributedDataParallel method was not compatible with the gradient penalty operation (autograd.grad).  Therefore, DistributedDataParallel is not recommended when using Wasserstein-GP loss.

## <u>Authors</u>
Jack Sklar (jack.sklar@nist.gov) and Adam Wunderlich (adam.wunderlich@nist.gov) \
Communications Technology Laboratory \
National Institute of Standards and Technology \
Boulder, Colorado 

## <u>Acknowledgements</u>
The authors thank Ian Wilkins and Sumeet Batra for their contributions to an early version of this software.

## <u>Licensing Statement</u>
This software was developed by employees of the National Institute of Standards and Technology (NIST), an
agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United
States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.
This software may be subject to foreign copyright.  Permission in the United States and in foreign countries,
to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this
software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this
notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY,
INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY
THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN
NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT
BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR
OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE
OR SERVICES PROVIDED HEREUNDER.

