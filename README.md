## Oropharyngeal primary tumor segmentation for radiotherapy planning on magnetic resonance imaging using deep learning

# Overview
The aim of the project is to segment the primary tumor with deep learning approaches.

We study the following:
* The segmentation performance with a conventional UNet.
* The effect of introducing multiple MRI sequences.
* The effect of reducing the context around the tumor.

Published article: https://www.sciencedirect.com/science/article/pii/S2405631621000348 

# Code walkthrough 

The configuration file can be found in config.py. It can be used to change the hyperparameters for training or inference.

The main file can be found in train.py. It is used to run the experiments as defined in the configuration file.

Inside the directory "tools" you can find the scripts needed during the training:
  * Model_factory: Script that loads the models and performs training, prediction and training
  * loaders: loader_full.py for loading the images of the fully automatic approach. loader_semi.py for loading the images of the semi-automatic approach.
