#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from module.entrypoint import train, load_model, run_model
from module.config import TrainGlobalConfig
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # verbose default 1
    config = TrainGlobalConfig()
    if config.continue_gdrive is not None:
        import gdown
        print("Downloading weights into model folder for transfer learning")
        gdown.download_folder(config.continue_gdrive, quiet=True, output="./model")
        train(data_folder, model_folder, verbose, continue_training=True)
    else:
        train(data_folder, model_folder, verbose)

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    return load_model(model_folder, verbose)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes, labels, probabilities = run_model(model, data, recordings, verbose)
    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
