########################################################################################################################
# This script defines the neural network architectures for the models used in the loan performance experiments
########################################################################################################################

import torch.nn as nn


def get_ffnn_structure_large(num_cols):
    """
    Larger feed-forward NN structure used for all models
    """
    return nn.Sequential(
        nn.Linear(num_cols, 250),
        nn.ReLU(),
        nn.Linear(250, 50),
        nn.ReLU(),
        nn.Linear(50, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )


def get_ffnn_structure_small(num_cols):
    """
    Smaller feed-forward NN structure used
    """
    return nn.Sequential(nn.Linear(num_cols, 2), nn.ReLU(), nn.Linear(2, 1))
