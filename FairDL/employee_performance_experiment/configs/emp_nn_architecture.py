########################################################################################################################
# This script defines the neural network architecture for the models used in the employee performance experiments
########################################################################################################################

import torch.nn as nn


def get_ffnn_structure():
    """
    Feed-forward NN structure used for Projection and F-Layer models in employee performance experiment
    """
    return nn.Sequential(nn.Linear(63, 5), nn.LayerNorm(5), nn.ReLU(), nn.Linear(5, 1))


def get_ffnn_structure_penalty():
    """
    Feed-forward NN structure used for Penalty and Strict Penalty models in employee performance experiment
    """
    return nn.Sequential(
        nn.Linear(63, 5), nn.LayerNorm(5), nn.ReLU(), nn.Linear(5, 1), nn.Sigmoid()
    )
