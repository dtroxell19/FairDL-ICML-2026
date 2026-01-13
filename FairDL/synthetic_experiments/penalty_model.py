################################################################################################################################################
#This script defines the lagrangian loss  (i.e. "Penalty" and "Strict Penalty") models used for Synthetic Experiments and Described in the Paper
################################################################################################################################################

import torch
import torch.nn as nn
from configs import nn_architecture, synthetic_xp_params
from baseline_model import BaselineRegressionModel


class PenaltyRegressionModel(BaselineRegressionModel):

    def __init__(self, lower, upper):
        super(BaselineRegressionModel, self).__init__()

        self.ffnn = nn_architecture.get_ffnn_structure()
        self._initialize_weights()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        """
        Define forward pass for baseline model
        """
        y_hat = super().forward(x)
        y_hat = self.lower + (self.upper - self.lower) * torch.sigmoid(
            y_hat
        )  #used to incorporate box constraints without adding yet another lagrangian term in loss

        return y_hat


def batch_mean_gap(pred, xb):
    """
    Gets gap in empirical mean predictions for 2 groups for a single batch

    @param pred: predictions from model
    @param xb: input X batch

    @returns difference in empirical means
    """
    s = xb[:, 0]
    m0 = s == 0
    m1 = s == 1
    if m0.sum() < 1 or m1.sum() < 1:
        return pred.new_tensor(0.0)
    return pred[m1].mean() - pred[m0].mean()


def get_test_loss(model, test_loader):
    """
    Helper function to get the test set loss for penalty

    @param base_model: instance of class PenaltyRegressionModel
    @param test_loader: Torch DataLoader for the test set

    @returns both terms in penalty loss function
    """

    all_xb = []
    all_pred = []
    criterion = nn.MSELoss()
    model.eval()
    penalty_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            if (xb[:, 0] == 0).sum() < 1 or (
                xb[:, 0] == 1
            ).sum() < 1:  #skip batch if insufficient data for a group
                continue
            pred = model(xb)
            penalty_loss += criterion(pred.squeeze(1), yb.float())
            all_pred.append(pred)
            all_xb.append(xb)

    penalty_loss = penalty_loss / synthetic_xp_params.get_num_batches_test()
    all_xb = torch.cat(all_xb, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    return batch_mean_gap(all_pred, all_xb), penalty_loss
