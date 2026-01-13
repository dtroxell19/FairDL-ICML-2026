########################################################################################################################
#This script defines the F-Layer model used for the employee performance experiments
########################################################################################################################


import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs import emp_xp_params
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from configs import emp_nn_architecture


class FairModelcvxpy(nn.Module):
    """
    Model trained with a declarative convex layer for fairness
    """

    def __init__(self, lb, ub, protected_cols):
        super(FairModelcvxpy, self).__init__()

        #define architecture from config, and initialize
        self.ffnn = emp_nn_architecture.get_ffnn_structure()
        self._initialize_weights()

        #set high number of max epochs as a safety net
        self.max_epochs = 15000
        self.lb = lb
        self.ub = ub
        self.protected_cols = protected_cols

        self.slack = np.full(self.max_epochs, emp_xp_params.get_slack())
        self.slack_current = self.slack[
            0
        ]  #schedule is updated in training loop in synthetic_experiments.py
        self.slack_current = torch.tensor([self.slack_current], dtype=torch.float32)

    def _initialize_weights(self):
        """
        Kaiming initialization, with final layer bias set to 0.5
        """
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                #initialize biases
                if m.bias is not None:
                    #last layer
                    if i == len(list(self.modules())) - 1:
                        nn.init.constant_(m.bias, 0.5)
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y):
        """
        Forward pass with marginal fairness constraints
        """
        y_hat = self.ffnn(x)
        batch_size = x.shape[0]

        #Binarize protected columns
        x_binary = x.clone()
        for col in self.protected_cols:
            x_binary[:, col] = (x_binary[:, col] > 0.5).float()

        #Create selection vectors for each marginal group
        selection_vectors = []
        for col in self.protected_cols:
            #Group 0: where col=0
            selector_0 = (x_binary[:, col] == 0).float()
            selection_vectors.append(selector_0)

            #Group 1: where col=1
            selector_1 = (x_binary[:, col] == 1).float()
            selection_vectors.append(selector_1)

        #Create cvxpy layer
        self.cvxpylayer = self.create_fair_projection_marginal(
            batch_size, len(self.protected_cols), self.lb, self.ub, selection_vectors
        )

        #Apply fairness projection
        #Pass: raw predictions, targets, then all selection vectors
        ytilde = self.cvxpylayer(y_hat.squeeze(), y.squeeze())

        (ytilde,) = self.cvxpylayer(y_hat.squeeze(), y.squeeze())
        return ytilde.unsqueeze(1)

    def create_fair_projection_marginal(
        self, batch_size, num_protected_cols, lb, ub, selection_matrices
    ):
        """
        Creates fairness layer with MARGINAL fairness constraints.
        The selection_matrices argument should be a fixed list of numpy arrays
        of shape (batch_size,) with 0/1 entries indicating group membership.
        """
        #Decision variable
        yhat = cp.Variable(batch_size)

        #Parameters (vary between forward passes)
        raw = cp.Parameter(batch_size)  #raw predictions
        y = cp.Parameter(batch_size)  #ground truth targets

        slack = emp_xp_params.get_slack()

        #Constraints
        constr = [yhat >= lb, yhat <= ub]

        #Marginal fairness constraints
        for col_idx in range(num_protected_cols):
            selector_0 = selection_matrices[col_idx * 2]
            selector_1 = selection_matrices[col_idx * 2 + 1]

            n_0 = selector_0.sum()
            n_1 = selector_1.sum()

            #Means for each group (constant selectors)
            mean_pred_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
            mean_y_0 = cp.sum(cp.multiply(y, selector_0)) / n_0

            mean_pred_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
            mean_y_1 = cp.sum(cp.multiply(y, selector_1)) / n_1

            #Fairness constraints
            constr += [
                mean_pred_0 - mean_y_0 <= slack,
                mean_pred_0 - mean_y_0 >= -slack,
                mean_pred_1 - mean_y_1 <= slack,
                mean_pred_1 - mean_y_1 >= -slack,
            ]

        #Objective: minimize distance from raw predictions
        objective = cp.Minimize(cp.sum_squares(yhat - raw))

        problem = cp.Problem(objective, constr)

        #Only raw and y are parameters
        cvxpylayer = CvxpyLayer(problem, parameters=[raw, y], variables=[yhat])

        return cvxpylayer


def get_test_loss_residuals(
    model,
    loader,
    protected_cols,
    save_dir="predictions",
    type="fair",
    save_predictions=True,
    pred_filename="predictions_fair.pkl",
):
    """
    Extended test evaluation for the fairness model with multiple protected attributes.

    @param model: FairModelcvxpy instance
    @param loader: data loader
    @param protected_cols: list of protected attribute column indices
    @param save_dir: directory to save figures
    @param type: "fair" or "penalty" - type of model tested
    @param save_predictions: if True, save predictions to .pkl file
    @param pred_filename: optional custom filename (default: "predictions_{type}.pkl")
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_fair_preds = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        for xb, yb in loader:

            #Get fair predictions (after fairness layer)
            if type == "fair":
                fair_preds = model(xb, yb).squeeze(1)
            else:
                fair_preds = model(xb).squeeze(1)

            all_fair_preds.append(fair_preds)
            all_targets.append(yb.float().view(-1))
            all_features.append(xb)

    #Concatenate all
    all_fair_preds = torch.cat(all_fair_preds)
    all_targets = torch.cat(all_targets)
    all_features = torch.cat(all_features)

    #Save predictions if requested
    if save_predictions:
        import pickle

        if pred_filename is None:
            pred_filename = f"predictions_{type}.pkl"

        pred_path = os.path.join(save_dir, pred_filename)

        predictions_dict = {
            "predictions": all_fair_preds.cpu().numpy(),
            "targets": all_targets.cpu().numpy(),
            "features": all_features.cpu().numpy(),
            "type": type,
            "protected_cols": protected_cols,
        }

        with open(pred_path, "wb") as f:
            pickle.dump(predictions_dict, f)

        print(f"Predictions saved to: {pred_path}")

    #Use the evaluation function
    audit_dict = emp_xp_params.evaluate_fairness_audit(
        all_fair_preds,
        all_fair_preds,
        all_targets,
        all_features,
        protected_cols,
        emp_xp_params.get_slack(),
    )

    return audit_dict
