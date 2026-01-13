########################################################################################################################
#This script defines the Projection model used in the employee performance experiments
########################################################################################################################


from configs import emp_xp_params
import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
from configs import emp_nn_architecture
import os


class BaselineRegressionModel(nn.Module):
    """
    Baseline model that trains in a typical fashion (ignoring constraint set). Predictions then
    predicted onto constraint set afterward
    """

    def __init__(self, type="baseline"):
        super(BaselineRegressionModel, self).__init__()

        #define architecture from config, and initialize
        if type == "penalty":
            self.ffnn = emp_nn_architecture.get_ffnn_structure_penalty()
        else:
            self.ffnn = emp_nn_architecture.get_ffnn_structure()
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming initialization, with final layer bias set to log-odds of positive class
        """
        #compute prevalence
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

    def forward(self, x):
        """
        Define forward pass for baseline model
        """
        return self.ffnn(x)

    def fair_projection_marginal(self, predictions, targets, x, protected_cols, lb, ub):
        """
        Post-hoc projection with marginal fairness constraints

        @param predictions: raw model predictions (numpy array)
        @param targets: ground truth labels (numpy array)
        @param x: input features (numpy array)
        @param protected_cols: list of protected column indices
        @param lb: lower bound for predictions
        @param ub: upper bound for predictions

        @return: fair predictions, problem status
        """
        n = len(predictions)

        #Binarize protected columns
        x_binary = x.copy()
        for col in protected_cols:
            x_binary[:, col] = (x_binary[:, col] > 0.5).astype(float)

        #Decision variable: fair predictions for all samples
        yhat = cp.Variable(n)

        #Constraints
        constr = [yhat >= lb, yhat <= ub]

        slack = emp_xp_params.get_slack()

        #For each protected column, add marginal fairness constraints
        for col in protected_cols:
            #Masks for the two groups
            mask_0 = x_binary[:, col] == 0
            mask_1 = x_binary[:, col] == 1

            n_0 = mask_0.sum()
            n_1 = mask_1.sum()

            if n_0 > 0 and n_1 > 0:
                #Create selection vectors (binary indicators)
                selector_0 = mask_0.astype(float)
                selector_1 = mask_1.astype(float)

                #Compute means for each group
                mean_pred_0 = cp.sum(cp.multiply(yhat, selector_0)) / n_0
                mean_y_0 = np.dot(targets, selector_0) / n_0

                mean_pred_1 = cp.sum(cp.multiply(yhat, selector_1)) / n_1
                mean_y_1 = np.dot(targets, selector_1) / n_1

                #Marginal fairness constraints
                constr += [mean_pred_0 - mean_y_0 <= slack]
                constr += [mean_pred_0 - mean_y_0 >= -slack]
                constr += [mean_pred_1 - mean_y_1 <= slack]
                constr += [mean_pred_1 - mean_y_1 >= -slack]

        #Objective: minimize distance from raw predictions
        objective = cp.Minimize(cp.sum_squares(yhat - predictions))

        problem = cp.Problem(objective, constr)
        problem.solve(solver=cp.ECOS, verbose=False)

        #Fallback to other solvers if needed
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            problem.solve(solver=cp.SCS, eps=1e-6, verbose=False)

        print(f"Marginal projection status: {problem.status}")

        return yhat.value, problem.status


def get_test_loss_residuals(
    base_model,
    loader,
    lb,
    ub,
    protected_cols=[18, 19, 21, 27, 29],
    save_dir="predictions",
    save_predictions=True,
    pred_filename="predictions_posthoc.pkl",
):
    """
    Test evaluation with marginal fairness post-hoc projection.

    @param base_model: base model instance
    @param loader: data loader
    @param lb: lower bound for predictions
    @param ub: upper bound for predictions
    @param protected_cols: list of protected attribute column indices
    @param save_dir: directory to save figures and predictions
    @param save_predictions: if True, save predictions to .pkl file
    @param pred_filename: filename for saving predictions (default: "predictions_posthoc.pkl")
    """
    os.makedirs(save_dir, exist_ok=True)
    base_model.eval()

    all_raw_preds = []
    all_targets = []
    all_x = []

    with torch.no_grad():
        for xb, yb in loader:
            #Get raw predictions
            pred_logits = base_model(xb).squeeze(1)

            all_raw_preds.append(pred_logits)
            all_targets.append(yb.float().view(-1))
            all_x.append(xb)

    #Concatenate all data
    all_raw_preds = torch.cat(all_raw_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_x = torch.cat(all_x).numpy()

    print(f"\nApplying marginal fairness projection...")

    #Apply marginal fairness projection
    fair_preds, status = base_model.fair_projection_marginal(
        all_raw_preds, all_targets, all_x, protected_cols, lb, ub
    )

    #Save predictions if requested
    if save_predictions:
        import pickle

        pred_path = os.path.join(save_dir, pred_filename)

        predictions_dict = {
            "raw_predictions": all_raw_preds,
            "fair_predictions": fair_preds,
            "targets": all_targets,
            "features": all_x,
            "protected_cols": protected_cols,
            "projection_status": status,
            "type": "posthoc_marginal",
        }

        with open(pred_path, "wb") as f:
            pickle.dump(predictions_dict, f)

        print(f"Predictions saved to: {pred_path}")

    #Convert back to tensors for evaluation
    all_raw_preds_torch = torch.from_numpy(all_raw_preds)
    fair_preds_torch = torch.from_numpy(fair_preds)
    all_targets_torch = torch.from_numpy(all_targets)
    all_x_torch = torch.from_numpy(all_x)

    audit_dict = emp_xp_params.evaluate_fairness_audit(
        all_raw_preds_torch,
        fair_preds_torch,
        all_targets_torch,
        all_x_torch,
        protected_cols,
        emp_xp_params.get_slack(),
    )

    return audit_dict
