########################################################################################################################
# This script defines the F-Layer model for the loan experiments
########################################################################################################################

import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs import loan_xp_params
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from configs import loan_nn_architecture
import os
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


class FairModelcvxpy(nn.Module):
    """
    Model trained with a declarative convex layer for fairness
    """

    def __init__(self, p_pos=None, num_cols=90, size="small"):
        super(FairModelcvxpy, self).__init__()

        #define architecture from config, and initialize
        if size == "small":
            self.ffnn = loan_nn_architecture.get_ffnn_structure_small(num_cols)
        else:
            self.ffnn = loan_nn_architecture.get_ffnn_structure_large(num_cols)
        #set high number of max epochs as a safety net
        self.max_epochs = 15000

        self.slack = np.full(self.max_epochs, loan_xp_params.get_slack())
        self.slack_current = self.slack[0]
        self.slack_current = torch.tensor([self.slack_current], dtype=torch.float32)
        self.final_layer = self._get_final_layer()

        self._initialize_weights(p_pos=p_pos)

    def _get_final_layer(self):
        """
        Helper func to get  last Linear layer in  network (used for init)
        """

        for module in reversed(list(self.ffnn.modules())):
            if isinstance(module, nn.Linear):
                return module
        return None

    def _initialize_weights(self, p_pos):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.08)
                if m.bias is not None:
                    #Initialize bias to predict mean target rate (logit of positive rate)
                    if m is self.final_layer:
                        nn.init.constant_(m.bias, np.log(p_pos / (1 - p_pos)))
                    else:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Define forward pass for fairness model
        """

        y_hat = self.ffnn(x)

        #get indicators for each protected attribute
        #x[:, -2] = first protected attribute (binary: 0 or 1)
        #x[:, -1] = second protected attribute / urban_rural (binary: 0 or 1)
        indicator_attr1 = x[:, -2]
        indicator_attr2 = x[:, -1]

        raw_preds = y_hat.squeeze(1)
        #re-declare the layer each forward pass if slack schedule is not constant
        if isinstance(self.slack_current, int) or isinstance(self.slack_current, float):
            self.slack_current = torch.tensor([self.slack_current], dtype=torch.float32)

        attr1_np = indicator_attr1.detach().cpu().numpy()
        attr2_np = indicator_attr2.detach().cpu().numpy()

        self.cvxpylayer = self.create_fair_layer(len(raw_preds), attr1_np, attr2_np)

        ytilde = self.cvxpylayer(raw_preds, self.slack_current)[0]

        return ytilde.unsqueeze(1)

    def create_fair_layer(self, n_samples, attr1_indicator, attr2_indicator):
        """
        Creates the fairness layer enforcing fairness independently for 2 protected attributes.


        @param n_samples: Number of samples in current batch
        @param attr1_indicator: numpy array of 0s and 1s for first protected attribute
        @param attr2_indicator: numpy array of 0s and 1s for second protected attribute

        @return cvxpylayer: a cvxpylayer's layer instance
        """

        #decision vars & parames
        yhat = cp.Variable(n_samples)
        raw = cp.Parameter(n_samples)
        slack = cp.Parameter(1)

        mask_attr1_0 = (attr1_indicator == 0).astype(float)
        mask_attr1_1 = (attr1_indicator == 1).astype(float)
        n_attr1_0 = mask_attr1_0.sum()
        n_attr1_1 = mask_attr1_1.sum()

        constr = []
        #constraints for attribute 1
        if n_attr1_0 > 0 and n_attr1_1 > 0:
            sum_attr1_0 = mask_attr1_0 @ yhat
            sum_attr1_1 = mask_attr1_1 @ yhat

            constr += [sum_attr1_0 / n_attr1_0 - sum_attr1_1 / n_attr1_1 <= slack]
            constr += [-(sum_attr1_0 / n_attr1_0 - sum_attr1_1 / n_attr1_1) <= slack]

        mask_attr2_0 = (attr2_indicator == 0).astype(float)
        mask_attr2_1 = (attr2_indicator == 1).astype(float)
        n_attr2_0 = mask_attr2_0.sum()
        n_attr2_1 = mask_attr2_1.sum()

        #constraints for attribute 2
        if n_attr2_0 > 0 and n_attr2_1 > 0:
            sum_attr2_0 = mask_attr2_0 @ yhat
            sum_attr2_1 = mask_attr2_1 @ yhat
            constr += [sum_attr2_0 / n_attr2_0 - sum_attr2_1 / n_attr2_1 <= slack]
            constr += [-(sum_attr2_0 / n_attr2_0 - sum_attr2_1 / n_attr2_1) <= slack]

        #objective: minimize squared distance from raw predictions
        objective = cp.Minimize(cp.sum_squares(yhat - raw))

        problem = cp.Problem(objective, constr)

        cvxpylayer = CvxpyLayer(problem, parameters=[raw, slack], variables=[yhat])

        return cvxpylayer


def get_test_loss(model, loader, pos_weight=None):
    """
    Gets inference BCE loss and prints misclassification rates / metrics

    @param model: model to test
    @param loader: torch DataLoader that gets inference data
    @param pos_weight: optional positive class weighting for BCEWithLogitsLoss

    @return: dict of metrics
    """

    device = next(model.parameters()).device

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_errors = 0

    class_errors = {0: 0, 1: 0}  #misclassification count per true class
    class_samples = {0: 0, 1: 0}  #number of samples per true class

    #stats per protected attribute
    attr1_groups = {0: {"preds": [], "logits": []}, 1: {"preds": [], "logits": []}}
    attr2_groups = {0: {"preds": [], "logits": []}, 1: {"preds": [], "logits": []}}

    all_targets = []
    all_preds = []
    all_logits = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(1)
            yb_float = yb.float().view(-1)

            #protected attribute indicators
            attr1_indicator = xb[:, -2].cpu().numpy()
            attr2_indicator = xb[:, -1].cpu().numpy()

            total_loss += criterion(pred, yb_float) * yb_float.size(0)
            pred_probs = torch.sigmoid(pred)
            pred_labels = (pred_probs >= 0.5).float()

            #overall misclassification
            total_errors += (pred_labels != yb_float).sum().item()
            total_samples += yb_float.size(0)

            #misclassification broken down by true class
            for cls in [0, 1]:
                mask = yb_float == cls
                class_errors[cls] += (pred_labels[mask] != yb_float[mask]).sum().item()
                class_samples[cls] += mask.sum().item()

            #sttore predictions by protected attribute groups
            pred_probs_cpu = pred_probs.cpu().numpy()
            pred_logits_cpu = pred.cpu().numpy()

            for i in range(len(xb)):
                attr1_val = int(attr1_indicator[i])
                attr2_val = int(attr2_indicator[i])

                attr1_groups[attr1_val]["preds"].append(pred_probs_cpu[i])
                attr1_groups[attr1_val]["logits"].append(pred_logits_cpu[i])

                attr2_groups[attr2_val]["preds"].append(pred_probs_cpu[i])
                attr2_groups[attr2_val]["logits"].append(pred_logits_cpu[i])

            all_targets.append(yb_float.cpu())
            all_preds.append(pred_probs.cpu())
            all_logits.append(pred.cpu())

    avg_loss = total_loss / total_samples
    overall_misclass = total_errors / total_samples if total_samples > 0 else 0.0
    misclass_by_class = {
        cls: class_errors[cls] / class_samples[cls] if class_samples[cls] > 0 else 0.0
        for cls in [0, 1]
    }

    print(f"Overall misclassification rate (0.5 threshold): {overall_misclass:.4f}")
    for cls in [0, 1]:
        print(
            f"Misclassification rate for y={cls} (0.5 threshold): {misclass_by_class[cls]:.4f}"
        )

    #gather all predictions and targets across test batches
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_logits = torch.cat(all_logits).numpy()

    #compute mean differences for each protected attribute independently
    mean_prob_attr1_0 = (
        np.mean(attr1_groups[0]["preds"]) if attr1_groups[0]["preds"] else np.nan
    )
    mean_prob_attr1_1 = (
        np.mean(attr1_groups[1]["preds"]) if attr1_groups[1]["preds"] else np.nan
    )
    mean_prob_diff_attr1 = (
        mean_prob_attr1_0 - mean_prob_attr1_1
        if not np.isnan(mean_prob_attr1_0) and not np.isnan(mean_prob_attr1_1)
        else np.nan
    )

    mean_logit_attr1_0 = (
        np.mean(attr1_groups[0]["logits"]) if attr1_groups[0]["logits"] else np.nan
    )
    mean_logit_attr1_1 = (
        np.mean(attr1_groups[1]["logits"]) if attr1_groups[1]["logits"] else np.nan
    )
    mean_logit_diff_attr1 = (
        mean_logit_attr1_0 - mean_logit_attr1_1
        if not np.isnan(mean_logit_attr1_0) and not np.isnan(mean_logit_attr1_1)
        else np.nan
    )

    mean_prob_attr2_0 = (
        np.mean(attr2_groups[0]["preds"]) if attr2_groups[0]["preds"] else np.nan
    )
    mean_prob_attr2_1 = (
        np.mean(attr2_groups[1]["preds"]) if attr2_groups[1]["preds"] else np.nan
    )
    mean_prob_diff_attr2 = (
        mean_prob_attr2_0 - mean_prob_attr2_1
        if not np.isnan(mean_prob_attr2_0) and not np.isnan(mean_prob_attr2_1)
        else np.nan
    )

    mean_logit_attr2_0 = (
        np.mean(attr2_groups[0]["logits"]) if attr2_groups[0]["logits"] else np.nan
    )
    mean_logit_attr2_1 = (
        np.mean(attr2_groups[1]["logits"]) if attr2_groups[1]["logits"] else np.nan
    )
    mean_logit_diff_attr2 = (
        mean_logit_attr2_0 - mean_logit_attr2_1
        if not np.isnan(mean_logit_attr2_0) and not np.isnan(mean_logit_attr2_1)
        else np.nan
    )

    print(f"\nProtected Attribute 1 (column -2):")
    print(
        f"  Mean prob (group 0): {mean_prob_attr1_0:.4f}, Mean prob (group 1): {mean_prob_attr1_1:.4f}"
    )
    print(f"  Mean prob diff (0 - 1): {mean_prob_diff_attr1:.4f}")
    print(
        f"  Mean logit (group 0): {mean_logit_attr1_0:.4f}, Mean logit (group 1): {mean_logit_attr1_1:.4f}"
    )
    print(f"  Mean logit diff (0 - 1): {mean_logit_diff_attr1:.4f}")

    print(f"\nProtected Attribute 2 / UrbanRural (column -1):")
    print(
        f"  Mean prob (group 0): {mean_prob_attr2_0:.4f}, Mean prob (group 1): {mean_prob_attr2_1:.4f}"
    )
    print(f"  Mean prob diff (0 - 1): {mean_prob_diff_attr2:.4f}")
    print(
        f"  Mean logit (group 0): {mean_logit_attr2_0:.4f}, Mean logit (group 1): {mean_logit_attr2_1:.4f}"
    )
    print(f"  Mean logit diff (0 - 1): {mean_logit_diff_attr2:.4f}")

    #ROC curve and AUC
    fpr, tpr, thresholds_roc = roc_curve(all_targets, all_preds)
    auc_score = roc_auc_score(all_targets, all_preds)

    #PR curve
    precision, recall, thresholds_pr = precision_recall_curve(all_targets, all_preds)
    ap_score = average_precision_score(all_targets, all_preds)

    print(f"\nAUC: {auc_score:.4f}, Average Precision: {ap_score:.4f}")

    #opt F1 threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = f1_scores.argmax()
    best_f1_score = f1_scores[best_idx]
    best_thresh = thresholds_pr[best_idx - 1] if best_idx > 0 else 0.5
    pred_labels_best = (all_preds >= best_thresh).astype(float)
    overall_misclass_best = (pred_labels_best != all_targets).mean()
    print(
        f"Overall misclassification at optimal F1 threshold ({best_thresh:.4f}): {overall_misclass_best:.4f}"
    )
    print(f"Best F1 score: {best_f1_score:.4f}")

    return (
        avg_loss,
        overall_misclass,
        misclass_by_class,
        auc_score,
        ap_score,
        best_thresh,
        overall_misclass_best,
        mean_prob_diff_attr1,
        mean_logit_diff_attr1,
        mean_prob_diff_attr2,
        mean_logit_diff_attr2,
        best_f1_score,
    )
