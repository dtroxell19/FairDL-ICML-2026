########################################################################################################################
#This script defines the Projection model used in the loan experiments
########################################################################################################################

from configs import loan_xp_params
import torch
import cvxpy as cp
import numpy as np
from configs import loan_nn_architecture
import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


class BaselineRegressionModel(nn.Module):
    """
    Baseline model that trains in a typical fashion (ignoring constraint set). Raw test-time predictions then
    predicted onto constraint set afterward.
    """

    def __init__(self, p_pos, num_cols, size):
        super(BaselineRegressionModel, self).__init__()

        #define architecture from config, and initialize
        if size == "small":
            self.ffnn = loan_nn_architecture.get_ffnn_structure_small(num_cols)
        else:
            self.ffnn = loan_nn_architecture.get_ffnn_structure_large(num_cols)
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
        Define forward pass for baseline model
        """
        return self.ffnn(x)

    def fair_projection(self, raw_preds, attr1_indicator, attr2_indicator):
        """
        Convex problem that projects baseline model's raw predictions onto constraint set.

        @param raw_preds: raw predictions from the model (all samples)
        @param attr1_indicator: binary indicator (0 or 1) for first protected attribute
        @param attr2_indicator: binary indicator (0 or 1) for second protected attribute
        @param lb: lower bound for any given prediction
        @param ub: upper bound for any given prediction

        @return yhat.value: decision variable (transformed predictions)
        @return problem.status: str describing state of solution (solved, failed, etc)
        """

        n = len(raw_preds)

        #decision variable
        yhat = cp.Variable(n)

        attr1_0_mask = attr1_indicator == 0
        attr1_1_mask = attr1_indicator == 1
        n_attr1_0 = attr1_0_mask.sum()
        n_attr1_1 = attr1_1_mask.sum()

        constr = []

        #constraints for attr1
        if n_attr1_0 > 0 and n_attr1_1 > 0:
            sum_attr1_0 = cp.sum(cp.multiply(yhat, attr1_0_mask))
            sum_attr1_1 = cp.sum(cp.multiply(yhat, attr1_1_mask))
            constr += [
                sum_attr1_0 / n_attr1_0 - sum_attr1_1 / n_attr1_1
                <= loan_xp_params.get_slack()
            ]
            constr += [
                -(sum_attr1_0 / n_attr1_0 - sum_attr1_1 / n_attr1_1)
                <= loan_xp_params.get_slack()
            ]

        attr2_0_mask = attr2_indicator == 0
        attr2_1_mask = attr2_indicator == 1
        n_attr2_0 = attr2_0_mask.sum()
        n_attr2_1 = attr2_1_mask.sum()

        #constraints for attr2 (if both groups exist)
        if n_attr2_0 > 0 and n_attr2_1 > 0:
            sum_attr2_0 = cp.sum(cp.multiply(yhat, attr2_0_mask))
            sum_attr2_1 = cp.sum(cp.multiply(yhat, attr2_1_mask))
            constr += [
                sum_attr2_0 / n_attr2_0 - sum_attr2_1 / n_attr2_1
                <= loan_xp_params.get_slack()
            ]
            constr += [
                -(sum_attr2_0 / n_attr2_0 - sum_attr2_1 / n_attr2_1)
                <= loan_xp_params.get_slack()
            ]

        objective = cp.Minimize(
            cp.sum_squares(yhat - raw_preds)
        )  #projection in logit space
        problem = cp.Problem(objective, constr)
        problem.solve(solver=cp.SCS, eps=1e-6, max_iters=80000, verbose=False)

        return yhat.value, problem.status


def get_test_loss(base_model, test_loader, save_dir="figures", pos_weight=None):
    """
    Test evaluation for baseline model with PER-BATCH projection.
    Assumes model outputs logits (not sigmoid probabilities).

    @param base_model: torch model to analyze
    @param test_loader (torch DataLoader): test set DataLoader object
    @param save_dir (str): directory to save results if desired
    @param pos_weight: weighting for BCEWithLogitsLoss

    @returns test set metrics
    """

    os.makedirs(save_dir, exist_ok=True)
    base_model.eval()

    all_raw_preds = []
    all_fair_preds = []  # Collect fair predictions per batch
    all_attr1_indicators = []
    all_attr2_indicators = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:

            pred_logits = base_model(xb).squeeze(1)
            
            # Project THIS BATCH
            fair_preds_np, status = base_model.fair_projection(
                pred_logits.cpu().numpy(),
                xb[:, -2].cpu().numpy(),  # attr1 for this batch
                xb[:, -1].cpu().numpy(),  # attr2 for this batch
            )
            
            if status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Projection failed with status {status}")

            # Store both raw and fair predictions for this batch
            all_raw_preds.append(pred_logits)
            all_fair_preds.append(torch.from_numpy(fair_preds_np).float())
            all_attr1_indicators.append(xb[:, -2])
            all_attr2_indicators.append(xb[:, -1])
            all_targets.append(yb.float().view(-1))

    # Concatenate all batches
    all_raw_preds = torch.cat(all_raw_preds)
    all_fair_preds = torch.cat(all_fair_preds)  # Now contains per-batch projections
    all_attr1_indicators = torch.cat(all_attr1_indicators)
    all_attr2_indicators = torch.cat(all_attr2_indicators)
    targets_all = torch.cat(all_targets)

    raw_preds = all_raw_preds.float()
    fair_preds = all_fair_preds.float()

    print(f"Fair projection status: per-batch projection completed")

    # Rest of the metrics computation stays the same
    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    raw_loss = loss_fn(raw_preds, targets_all).item()
    fair_loss = loss_fn(fair_preds, targets_all).item()

    print(f"Original prediction test loss: {raw_loss:.4f}")
    print(f"Fair prediction test loss: {fair_loss:.4f}")

    # Get predicted probabilities for metric computation
    y_true = targets_all.numpy()
    y_raw_prob = torch.sigmoid(raw_preds).numpy()
    y_fair_prob = torch.sigmoid(fair_preds).numpy()

    y_raw_logits = raw_preds.numpy()
    y_fair_logits = fair_preds.numpy()

    attr1 = all_attr1_indicators.numpy()
    attr2 = all_attr2_indicators.numpy()

    y_raw_pred_labels = (y_raw_prob >= 0.5).astype(float)
    overall_misclass_raw = (y_raw_pred_labels != y_true).mean()

    mask_y0_raw = y_true == 0
    mask_y1_raw = y_true == 1
    misclass_y0_raw = (
        (y_raw_pred_labels[mask_y0_raw] != y_true[mask_y0_raw]).mean()
        if mask_y0_raw.sum() > 0
        else 0.0
    )
    misclass_y1_raw = (
        (y_raw_pred_labels[mask_y1_raw] != y_true[mask_y1_raw]).mean()
        if mask_y1_raw.sum() > 0
        else 0.0
    )

    print(f"\nRAW predictions (threshold=0.5):")
    print(f"  Overall misclassification: {overall_misclass_raw:.4f}")
    print(f"  Misclassification when y=0: {misclass_y0_raw:.4f}")
    print(f"  Misclassification when y=1: {misclass_y1_raw:.4f}")

    #misclassification at 0.5 threshold for projected outputs
    y_fair_pred_labels = (y_fair_prob >= 0.5).astype(float)
    overall_misclass_fair = (y_fair_pred_labels != y_true).mean()

    mask_y0_fair = y_true == 0
    mask_y1_fair = y_true == 1
    misclass_y0_fair = (
        (y_fair_pred_labels[mask_y0_fair] != y_true[mask_y0_fair]).mean()
        if mask_y0_fair.sum() > 0
        else 0.0
    )
    misclass_y1_fair = (
        (y_fair_pred_labels[mask_y1_fair] != y_true[mask_y1_fair]).mean()
        if mask_y1_fair.sum() > 0
        else 0.0
    )

    print(f"\nFAIR predictions (threshold=0.5):")
    print(f"  Overall misclassification: {overall_misclass_fair:.4f}")
    print(f"  Misclassification when y=0: {misclass_y0_fair:.4f}")
    print(f"  Misclassification when y=1: {misclass_y1_fair:.4f}")

    #compute mean differences for each protected attribute

    #attribute 1
    attr1_g0_mask = attr1 == 0
    attr1_g1_mask = attr1 == 1
    mean_prob_diff_raw_attr1 = (
        y_raw_prob[attr1_g0_mask].mean() - y_raw_prob[attr1_g1_mask].mean()
    )
    mean_logit_diff_raw_attr1 = (
        y_raw_logits[attr1_g0_mask].mean() - y_raw_logits[attr1_g1_mask].mean()
    )

    mean_prob_diff_fair_attr1 = (
        y_fair_prob[attr1_g0_mask].mean() - y_fair_prob[attr1_g1_mask].mean()
    )
    mean_logit_diff_fair_attr1 = (
        y_fair_logits[attr1_g0_mask].mean() - y_fair_logits[attr1_g1_mask].mean()
    )

    #attribute 2
    attr2_g0_mask = attr2 == 0
    attr2_g1_mask = attr2 == 1
    mean_prob_diff_raw_attr2 = (
        y_raw_prob[attr2_g0_mask].mean() - y_raw_prob[attr2_g1_mask].mean()
    )
    mean_logit_diff_raw_attr2 = (
        y_raw_logits[attr2_g0_mask].mean() - y_raw_logits[attr2_g1_mask].mean()
    )

    mean_prob_diff_fair_attr2 = (
        y_fair_prob[attr2_g0_mask].mean() - y_fair_prob[attr2_g1_mask].mean()
    )
    mean_logit_diff_fair_attr2 = (
        y_fair_logits[attr2_g0_mask].mean() - y_fair_logits[attr2_g1_mask].mean()
    )

    print(f"\nProtected Attribute 1:")
    print(f"  Mean prob diff RAW (group 0 - group 1): {mean_prob_diff_raw_attr1:.4f}")
    print(f"  Mean logit diff RAW (group 0 - group 1): {mean_logit_diff_raw_attr1:.4f}")
    print(f"  Mean prob diff FAIR (group 0 - group 1): {mean_prob_diff_fair_attr1:.4f}")
    print(
        f"  Mean logit diff FAIR (group 0 - group 1): {mean_logit_diff_fair_attr1:.4f}"
    )

    print(f"\nProtected Attribute 2:")
    print(f"  Mean prob diff RAW (group 0 - group 1): {mean_prob_diff_raw_attr2:.4f}")
    print(f"  Mean logit diff RAW (group 0 - group 1): {mean_logit_diff_raw_attr2:.4f}")
    print(f"  Mean prob diff FAIR (group 0 - group 1): {mean_prob_diff_fair_attr2:.4f}")
    print(
        f"  Mean logit diff FAIR (group 0 - group 1): {mean_logit_diff_fair_attr2:.4f}"
    )

    #Get AUC and Precision
    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_raw_prob)
    auc_raw = roc_auc_score(y_true, y_raw_prob)
    prec_raw, rec_raw, thresholds_pr_raw = precision_recall_curve(y_true, y_raw_prob)
    ap_raw = average_precision_score(y_true, y_raw_prob)

    fpr_fair, tpr_fair, _ = roc_curve(y_true, y_fair_prob)
    auc_fair = roc_auc_score(y_true, y_fair_prob)
    prec_fair, rec_fair, thresholds_pr_fair = precision_recall_curve(
        y_true, y_fair_prob
    )
    ap_fair = average_precision_score(y_true, y_fair_prob)

    #get f1 scores
    f1_scores_raw = 2 * prec_raw * rec_raw / (prec_raw + rec_raw + 1e-12)
    best_idx_raw = f1_scores_raw.argmax()
    best_f1_score_raw = f1_scores_raw[best_idx_raw]
    best_thresh_raw = thresholds_pr_raw[best_idx_raw - 1] if best_idx_raw > 0 else 0.5
    pred_labels_best_raw = (y_raw_prob >= best_thresh_raw).astype(float)
    overall_misclass_best_raw = (pred_labels_best_raw != y_true).mean()

    #best F1 and misclassification at optimal threshold for fair outputs
    f1_scores_fair = 2 * prec_fair * rec_fair / (prec_fair + rec_fair + 1e-12)
    best_idx_fair = f1_scores_fair.argmax()
    best_f1_score_fair = f1_scores_fair[best_idx_fair]
    best_thresh_fair = (
        thresholds_pr_fair[best_idx_fair - 1] if best_idx_fair > 0 else 0.5
    )
    pred_labels_best_fair = (y_fair_prob >= best_thresh_fair).astype(float)
    overall_misclass_best_fair = (pred_labels_best_fair != y_true).mean()

    print(
        f"\nBest F1 RAW: {best_f1_score_raw:.4f}, threshold: {best_thresh_raw:.4f}, misclass: {overall_misclass_best_raw:.4f}"
    )
    print(
        f"Best F1 FAIR: {best_f1_score_fair:.4f}, threshold: {best_thresh_fair:.4f}, misclass: {overall_misclass_best_fair:.4f}"
    )

    print(f"\nAUC (raw): {auc_raw:.4f}, AP (raw): {ap_raw:.4f}")
    print(f"AUC (fair): {auc_fair:.4f}, AP (fair): {ap_fair:.4f}")

    return {
        "raw_loss": raw_loss,
        "fair_loss": fair_loss,
        "auc_raw": auc_raw,
        "auc_fair": auc_fair,
        "ap_raw": ap_raw,
        "ap_fair": ap_fair,
        "overall_misclass_fair": overall_misclass_fair,
        "misclass_y0_fair": misclass_y0_fair,
        "misclass_y1_fair": misclass_y1_fair,
        "best_thresh_raw": best_thresh_raw,
        "best_thresh_fair": best_thresh_fair,
        "overall_misclass_best_raw": overall_misclass_best_raw,
        "overall_misclass_best_fair": overall_misclass_best_fair,
        "mean_prob_diff_raw_attr1": mean_prob_diff_raw_attr1,
        "mean_prob_diff_fair_attr1": mean_prob_diff_fair_attr1,
        "mean_logit_diff_raw_attr1": mean_logit_diff_raw_attr1,
        "mean_logit_diff_fair_attr1": mean_logit_diff_fair_attr1,
        "mean_prob_diff_raw_attr2": mean_prob_diff_raw_attr2,
        "mean_prob_diff_fair_attr2": mean_prob_diff_fair_attr2,
        "mean_logit_diff_raw_attr2": mean_logit_diff_raw_attr2,
        "mean_logit_diff_fair_attr2": mean_logit_diff_fair_attr2,
        "best_f1_score_raw": best_f1_score_raw,
        "best_f1_score_fair": best_f1_score_fair,
    }
