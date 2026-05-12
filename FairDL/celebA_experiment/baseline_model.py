########################################################################################################################
# Baseline (Projection) model for the CelebA experiments.
#
# Trains with standard BCE loss; at inference, projects logits onto the pairwise
# demographic parity constraint set for all C(4,2) = 6 intersectional group pairs.
########################################################################################################################

from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp

from configs import celeba_nn_architecture as nn_arch
from configs import celeba_xp_params as xp_params
from load_data import NUM_GROUPS


# ── Helpers ──────────────────────────────────────────────────────────────────────

def get_group_pairs(group_keys):
    """Return all ordered pairs (g_i, g_j) with g_i < g_j."""
    return list(combinations(sorted(group_keys), 2))


def compute_group_means(logits, groups):
    """Compute mean logit per group. Returns dict {group_id: mean_logit}."""
    means = {}
    for g in torch.unique(groups):
        g = g.item()
        mask = groups == g
        if mask.sum() > 0:
            means[g] = logits[mask].mean()
    return means


def compute_max_pairwise_gap(logits, groups):
    """Max |mean(logit | g_i) - mean(logit | g_j)| across all group pairs."""
    means = compute_group_means(logits, groups)
    if len(means) < 2:
        return 0.0
    max_gap = 0.0
    for g_i, g_j in get_group_pairs(means.keys()):
        gap = abs(means[g_i].item() - means[g_j].item()) if isinstance(means[g_i], torch.Tensor) \
            else abs(means[g_i] - means[g_j])
        max_gap = max(max_gap, gap)
    return max_gap


def compute_sum_squared_gaps(logits, groups):
    """Sum of squared pairwise gaps. Differentiable when inputs are torch tensors."""
    means = compute_group_means(logits, groups)
    if len(means) < 2:
        return logits.new_tensor(0.0)
    total = logits.new_tensor(0.0)
    for g_i, g_j in get_group_pairs(means.keys()):
        total = total + (means[g_i] - means[g_j]) ** 2
    return total


# ── Baseline (Projection) model ─────────────────────────────────────────────────

class BaselineModel(nn.Module):
    """
    Baseline model: trains with standard BCE loss, projects logits onto the
    pairwise demographic parity constraint set at inference.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, head_type="linear",
                 p_pos=0.5, freeze_backbone=False, lora=False, lora_rank=8, lora_alpha=16):
        super(BaselineModel, self).__init__()

        self.backbone, num_features = nn_arch.get_backbone(
            backbone_name, pretrained, freeze=freeze_backbone,
            lora=lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
        )
        self.classifier = nn_arch.get_classifier_head(num_features, head_type)
        self.pretrained = pretrained
        self._initialize_weights(p_pos)

    def _initialize_weights(self, p_pos):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        final_layer = self._get_final_linear(self.classifier)
        if final_layer is not None:
            nn.init.xavier_uniform_(final_layer.weight, gain=0.3)
            if final_layer.bias is not None:
                p_pos = np.clip(p_pos, 1e-4, 1 - 1e-4)
                nn.init.constant_(final_layer.bias, np.log(p_pos / (1 - p_pos)))

        if not self.pretrained:
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)

    @staticmethod
    def _get_final_linear(module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def forward(self, images):
        features = self.backbone(images)
        return self.classifier(features)

    def fair_projection(self, raw_logits_np, groups_np):
        """
        Project raw logits onto the pairwise demographic parity constraint set:

            min  ||yhat - raw||_2^2
            s.t. |mean(yhat | g_i) - mean(yhat | g_j)| <= slack  for all pairs (i, j)

        @param raw_logits_np (ndarray): (N,) raw logits
        @param groups_np (ndarray): (N,) integer group IDs (0–3)

        @returns (projected_logits, solver_status)
        """
        n = len(raw_logits_np)
        slack = xp_params.get_slack()

        yhat = cp.Variable(n)
        constr = []

        group_masks = {}
        for g in range(NUM_GROUPS):
            mask = (groups_np == g).astype(float)
            if mask.sum() > 0:
                group_masks[g] = (mask, mask.sum())

        for g_i, g_j in get_group_pairs(group_masks.keys()):
            mask_i, n_i = group_masks[g_i]
            mask_j, n_j = group_masks[g_j]
            mean_i = (mask_i @ yhat) / n_i
            mean_j = (mask_j @ yhat) / n_j
            constr += [mean_i - mean_j <= slack, -(mean_i - mean_j) <= slack]

        objective = cp.Minimize(cp.sum_squares(yhat - raw_logits_np))
        problem = cp.Problem(objective, constr)
        problem.solve(solver=cp.SCS, eps=1e-6, max_iters=80000, verbose=False)

        return yhat.value, problem.status
