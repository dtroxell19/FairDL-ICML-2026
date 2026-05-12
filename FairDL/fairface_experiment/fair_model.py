########################################################################################################################
# This script defines the F-Layer (fairness layer) model for the FairFace experiments.
#
# The fairness layer projects logits onto the pairwise demographic parity constraint
# set for all C(14,2) = 91 intersectional group pairs via a differentiable cvxpylayer.
########################################################################################################################

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from baseline_model import BaselineModel, get_group_pairs
from configs import fairface_xp_params as xp_params
from load_data import NUM_GROUPS


class FairModel(BaselineModel):
    """
    F-Layer model: end-to-end trainable fairness via a differentiable optimization layer
    enforcing pairwise demographic parity across 14 intersectional groups.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, head_type="linear",
                 p_pos=0.5, freeze_backbone=False, lora=False, lora_rank=8, lora_alpha=16):
        super(FairModel, self).__init__(
            backbone_name, pretrained, head_type, p_pos,
            freeze_backbone, lora, lora_rank, lora_alpha,
        )

        # Primal-dual state for the online algorithm (b < b_tau)
        self.lambda_dual = 0.2
        self.lambda_dual_train = 0.0
        self.dual_update_count = 0
        self.dual_update_count_train = 0
        self.eta_0 = 0.1

    def get_adaptive_eta(self, training=False):
        count = self.dual_update_count_train if training else self.dual_update_count
        return self.eta_0 / np.sqrt(count + 1)

    # ── CvxpyLayer factories ─────────────────────────────────────────────────────

    def create_fair_layer(self, n, group_masks_np):
        """
        Standard fairness layer (b >= b_tau): hard pairwise constraints.

            min  ||yhat - raw||^2
            s.t. |mean(yhat | g_i) - mean(yhat | g_j)| <= slack  for all pairs

        @param n (int): batch size
        @param group_masks_np: dict {group_id: (mask_array, count)} for present groups

        @returns CvxpyLayer with parameters [raw, slack]
        """
        yhat = cp.Variable(n)
        raw = cp.Parameter(n)
        slack = cp.Parameter(1, nonneg=True)

        constr = []
        for g_i, g_j in get_group_pairs(group_masks_np.keys()):
            mask_i, n_i = group_masks_np[g_i]
            mask_j, n_j = group_masks_np[g_j]
            mean_i = (mask_i @ yhat) / n_i
            mean_j = (mask_j @ yhat) / n_j
            constr += [mean_i - mean_j <= slack, -(mean_i - mean_j) <= slack]

        objective = cp.Minimize(cp.sum_squares(yhat - raw))
        problem = cp.Problem(objective, constr)

        return CvxpyLayer(problem, parameters=[raw, slack], variables=[yhat])

    def create_primal_dual_layer(self, n, group_masks_np, batch_size):
        """
        Primal-dual fairness layer (b < b_tau): penalize aggregate gap.

            min  ||yhat - raw||^2 + lambda * batch_size * sum_pairs |gap_ij|

        @param n (int): batch size
        @param group_masks_np: dict {group_id: (mask_array, count)} for present groups
        @param batch_size (int): current batch size

        @returns CvxpyLayer with parameters [raw, lambda_param]
        """
        yhat = cp.Variable(n)
        raw = cp.Parameter(n)
        lambda_param = cp.Parameter(1, nonneg=True)

        pairs = get_group_pairs(group_masks_np.keys())
        gap_vars = []
        constr = []

        for g_i, g_j in pairs:
            mask_i, n_i = group_masks_np[g_i]
            mask_j, n_j = group_masks_np[g_j]
            gap = cp.Variable(1, nonneg=True)
            mean_diff = (mask_i @ yhat) / n_i - (mask_j @ yhat) / n_j
            constr += [gap >= mean_diff, gap >= -mean_diff]
            gap_vars.append(gap)

        penalty = cp.sum(cp.hstack(gap_vars)) if gap_vars else 0.0

        objective = cp.Minimize(
            cp.sum_squares(yhat - raw) + lambda_param[0] * batch_size * penalty
        )
        problem = cp.Problem(objective, constr)

        return CvxpyLayer(problem, parameters=[raw, lambda_param], variables=[yhat])

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _build_group_masks(self, groups_np):
        """
        Build dict of {group_id: (float_mask, count)} for groups present in the batch.
        """
        masks = {}
        for g in range(NUM_GROUPS):
            m = (groups_np == g).astype(float)
            if m.sum() > 0:
                masks[g] = (m, m.sum())
        return masks

    # ── Forward pass ─────────────────────────────────────────────────────────────

    def _solve_fair_plain_cvxpy(self, raw_np, group_masks_np, epsilon):
        """
        Plain CVXPY fallback for hard-constraint projection (inference only).
    
            min  ||yhat - raw||^2
            s.t. |mean(yhat|g_i) - mean(yhat|g_j)| <= epsilon  for all pairs
        """
        import cvxpy as cp
        n = len(raw_np)
        yhat = cp.Variable(n)
    
        constr = []
        for g_i, g_j in get_group_pairs(group_masks_np.keys()):
            mask_i, n_i = group_masks_np[g_i]
            mask_j, n_j = group_masks_np[g_j]
            mean_i = (mask_i @ yhat) / n_i
            mean_j = (mask_j @ yhat) / n_j
            constr += [mean_i - mean_j <= epsilon, -(mean_i - mean_j) <= epsilon]
    
        problem = cp.Problem(cp.Minimize(cp.sum_squares(yhat - raw_np)), constr)
        problem.solve(solver=cp.SCS, eps=1e-8, max_iters=10000,
                    acceleration_lookback=0, verbose=False)
    
        if yhat.value is not None:
            return yhat.value.astype(np.float32)
        return raw_np.copy()
    
    
    def _solve_primal_dual_plain_cvxpy(self, raw_np, group_masks_np, lambda_dual, batch_size):
        """
        Plain CVXPY fallback for primal-dual projection (inference only).
    
            min  ||yhat - raw||^2 + lambda * batch_size * sum_pairs |gap_ij|
        """
        import cvxpy as cp
        n = len(raw_np)
        yhat = cp.Variable(n)
    
        gap_vars, constr = [], []
        for g_i, g_j in get_group_pairs(group_masks_np.keys()):
            mask_i, n_i = group_masks_np[g_i]
            mask_j, n_j = group_masks_np[g_j]
            gap = cp.Variable(1, nonneg=True)
            mean_diff = (mask_i @ yhat) / n_i - (mask_j @ yhat) / n_j
            constr += [gap >= mean_diff, gap >= -mean_diff]
            gap_vars.append(gap)
    
        penalty = cp.sum(cp.hstack(gap_vars)) if gap_vars else 0.0
        objective = cp.Minimize(
            cp.sum_squares(yhat - raw_np) + lambda_dual * batch_size * penalty
        )
        problem = cp.Problem(objective, constr)
        problem.solve(solver=cp.SCS, eps=1e-8, max_iters=10000,
                    acceleration_lookback=0, verbose=False)
    
        if yhat.value is not None:
            return yhat.value.astype(np.float32)
        return raw_np.copy()
    
    def forward(self, images, groups, inference=False):
        """
        Forward pass: images -> backbone -> classifier -> raw logits -> fairness layer.
    
        During training: uses CvxpyLayer (differentiable, supports backprop).
        During inference: tries CvxpyLayer first, falls back to plain CVXPY if diffcp crashes.
        """
        raw = self.classifier(self.backbone(images)).squeeze(1)
        orig_device = raw.device
        n = len(raw)
    
        groups_np = groups.cpu().numpy()
        group_masks_np = self._build_group_masks(groups_np)
    
        if len(group_masks_np) < 2:
            return raw.unsqueeze(1)
    
        raw_cpu = raw.cpu().float()
        epsilon = xp_params.get_slack()
        b_tau = xp_params.b_tau()
    
        if n < b_tau:
            # -- Primal-dual regime --
            lam = self.lambda_dual if inference else self.lambda_dual_train
    
            try:
                layer = self.create_primal_dual_layer(n, group_masks_np, n)
                (ytilde,) = layer(raw_cpu, torch.tensor([lam], dtype=torch.float32))
            except Exception:
                if inference:
                    proj_np = self._solve_primal_dual_plain_cvxpy(
                        raw_cpu.numpy(), group_masks_np, lam, n
                    )
                    ytilde = torch.from_numpy(proj_np).float()
                else:
                    raise
    
            with torch.no_grad():
                means = {}
                for g, (mask, cnt) in group_masks_np.items():
                    means[g] = ytilde[groups_np == g].mean().item()
                max_gap = max(
                    abs(means[gi] - means[gj])
                    for gi, gj in get_group_pairs(means.keys())
                ) if len(means) >= 2 else 0.0
    
                violation = n * (max_gap - epsilon)
                eta = self.get_adaptive_eta(training=not inference)
    
                if inference:
                    self.lambda_dual = max(0.0, self.lambda_dual + eta * violation)
                    self.dual_update_count += 1
                else:
                    self.lambda_dual_train = max(0.0, self.lambda_dual_train + eta * violation)
                    self.dual_update_count_train += 1
        else:
            # -- Standard constraint regime --
            slack_tensor = torch.tensor([epsilon], dtype=torch.float32)
    
            try:
                layer = self.create_fair_layer(n, group_masks_np)
                (ytilde,) = layer(raw_cpu, slack_tensor)
            except Exception:
                if inference:
                    proj_np = self._solve_fair_plain_cvxpy(
                        raw_cpu.numpy(), group_masks_np, epsilon
                    )
                    ytilde = torch.from_numpy(proj_np).float()
                else:
                    raise
    
        return ytilde.to(dtype=raw.dtype, device=orig_device).unsqueeze(1)

    # def forward(self, images, groups, inference=False):
    #     """
    #     Forward pass: images → backbone → classifier → raw logits → fairness layer.

    #     @param images (Tensor): (B, 3, H, W)
    #     @param groups (Tensor): (B,) integer intersectional group IDs (0–13)
    #     @param inference (bool): if True, use inference dual variable

    #     @returns (B, 1) projected logits
    #     """
    #     raw = self.classifier(self.backbone(images)).squeeze(1)  # (B,)
    #     orig_device = raw.device
    #     n = len(raw)

    #     # Build group masks from numpy (cvxpylayers needs known structure at build time)
    #     groups_np = groups.cpu().numpy()
    #     group_masks_np = self._build_group_masks(groups_np)

    #     # Need at least 2 groups in the batch to form constraints
    #     if len(group_masks_np) < 2:
    #         return raw.unsqueeze(1)

    #     # cvxpylayers solves on CPU in float32 (AMP may have produced float16 logits)
    #     raw_cpu = raw.cpu().float()
    #     epsilon = xp_params.get_slack()
    #     b_tau = xp_params.b_tau()

    #     if n < b_tau:
    #         # ── Primal-dual regime ───────────────────────────────────────────
    #         lam = self.lambda_dual if inference else self.lambda_dual_train
    #         layer = self.create_primal_dual_layer(n, group_masks_np, n)
    #         (ytilde,) = layer(raw_cpu, torch.tensor([lam], dtype=torch.float32))

    #         with torch.no_grad():
    #             # Compute max pairwise gap for dual update
    #             means = {}
    #             for g, (mask, cnt) in group_masks_np.items():
    #                 means[g] = ytilde[groups_np == g].mean().item()
    #             max_gap = max(
    #                 abs(means[gi] - means[gj])
    #                 for gi, gj in get_group_pairs(means.keys())
    #             ) if len(means) >= 2 else 0.0

    #             violation = n * (max_gap - epsilon)
    #             eta = self.get_adaptive_eta(training=not inference)

    #             if inference:
    #                 self.lambda_dual = max(0.0, self.lambda_dual + eta * violation)
    #                 self.dual_update_count += 1
    #             else:
    #                 self.lambda_dual_train = max(0.0, self.lambda_dual_train + eta * violation)
    #                 self.dual_update_count_train += 1
    #     else:
    #         # ── Standard constraint regime ───────────────────────────────────
    #         slack_tensor = torch.tensor([epsilon], dtype=torch.float32)
    #         layer = self.create_fair_layer(n, group_masks_np)
    #         (ytilde,) = layer(raw_cpu, slack_tensor)

    #     return ytilde.to(dtype=raw.dtype, device=orig_device).unsqueeze(1)

    def reset_dual_variables(self, inference=False):
        if inference:
            self.lambda_dual = 0.0
            self.dual_update_count = 0
        else:
            self.lambda_dual_train = 0.0
            self.dual_update_count_train = 0