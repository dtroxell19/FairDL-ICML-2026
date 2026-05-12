########################################################################################################################
# This script defines the Lagrangian penalty model for the FairFace experiments.
#
# L_penalty = BCE(logits, labels) + lambda * sum_over_pairs (mean_gi - mean_gj)^2
########################################################################################################################

from baseline_model import BaselineModel, compute_sum_squared_gaps


class PenaltyModel(BaselineModel):
    """
    Same architecture as the Projection baseline. Fairness is encouraged via a
    penalty term in the loss rather than a post-hoc projection.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, head_type="linear",
                 p_pos=0.5, freeze_backbone=False, lora=False, lora_rank=8, lora_alpha=16):
        super(PenaltyModel, self).__init__(
            backbone_name, pretrained, head_type, p_pos,
            freeze_backbone, lora, lora_rank, lora_alpha,
        )


def compute_penalty_term(logits, groups):
    """
    Penalty term: sum of squared pairwise mean-logit gaps across all intersectional
    groups present in the batch.

    @param logits (Tensor): (N,) raw logits
    @param groups (Tensor): (N,) integer group IDs (0–13)

    @returns differentiable scalar tensor
    """
    return compute_sum_squared_gaps(logits, groups)