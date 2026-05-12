########################################################################################################################
# This script defines hyperparameters for the CelebA experiments.
#
# All tunable experiment settings live here so that changes propagate automatically
# to every script that imports this module.
########################################################################################################################


# ── Fairness constraint ──────────────────────────────────────────────────────────

def get_slack():
    """
    Demographic parity tolerance on logits:
        |mean(logit | group_i) - mean(logit | group_j)| <= slack
    for all intersectional group pairs.
    """
    return .001


# ── Batching ─────────────────────────────────────────────────────────────────────

def get_batch_size_train():
    """
    Minibatch size during training.
    CelebA is ~2x larger than FairFace, so we use a slightly larger batch.
    """
    return 128#256


def get_batch_size_eval():
    """
    Minibatch size during validation/inference.
    Can be larger than training since no gradients are stored.
    """
    return 128#256


def b_tau():
    """
    Minimum batch size threshold for the primal-dual inference algorithm.
    When b_infer < b_tau, the online algorithm is used.
    """
    return 100


# ── Optimizer & scheduler ────────────────────────────────────────────────────────

def get_backbone_lr():
    """Learning rate for the pretrained backbone (low to preserve features)."""
    return 1e-6


def get_head_lr():
    """Learning rate for the classifier head (higher for faster convergence)."""
    return 1e-4


def get_lr_decay_factor():
    """
    Multiplicative factor applied to the learning rate when validation loss plateaus.
    """
    return 0.8


def get_lr_patience():
    """
    Number of epochs without validation improvement before the learning rate is reduced.
    """
    return 1


def get_min_lr():
    """
    Lower bound on the learning rate after decay.
    """
    return 1e-6


# ── Early stopping ───────────────────────────────────────────────────────────────

def get_early_stop_patience():
    """
    Number of epochs without validation improvement before training is halted.
    """
    return 2


def get_max_epochs():
    """
    Hard upper bound on training epochs (pretrained backbones converge faster).
    """
    return 15


# ── Penalty model ────────────────────────────────────────────────────────────────

def get_default_penalty_lambda():
    """
    Default lambda for the Penalty model (selected via cross-validation in practice).
    """
    return 0.01


def get_strict_penalty_lambda():
    """
    Lambda for the Strict Penalty model (intentionally large to dominate the loss).
    """
    return 10000.0
