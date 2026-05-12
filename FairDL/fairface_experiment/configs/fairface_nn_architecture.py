########################################################################################################################
# Neural network architectures for the FairFace experiments.
#
# Supports 5 architecture families for ablation studies:
#   - ResNet:      resnet18, resnet34, resnet50, resnet101
#   - ViT:         vit_b_16, vit_b_32  (+ optional LoRA adapters)
#   - EfficientNet: efficientnet_b0, efficientnet_b1
#   - MobileNet:   mobilenet_v3_small, mobilenet_v3_large
#   - SimpleCNN:   simple_cnn  (lightweight from-scratch baseline)
#
# All backbones output a (B, num_features) tensor; the classifier head maps to (B, 1).
# Use freeze_backbone() to freeze all backbone params (linear probe / LoRA-only training).
########################################################################################################################

import math
import torch
import torch.nn as nn
from torchvision import models


# ── LoRA adapter ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a frozen nn.Linear layer.

    Adds trainable low-rank matrices A and B such that:
        output = frozen_linear(x) + (x @ A @ B) * scale

    The original weight is frozen; only A and B are trained.
    """

    def __init__(self, original_linear, rank=8, alpha=16):
        """
        @param original_linear (nn.Linear): the layer to adapt
        @param rank (int): LoRA rank (lower = fewer params)
        @param alpha (float): scaling factor (effective scale = alpha / rank)
        """
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scale = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze the original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


def apply_lora(model, rank=8, alpha=16):
    """
    Apply LoRA adapters to the MLP Linear layers in a ViT's encoder blocks.

    Only targets Linear layers inside 'mlp' submodules, avoiding MultiheadAttention
    layers which access .weight directly and cannot be wrapped.

    @param model (nn.Module): the ViT backbone to adapt
    @param rank (int): LoRA rank
    @param alpha (float): LoRA scaling factor
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Collect replacements first to avoid mutating the tree during iteration
    replacements = []
    for name, module in model.named_modules():
        if "mlp" not in name:
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                replacements.append((module, child_name, child))

    # Apply replacements
    for parent, child_name, original_linear in replacements:
        setattr(parent, child_name, LoRALinear(original_linear, rank=rank, alpha=alpha))

    return model


# ── Simple CNN (from-scratch baseline) ───────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for from-scratch training. ~1.2M params.
    Serves as a baseline to show the F-Layer advantage isn't architecture-dependent.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 112 → 56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 56 → 28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 28 → 14
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 5: 14 → 7
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (B, 512, 1, 1)
        )
        self.num_features = 512

    def forward(self, x):
        return self.features(x).flatten(1)  # (B, 512)


# ── Backbone builders ────────────────────────────────────────────────────────────
# Each returns (backbone_module, num_features). The backbone's forward() must
# output a (B, num_features) tensor with the classification head removed.

def _load_pretrained(constructor, pretrained):
    """Helper to handle both old and new torchvision weight APIs."""
    if pretrained:
        try:
            return constructor(weights="DEFAULT")
        except TypeError:
            return constructor(pretrained=True)
    return constructor(weights=None)


def _build_resnet(constructor, num_features, pretrained):
    backbone = _load_pretrained(constructor, pretrained)
    backbone.fc = nn.Identity()
    return backbone, num_features


def _build_vit(constructor, num_features, pretrained):
    backbone = _load_pretrained(constructor, pretrained)
    backbone.heads = nn.Identity()
    return backbone, num_features


def _build_efficientnet(constructor, num_features, pretrained):
    backbone = _load_pretrained(constructor, pretrained)
    backbone.classifier = nn.Identity()
    return backbone, num_features


def _build_densenet(constructor, num_features, pretrained):
    backbone = _load_pretrained(constructor, pretrained)
    backbone.classifier = nn.Identity()
    return backbone, num_features


def _build_swin(constructor, num_features, pretrained):
    backbone = _load_pretrained(constructor, pretrained)
    backbone.head = nn.Identity()
    return backbone, num_features


def _build_mobilenet(constructor, num_features, pretrained):
    backbone = _load_pretrained(constructor, pretrained)
    backbone.classifier = nn.Identity()
    return backbone, num_features


def _build_simple_cnn(pretrained=False):
    """SimpleCNN is always from scratch; pretrained arg is ignored."""
    cnn = SimpleCNN()
    return cnn, cnn.num_features


# ── Registry ─────────────────────────────────────────────────────────────────────
# Maps name → (builder_func, kwargs).  builder_func(pretrained) → (backbone, nf).

_BACKBONE_REGISTRY = {
    # ResNets
    "resnet18":    lambda p: _build_resnet(models.resnet18,  512,  p),
    "resnet34":    lambda p: _build_resnet(models.resnet34,  512,  p),
    "resnet50":    lambda p: _build_resnet(models.resnet50,  2048, p),
    "resnet101":   lambda p: _build_resnet(models.resnet101, 2048, p),

    # DenseNet
    "densenet121": lambda p: _build_densenet(models.densenet121, 1024, p),
    "densenet169": lambda p: _build_densenet(models.densenet169, 1664, p),

    # Vision Transformers
    "vit_b_16":    lambda p: _build_vit(models.vit_b_16, 768, p),
    "vit_b_32":    lambda p: _build_vit(models.vit_b_32, 768, p),

    # Swin Transformer
    "swin_t":      lambda p: _build_swin(models.swin_t,  768, p),
    "swin_s":      lambda p: _build_swin(models.swin_s,  768, p),

    # EfficientNet
    "efficientnet_b0": lambda p: _build_efficientnet(models.efficientnet_b0, 1280, p),
    "efficientnet_b1": lambda p: _build_efficientnet(models.efficientnet_b1, 1280, p),

    # MobileNet
    "mobilenet_v3_small": lambda p: _build_mobilenet(models.mobilenet_v3_small, 576,  p),
    "mobilenet_v3_large": lambda p: _build_mobilenet(models.mobilenet_v3_large, 960,  p),

    # Simple CNN (from scratch)
    "simple_cnn":  lambda p: _build_simple_cnn(p),
}


# ── Public API ───────────────────────────────────────────────────────────────────

def get_backbone(name="resnet18", pretrained=True, freeze=False, lora=False,
                 lora_rank=8, lora_alpha=16):
    """
    Return a feature extractor backbone with its classification head removed.

    @param name (str): key into _BACKBONE_REGISTRY
    @param pretrained (bool): load pretrained weights (ignored for simple_cnn)
    @param freeze (bool): freeze all backbone parameters (linear probe mode)
    @param lora (bool): apply LoRA adapters to transformer layers (only for ViT backbones)
    @param lora_rank (int): LoRA rank
    @param lora_alpha (float): LoRA scaling factor

    @returns (backbone, num_features)
    """
    if name not in _BACKBONE_REGISTRY:
        supported = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{name}'. Supported:\n  {supported}")

    backbone, num_features = _BACKBONE_REGISTRY[name](pretrained)

    # LoRA (ViT only — apply before freeze check so LoRA params stay trainable)
    if lora:
        if not (name.startswith("vit") or name.startswith("swin")):
            raise ValueError(f"LoRA is only supported for ViT/Swin backbones, got '{name}'.")
        backbone = apply_lora(backbone, rank=lora_rank, alpha=lora_alpha)
        print(f"  LoRA applied (rank={lora_rank}, alpha={lora_alpha})")

    # Freeze (applied after LoRA so LoRA adapters remain trainable)
    elif freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen (linear probe mode)")

    return backbone, num_features


def get_classifier_head(num_features, head_type="linear"):
    """
    Classification head: maps (B, num_features) → (B, 1) logit for binary classification.

    @param num_features (int): dimensionality of backbone output
    @param head_type (str): "linear" or "mlp"

    @returns nn.Module
    """
    if head_type == "linear":
        return nn.Linear(num_features, 1)
    elif head_type == "mlp":
        return nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    else:
        raise ValueError(f"Unknown head_type '{head_type}'. Choose from: linear, mlp")