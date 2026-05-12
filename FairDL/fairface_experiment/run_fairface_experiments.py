########################################################################################################################
# Main driver for training and evaluating all model types in the FairFace experiments.
#
# Protected groups: 14 intersectional (gender × race). Constraint: pairwise demographic
# parity across all C(14,2) = 91 pairs.
#
# Usage:
#   python run_fairface_experiments.py --model_type fair
#   python run_fairface_experiments.py --model_type baseline --backbone resnet34
########################################################################################################################

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from load_data import load_fairface_splits, NUM_GROUPS, NUM_RACES
from baseline_model import BaselineModel, compute_max_pairwise_gap
from penalty_model import PenaltyModel, compute_penalty_term
from fair_model import FairModel
from configs import fairface_xp_params as xp_params


# ── Reproducibility ──────────────────────────────────────────────────────────────

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Stratified batch sampler (on intersectional groups) ──────────────────────────

class StratifiedBatchSampler(Sampler):
    """
    Ensures every batch mirrors the population ratio of all intersectional groups.
    This makes per-batch fairness constraints equivalent to population-level constraints.
    """

    def __init__(self, group_labels, batch_size, shuffle=False):
        self.labels = np.asarray(group_labels)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.unique_groups = np.unique(self.labels)
        self.group_indices = {g: np.where(self.labels == g)[0] for g in self.unique_groups}
        self.group_ratios = {g: len(idx) / len(self.labels) for g, idx in self.group_indices.items()}

        # Allocate samples per group per batch, proportional to population
        self.group_batch_sizes = {}
        assigned = 0
        groups_sorted = sorted(self.unique_groups)
        for i, g in enumerate(groups_sorted):
            if i == len(groups_sorted) - 1:
                self.group_batch_sizes[g] = batch_size - assigned
            else:
                n = max(1, int(round(self.group_ratios[g] * batch_size)))
                self.group_batch_sizes[g] = n
                assigned += n

    def __iter__(self):
        group_idx = {}
        for g, idx in self.group_indices.items():
            group_idx[g] = idx[np.random.permutation(len(idx))] if self.shuffle else idx.copy()

        group_pos = {g: 0 for g in self.unique_groups}
        n_batches = len(self.labels) // self.batch_size

        for _ in range(n_batches):
            batch = []
            for g in sorted(self.unique_groups):
                n_take = self.group_batch_sizes[g]
                start = group_pos[g]
                end = start + n_take
                pool = group_idx[g]

                if end <= len(pool):
                    batch.extend(pool[start:end].tolist())
                else:
                    remaining = pool[start:].tolist()
                    if self.shuffle:
                        group_idx[g] = pool[np.random.permutation(len(pool))]
                    needed = n_take - len(remaining)
                    batch.extend(remaining)
                    batch.extend(group_idx[g][:needed].tolist())
                    end = needed

                group_pos[g] = end % len(pool)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size


# ── Model factory ────────────────────────────────────────────────────────────────

def create_model(model_type, backbone_name, pretrained, head_type, p_pos,
                 freeze_backbone=False, lora=False, lora_rank=8, lora_alpha=16):
    common = dict(backbone_name=backbone_name, pretrained=pretrained,
                  head_type=head_type, p_pos=p_pos,
                  freeze_backbone=freeze_backbone, lora=lora,
                  lora_rank=lora_rank, lora_alpha=lora_alpha)
    if model_type == "fair":
        return FairModel(**common)
    elif model_type in ("penalty", "strict_penalty"):
        return PenaltyModel(**common)
    else:
        return BaselineModel(**common)


# ── Data helpers ─────────────────────────────────────────────────────────────────

def get_dataloaders(train_dataset, val_dataset, test_dataset, bs_train=None, bs_eval=None):
    bs_train = bs_train or xp_params.get_batch_size_train()
    bs_eval = bs_eval or xp_params.get_batch_size_eval()

    # Stratify all loaders on intersectional group (gender * 7 + race)
    # This ensures train and eval batches have identical group composition,
    # so per-batch constraints are equivalent to population-level constraints.
    train_meta = train_dataset.tensors[1]
    val_meta = val_dataset.tensors[1]
    test_meta = test_dataset.tensors[1]
    train_groups = (train_meta[:, 1] * NUM_RACES + train_meta[:, 2]).numpy()
    val_groups = (val_meta[:, 1] * NUM_RACES + val_meta[:, 2]).numpy()
    test_groups = (test_meta[:, 1] * NUM_RACES + test_meta[:, 2]).numpy()

    train_loader = DataLoader(train_dataset,
                              batch_sampler=StratifiedBatchSampler(train_groups, bs_train, shuffle=True))
    val_loader = DataLoader(val_dataset,
                            batch_sampler=StratifiedBatchSampler(val_groups, bs_eval))
    test_loader = DataLoader(test_dataset,
                             batch_sampler=StratifiedBatchSampler(test_groups, bs_eval))

    print(f"  Train batch size: {bs_train} ({len(train_loader)} batches, stratified)")
    print(f"  Eval batch size:  {bs_eval} (stratified)")
    print(f"  All loaders stratified on {NUM_GROUPS} intersectional groups")

    return train_loader, val_loader, test_loader


def unpack_batch(batch, device="cpu"):
    """
    Unpack batch and compute intersectional group IDs.

    @returns (images, labels, groups) where groups = gender * 7 + race
    """
    images, meta = batch
    images = images.to(device)
    labels = meta[:, 0].float().to(device)
    gender = meta[:, 1].to(device)
    race = meta[:, 2].to(device)
    groups = gender * NUM_RACES + race  # 0..13
    return images, labels, groups


# ── Loss computation ─────────────────────────────────────────────────────────────

def compute_loss(model, images, labels, groups, model_type, criterion, penalty_lambda):
    if model_type == "fair":
        logits = model(images, groups).squeeze(1)
        loss = criterion(logits, labels)
    elif model_type in ("penalty", "strict_penalty"):
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels) + penalty_lambda * compute_penalty_term(logits, groups)
    else:
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
    return loss


# ── Evaluation ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, model_type, device="cpu", use_amp=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    all_logits, all_groups = [], []

    for batch in loader:
        images, labels, groups = unpack_batch(batch, device)

        with autocast(device_type=device.type, enabled=use_amp):
            if model_type == "fair":
                logits = model(images, groups, inference=True).squeeze(1)
            else:
                logits = model(images).squeeze(1)

            batch_loss = criterion(logits, labels)

        total_loss += batch_loss.item() * len(labels)
        correct += ((logits >= 0.0).float() == labels).sum().item()
        n_samples += len(labels)
        all_logits.append(logits.float().cpu())
        all_groups.append(groups.cpu())

    all_logits = torch.cat(all_logits)
    all_groups = torch.cat(all_groups)
    max_gap = compute_max_pairwise_gap(all_logits, all_groups)
    slack = xp_params.get_slack()

    return {
        "loss": total_loss / max(n_samples, 1),
        "accuracy": correct / max(n_samples, 1),
        "max_pairwise_gap": max_gap,
        "logit_std": all_logits.std().item(),
        "constraint_satisfied": max_gap <= 1.01*slack,
    }


@torch.no_grad()
def evaluate_with_projection(model, loader, criterion, device="cpu", use_amp=False):
    """Evaluate baseline with post-hoc projection."""
    model.eval()
    all_raw, all_proj, all_labels, all_groups = [], [], [], []

    for batch in loader:
        images, labels, groups = unpack_batch(batch, device)

        with autocast(device_type=device.type, enabled=use_amp):
            raw = model(images).squeeze(1)

        # Projection runs on CPU in float32
        raw_cpu = raw.float().cpu()
        proj_np, status = model.fair_projection(raw_cpu.numpy(), groups.cpu().numpy())
        proj = torch.from_numpy(proj_np).float()

        all_raw.append(raw_cpu)
        all_proj.append(proj)
        all_labels.append(labels.cpu())
        all_groups.append(groups.cpu())

    raw = torch.cat(all_raw)
    proj = torch.cat(all_proj)
    labels = torch.cat(all_labels)
    groups = torch.cat(all_groups)

    slack = xp_params.get_slack()
    gap_raw = compute_max_pairwise_gap(raw, groups)
    gap_proj = compute_max_pairwise_gap(proj, groups)

    return {
        "loss_raw":     criterion(raw, labels).item(),
        "loss_proj":    criterion(proj, labels).item(),
        "accuracy_raw": ((raw >= 0).float() == labels).float().mean().item(),
        "accuracy_proj": ((proj >= 0).float() == labels).float().mean().item(),
        "gap_raw":      gap_raw,
        "gap_proj":     gap_proj,
        "logit_std_raw":  raw.std().item(),
        "logit_std_proj": proj.std().item(),
        "constraint_satisfied": gap_proj <= slack,
        "solver_status": status,
    }


# ── Training loop ────────────────────────────────────────────────────────────────

def train_models(args):
    set_seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading FairFace dataset...")
    splits_dir = Path(args.splits_dir)
    cached = [splits_dir / f"{s}_dataset.pt" for s in ("train", "val", "test")]

    if all(p.exists() for p in cached):
        print(f"  Found cached splits in {splits_dir}, loading from disk...")
        train_dataset = torch.load(cached[0], weights_only=False)
        val_dataset = torch.load(cached[1], weights_only=False)
        test_dataset = torch.load(cached[2], weights_only=False)
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    else:
        print(f"  No cached splits at {splits_dir}, loading from HuggingFace...")
        train_dataset, val_dataset, test_dataset = load_fairface_splits(
            val_fraction=args.val_fraction, data_seed=args.seed, image_size=args.image_size,
        )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        bs_train=args.batch_size,
    )

    p_pos = train_dataset.tensors[1][:, 0].float().mean().item()
    print(f"  Positive class rate (30+): {p_pos:.3f}")

    # ── Create model ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Creating {args.model_type} model (backbone={args.backbone}, head={args.head})...")
    model = create_model(
        args.model_type, args.backbone, args.pretrained, args.head, p_pos,
        freeze_backbone=args.freeze_backbone, lora=args.lora,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    print(f"  Device: {device} (AMP: {'on' if use_amp else 'off'})")

    # ── Optimizer ────────────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    # Differential learning rates: backbone gets a low lr to preserve pretrained
    # features, classifier head gets a higher lr for faster convergence.
    # For frozen/LoRA backbones, only trainable params are in the optimizer.
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": xp_params.get_backbone_lr()})
    param_groups.append({"params": head_params, "lr": xp_params.get_head_lr()})

    optimizer = optim.AdamW(param_groups)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=xp_params.get_lr_decay_factor(),
        patience=xp_params.get_lr_patience(), min_lr=xp_params.get_min_lr(),
    )

    backbone_lr = xp_params.get_backbone_lr() if backbone_params else 0.0
    print(f"  Backbone lr: {backbone_lr:.1e}, Head lr: {xp_params.get_head_lr():.1e}")

    if args.model_type == "strict_penalty":
        penalty_lambda = xp_params.get_strict_penalty_lambda()
    elif args.model_type == "penalty":
        penalty_lambda = args.penalty_lambda
    else:
        penalty_lambda = 0.0

    # ── Training ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training (max {xp_params.get_max_epochs()} epochs, "
          f"patience={xp_params.get_early_stop_patience()})...")

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_history = []
    training_start = time.time()

    # Pre-training evaluation (before any weight updates)
    if args.model_type == "fair":
        model.reset_dual_variables(inference=True)
    pre_metrics = evaluate(model, val_loader, criterion, args.model_type, device, use_amp)
    print(f"  Pre-train | "
          f"val_loss={pre_metrics['loss']:.4f} | "
          f"val_acc={pre_metrics['accuracy']:.4f} | "
          f"val_gap={pre_metrics['max_pairwise_gap']:.4f} | "
          f"logit_std={pre_metrics['logit_std']:.4f}")

    for epoch in range(xp_params.get_max_epochs()):
        epoch_start = time.time()

        if args.model_type == "fair":
            model.reset_dual_variables(inference=False)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch:>3d}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

        for batch in pbar:
            images, labels, groups = unpack_batch(batch, device)

            # Need at least 2 groups present to form constraints
            n_present = len(torch.unique(groups))
            if n_present < 2:
                continue

            with autocast(device_type=device.type, enabled=use_amp):
                loss = compute_loss(model, images, labels, groups, args.model_type, criterion, penalty_lambda)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validate ─────────────────────────────────────────────────────────
        if args.model_type == "fair":
            model.reset_dual_variables(inference=True)

        val_metrics = evaluate(model, val_loader, criterion, args.model_type, device, use_amp)
        scheduler.step(val_metrics["loss"])

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[-1]["lr"]  # head lr
        train_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": current_lr,
            "epoch_time_sec": round(epoch_time, 2),
        })

        if epoch % 1 == 0 or epoch == xp_params.get_max_epochs() - 1:
            print(f"  Epoch {epoch:>3d} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | "
                  f"val_acc={val_metrics['accuracy']:.4f} | "
                  f"val_gap={val_metrics['max_pairwise_gap']:.4f} | "
                  f"logit_std={val_metrics['logit_std']:.4f} | "
                  f"lr={current_lr:.2e} | {epoch_time:.1f}s")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= xp_params.get_early_stop_patience():
                print(f"  Early stopping at epoch {epoch}")
                break

    total_training_time = time.time() - training_start
    print(f"  Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best model (val_loss={best_val_loss:.4f})")

    # ── Test ─────────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating on test set...")
    if args.model_type == "fair":
        model.reset_dual_variables(inference=True)

    if args.model_type == "baseline":
        test_metrics = evaluate_with_projection(model, test_loader, criterion, device, use_amp)
        print(f"  Raw  — loss={test_metrics['loss_raw']:.4f}, "
              f"acc={test_metrics['accuracy_raw']:.4f}, max_gap={test_metrics['gap_raw']:.4f}")
        print(f"  Proj — loss={test_metrics['loss_proj']:.4f}, "
              f"acc={test_metrics['accuracy_proj']:.4f}, max_gap={test_metrics['gap_proj']:.4f}")
        print(f"  Solver: {test_metrics['solver_status']}")
    else:
        test_metrics = evaluate(model, test_loader, criterion, args.model_type, device, use_amp)
        print(f"  loss={test_metrics['loss']:.4f}, "
              f"acc={test_metrics['accuracy']:.4f}, max_gap={test_metrics['max_pairwise_gap']:.4f}, "
              f"logit_std={test_metrics['logit_std']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Build a descriptive suffix for filenames
    suffix = f"{args.model_type}_{args.backbone}"
    if args.lora:
        suffix += f"_lora_r{args.lora_rank}"
    elif args.freeze_backbone:
        suffix += "_frozen"

    history_path = results_dir / f"train_history_{suffix}.csv"
    if train_history:
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=train_history[0].keys())
            writer.writeheader()
            writer.writerows(train_history)
        print(f"\n  Training history → {history_path}")

    test_metrics["total_training_time_sec"] = round(total_training_time, 2)
    test_metrics["num_epochs"] = len(train_history)
    test_metrics["model_type"] = args.model_type
    test_metrics["backbone"] = args.backbone
    test_metrics["head"] = args.head
    test_metrics["lora"] = args.lora
    test_metrics["freeze_backbone"] = args.freeze_backbone
    if args.model_type in ("penalty", "strict_penalty"):
        test_metrics["penalty_lambda"] = penalty_lambda
    results_path = results_dir / f"test_results_{suffix}.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
        writer.writeheader()
        writer.writerow(test_metrics)
    print(f"  Test results     → {results_path}")

    model_dir = results_dir / "trained_models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Model weights    → {model_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def get_args():
    ap = argparse.ArgumentParser(description="Train FairFace fairness models (intersectional constraints).")
    ap.add_argument("--model_type", type=str, required=True,
                    choices=["fair", "baseline", "penalty", "strict_penalty"])
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Training batch size (overrides config; default: from fairface_xp_params).")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    ap.add_argument("--freeze_backbone", action="store_true", default=False,
                    help="Freeze backbone weights (linear probe mode).")
    ap.add_argument("--lora", action="store_true", default=False,
                    help="Apply LoRA adapters to ViT backbone (freezes base weights, trains adapters).")
    ap.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (default: 8).")
    ap.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling (default: 16).")
    ap.add_argument("--penalty_lambda", type=float, default=xp_params.get_default_penalty_lambda())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--splits_dir", type=str, default="fairface_splits")
    ap.add_argument("--results_dir", type=str, default="results")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_models(args)