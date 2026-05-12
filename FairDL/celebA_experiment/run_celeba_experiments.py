########################################################################################################################
# Main driver for training and evaluating all model types in the CelebA experiments.
#
# Protected groups: 4 intersectional (Male × Young). Constraint: pairwise demographic
# parity across all C(4,2) = 6 pairs.
#
# Usage:
#   python run_celeba_experiments.py --model_type fair
#   python run_celeba_experiments.py --model_type baseline --backbone resnet34
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

from load_data import load_celeba_splits, NUM_GROUPS, NUM_AGE
from baseline_model import BaselineModel, compute_max_pairwise_gap
from penalty_model import PenaltyModel, compute_penalty_term
from fair_model import FairModel
from configs import celeba_xp_params as xp_params


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

    # Stratify all loaders on intersectional group (male * 2 + young)
    # This ensures train and eval batches have identical group composition,
    # so per-batch constraints are equivalent to population-level constraints.
    def make_loader(dataset, bs, shuffle):
        meta = dataset.tensors[1]
        groups = (meta[:, 1] * NUM_AGE + meta[:, 2]).numpy()
        sampler = StratifiedBatchSampler(groups, bs, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=sampler)

    return (
        make_loader(train_dataset, bs_train, shuffle=True),
        make_loader(val_dataset, bs_eval, shuffle=False),
        make_loader(test_dataset, bs_eval, shuffle=False),
    )


def unpack_batch(batch, device):
    """Unpack a batch from the TensorDataset into (images, labels, groups)."""
    images, meta = batch
    images = images.to(device)
    labels = meta[:, 0].float().to(device)            # target (Smiling)
    groups = (meta[:, 1] * NUM_AGE + meta[:, 2]).to(device)  # male * 2 + young
    return images, labels, groups


# ── Loss computation ─────────────────────────────────────────────────────────────

def compute_loss(model, images, labels, groups, model_type, criterion, penalty_lambda=None):
    """Compute loss for any model type."""
    if model_type == "fair":
        projected = model(images, groups).squeeze(1)
        return criterion(projected, labels)

    elif model_type in ("penalty", "strict_penalty"):
        logits = model(images).squeeze(1)
        bce = criterion(logits, labels)
        penalty = compute_penalty_term(logits, groups)
        lam = penalty_lambda if model_type == "penalty" else xp_params.get_strict_penalty_lambda()
        return bce + lam * penalty

    else:  # baseline
        logits = model(images).squeeze(1)
        return criterion(logits, labels)


# ── Evaluation ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, model_type, device, use_amp):
    """Evaluate model on a loader. Returns metrics dict."""
    model.eval()

    if model_type == "fair":
        return _evaluate_fair(model, loader, criterion, device, use_amp)
    elif model_type == "baseline":
        return _evaluate_baseline(model, loader, criterion, device, use_amp)
    else:
        return _evaluate_penalty(model, loader, criterion, device, use_amp)


@torch.no_grad()
def _evaluate_fair(model, loader, criterion, device, use_amp):
    """Evaluate F-Layer model (projects logits via cvxpylayer)."""
    all_proj, all_labels, all_groups = [], [], []

    for batch in loader:
        images, labels, groups = unpack_batch(batch, device)
        with autocast(device_type=device.type, enabled=use_amp):
            projected = model(images, groups, inference=True).squeeze(1)
        all_proj.append(projected.cpu().float())
        all_labels.append(labels.cpu())
        all_groups.append(groups.cpu())

    proj = torch.cat(all_proj)
    labels = torch.cat(all_labels)
    groups = torch.cat(all_groups)

    gap = compute_max_pairwise_gap(proj, groups)
    slack = xp_params.get_slack()

    return {
        "loss": criterion(proj, labels).item(),
        "accuracy": ((proj >= 0).float() == labels).float().mean().item(),
        "max_pairwise_gap": gap,
        "logit_std": proj.std().item(),
        "constraint_satisfied": gap <= slack,
    }


@torch.no_grad()
def _evaluate_baseline(model, loader, criterion, device, use_amp):
    """Evaluate Projection baseline (projects at inference only)."""
    all_raw, all_proj, all_labels, all_groups = [], [], [], []

    for batch in loader:
        images, labels, groups = unpack_batch(batch, device)
        with autocast(device_type=device.type, enabled=use_amp):
            raw = model(images).squeeze(1)

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
        "loss_raw": criterion(raw, labels).item(),
        "loss_proj": criterion(proj, labels).item(),
        "accuracy_raw": ((raw >= 0).float() == labels).float().mean().item(),
        "accuracy_proj": ((proj >= 0).float() == labels).float().mean().item(),
        "gap_raw": gap_raw,
        "gap_proj": gap_proj,
        "logit_std_raw": raw.std().item(),
        "logit_std_proj": proj.std().item(),
        "constraint_satisfied": gap_proj <= slack,
        "solver_status": status,
    }


@torch.no_grad()
def _evaluate_penalty(model, loader, criterion, device, use_amp):
    """Evaluate Penalty / Strict Penalty model (no projection)."""
    all_logits, all_labels, all_groups = [], [], []

    for batch in loader:
        images, labels, groups = unpack_batch(batch, device)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images).squeeze(1)
        all_logits.append(logits.cpu().float())
        all_labels.append(labels.cpu())
        all_groups.append(groups.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    groups = torch.cat(all_groups)

    gap = compute_max_pairwise_gap(logits, groups)
    slack = xp_params.get_slack()

    return {
        "loss": criterion(logits, labels).item(),
        "accuracy": ((logits >= 0).float() == labels).float().mean().item(),
        "max_pairwise_gap": gap,
        "logit_std": logits.std().item(),
        "constraint_satisfied": gap <= slack,
    }


# ── Training loop ────────────────────────────────────────────────────────────────

def train_models(args):
    set_seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading CelebA dataset...")
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
        train_dataset, val_dataset, test_dataset = load_celeba_splits(
            val_fraction=args.val_fraction, data_seed=args.seed, image_size=args.image_size,
        )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        bs_train=args.batch_size,
    )

    # ── Compute positive class rate (for bias initialization) ────────────────────
    p_pos = train_dataset.tensors[1][:, 0].float().mean().item()
    print(f"  Target positive rate: {p_pos:.3f}")

    # ── Penalty lambda ───────────────────────────────────────────────────────────
    penalty_lambda = None
    if args.model_type == "penalty":
        penalty_lambda = args.penalty_lambda or xp_params.get_default_penalty_lambda()
        print(f"  Penalty lambda: {penalty_lambda}")
    elif args.model_type == "strict_penalty":
        penalty_lambda = xp_params.get_strict_penalty_lambda()
        print(f"  Strict penalty lambda: {penalty_lambda}")

    # ── Build model ──────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building {args.model_type} model (backbone={args.backbone})...")
    model = create_model(
        args.model_type, args.backbone, args.pretrained, args.head,
        p_pos, args.freeze_backbone, args.lora, args.lora_rank, args.lora_alpha,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Device: {device} (AMP: {'on' if use_amp else 'off'})")
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ── Optimizer (differential LR for backbone vs head) ─────────────────────────
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": xp_params.get_backbone_lr()},
        {"params": head_params, "lr": xp_params.get_head_lr()},
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=xp_params.get_lr_decay_factor(),
        patience=xp_params.get_lr_patience(), min_lr=xp_params.get_min_lr(),
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=use_amp)

    # ── Train ────────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training (max {xp_params.get_max_epochs()} epochs, "
          f"early stop patience={xp_params.get_early_stop_patience()})...")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_history = []
    total_training_time = 0.0

    # Stagnation detector: stop if accuracy doesn't improve beyond chance
    best_val_acc = 0.0
    stagnation_patience = 5       # epochs to wait for meaningful accuracy gain
    min_acc_threshold = 0.52      # must beat this (just above majority-class rate)
    stagnation_counter = 0

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

            n_present = len(torch.unique(groups))
            if n_present < 2:
                continue

            with autocast(device_type=device.type, enabled=use_amp):
                loss = compute_loss(model, images, labels, groups, args.model_type,
                                    criterion, penalty_lambda)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=8.0)
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
        scheduler.step(val_metrics.get("loss", val_metrics.get("loss_proj", 0)))

        epoch_time = time.time() - epoch_start
        total_training_time += epoch_time
        current_lr = optimizer.param_groups[-1]["lr"]
        train_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": current_lr,
            "epoch_time_sec": round(epoch_time, 2),
        })

        # Pick validation loss key based on model type
        val_loss_key = "loss_proj" if args.model_type == "baseline" else "loss"
        val_loss = val_metrics.get(val_loss_key, val_metrics.get("loss", float("inf")))

        if epoch % 1 == 0 or epoch == xp_params.get_max_epochs() - 1:
            val_gap_key = "gap_proj" if args.model_type == "baseline" else "max_pairwise_gap"
            val_acc_key = "accuracy_proj" if args.model_type == "baseline" else "accuracy"
            print(f"  Epoch {epoch:>3d} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_metrics.get(val_acc_key, 0):.4f} | "
                  f"val_gap={val_metrics.get(val_gap_key, 0):.6f} | "
                  f"lr={current_lr:.2e} | "
                  f"{epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= xp_params.get_early_stop_patience():
                print(f"\n  Early stopping at epoch {epoch} (patience={xp_params.get_early_stop_patience()})")
                break

        # ── Stagnation detection ─────────────────────────────────────────
        # If accuracy never meaningfully exceeds majority-class rate, the
        # model is stuck (e.g., penalty term overwhelms BCE). Abort early.
        val_acc = val_metrics.get(val_acc_key, 0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if best_val_acc > min_acc_threshold:
                stagnation_counter = 0  # real learning detected, reset
        else:
            stagnation_counter += 1

        if epoch >= stagnation_patience and best_val_acc <= min_acc_threshold:
            print(f"\n  Stagnation detected at epoch {epoch}: best accuracy "
                  f"{best_val_acc:.4f} never exceeded {min_acc_threshold}. Stopping.")
            break

    # ── Load best model & evaluate on test set ───────────────────────────────────
    print(f"\n[4/4] Evaluating best model on test set...")
    model.load_state_dict(best_state)
    model = model.to(device)

    if args.model_type == "fair":
        model.reset_dual_variables(inference=True)

    test_metrics = evaluate(model, test_loader, criterion, args.model_type, device, use_amp)

    if args.model_type == "baseline":
        print(f"  Test — loss={test_metrics['loss_proj']:.4f}, "
              f"acc={test_metrics['accuracy_proj']:.4f}, "
              f"gap_raw={test_metrics['gap_raw']:.4f}, "
              f"gap_proj={test_metrics['gap_proj']:.6f}, "
              f"logit_std={test_metrics['logit_std_proj']:.4f}")
    else:
        print(f"  Test — loss={test_metrics['loss']:.4f}, "
              f"acc={test_metrics['accuracy']:.4f}, "
              f"gap={test_metrics['max_pairwise_gap']:.6f}, "
              f"logit_std={test_metrics['logit_std']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

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
    ap = argparse.ArgumentParser(description="Train CelebA fairness models (intersectional constraints).")
    ap.add_argument("--model_type", type=str, required=True,
                    choices=["fair", "baseline", "penalty", "strict_penalty"])
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Training batch size (overrides config).")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    ap.add_argument("--freeze_backbone", action="store_true", default=False)
    ap.add_argument("--lora", action="store_true", default=False)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--penalty_lambda", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--splits_dir", type=str, default="celeba_splits")
    ap.add_argument("--results_dir", type=str, default="results")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_models(args)