########################################################################################################################
# Cross-validation for the Penalty model's lambda hyperparameter.
#
# Searches from largest lambda to smallest. Once a lambda fails to satisfy the
# constraint on the validation set, all smaller lambdas are skipped (they would
# also fail). The smallest satisfying lambda is selected.
#
# Usage:
#   python cross_val.py --backbone resnet18
#   python cross_val.py --backbone vit_b_16 --lora --batch_size 64
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
from tqdm import tqdm

from load_data import NUM_GROUPS, NUM_AGE
from penalty_model import PenaltyModel, compute_penalty_term
from baseline_model import compute_max_pairwise_gap
from configs import celeba_xp_params as xp_params

from run_celeba_experiments import (
    set_seed,
    StratifiedBatchSampler,
    get_dataloaders,
    unpack_batch,
    evaluate,
)


DEFAULT_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


# ── Train one model for a given lambda ───────────────────────────────────────────

def train_penalty_model(
    lam, train_loader, val_loader, backbone_name, pretrained, head_type, p_pos,
    freeze_backbone, lora, lora_rank, lora_alpha, device, use_amp, max_epochs=None,
):
    """
    Train a penalty model with a specific lambda value. Returns the best model
    state (by val loss) and the val metrics at that state.
    """
    max_epochs = max_epochs or xp_params.get_max_epochs()
    model = PenaltyModel(
        backbone_name=backbone_name, pretrained=pretrained, head_type=head_type,
        p_pos=p_pos, freeze_backbone=freeze_backbone, lora=lora,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
    ).to(device)

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

    best_val_loss = float("inf")
    best_state = None
    best_val_metrics = None
    patience = 0
    best_val_acc_cv = 0.0
    stagnation_patience_cv = 5
    min_acc_threshold_cv = 0.52

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images, labels, groups = unpack_batch(batch, device)
            if len(torch.unique(groups)) < 2:
                continue

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).squeeze(1)
                bce = criterion(logits, labels)
                penalty = compute_penalty_term(logits, groups)
                loss = bce + lam * penalty

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_loader, criterion, "penalty", device, use_amp)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics.get("accuracy", 0)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            patience = 0
        else:
            patience += 1
            if patience >= xp_params.get_early_stop_patience():
                break

        # Stagnation: abort if accuracy never exceeds chance
        best_val_acc_cv = max(best_val_acc_cv, val_acc)
        if epoch >= stagnation_patience_cv and best_val_acc_cv <= min_acc_threshold_cv:
            break

    return best_state, best_val_metrics


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)

    # Load data
    splits_dir = Path(args.splits_dir)
    cached = [splits_dir / f"{s}_dataset.pt" for s in ("train", "val", "test")]
    assert all(p.exists() for p in cached), \
        f"Run load_data.py first to create splits in {splits_dir}"

    train_dataset = torch.load(cached[0], weights_only=False)
    val_dataset = torch.load(cached[1], weights_only=False)
    test_dataset = torch.load(cached[2], weights_only=False)

    train_loader, val_loader, _ = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        bs_train=args.batch_size,
    )

    p_pos = train_dataset.tensors[1][:, 0].float().mean().item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"\n{'='*70}")
    print(f"  Cross-validation for Penalty lambda")
    print(f"  Backbone: {args.backbone}, Device: {device}")
    print(f"  Lambdas: {DEFAULT_LAMBDAS}")
    print(f"{'='*70}\n")

    results = []
    best_lambda = DEFAULT_LAMBDAS[-1]  # default: largest
    slack = xp_params.get_slack()

    # Search from largest to smallest
    for lam in reversed(DEFAULT_LAMBDAS):
        print(f"  λ = {lam:.4f} ...")
        t0 = time.time()

        state, val_metrics = train_penalty_model(
            lam, train_loader, val_loader,
            args.backbone, args.pretrained, args.head,
            p_pos, args.freeze_backbone, args.lora,
            args.lora_rank, args.lora_alpha, device, use_amp,
            max_epochs=args.cv_max_epochs,
        )

        gap = val_metrics["max_pairwise_gap"]
        satisfied = gap <= slack
        elapsed = time.time() - t0

        print(f"    gap={gap:.6f}, satisfied={satisfied}, "
              f"acc={val_metrics['accuracy']:.4f}, time={elapsed:.1f}s")

        results.append({"lambda": lam, "gap": gap, "satisfied": satisfied,
                        **val_metrics})

        if satisfied:
            best_lambda = lam
        else:
            print(f"    Constraint violated — stopping (smaller lambdas would also fail)")
            break

    print(f"\n  Best lambda: {best_lambda}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    suffix = args.backbone
    if args.lora:
        suffix += f"_lora_r{args.lora_rank}"

    # Save best lambda
    lambda_path = results_dir / f"best_lambda_{suffix}.txt"
    with open(lambda_path, "w") as f:
        f.write(str(best_lambda))
    print(f"  Saved best lambda → {lambda_path}")

    # Save CV results
    cv_path = results_dir / f"cv_results_{suffix}.csv"
    if results:
        with open(cv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved CV results  → {cv_path}")


def get_args():
    ap = argparse.ArgumentParser(description="Cross-validate penalty lambda for CelebA.")
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    ap.add_argument("--freeze_backbone", action="store_true", default=False)
    ap.add_argument("--lora", action="store_true", default=False)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits_dir", type=str, default="celeba_splits")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--cv_max_epochs", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    main()