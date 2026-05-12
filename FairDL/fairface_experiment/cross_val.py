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

from load_data import NUM_GROUPS, NUM_RACES
from penalty_model import PenaltyModel, compute_penalty_term
from baseline_model import compute_max_pairwise_gap
from configs import fairface_xp_params as xp_params

from run_fairface_experiments import (
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
    model = PenaltyModel(
        backbone_name=backbone_name, pretrained=pretrained, head_type=head_type,
        p_pos=p_pos, freeze_backbone=freeze_backbone, lora=lora,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=use_amp)

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

    best_val_loss = float("inf")
    best_model_state = None
    best_val_metrics = None
    patience_counter = 0

    max_epochs = max_epochs or xp_params.get_max_epochs()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"    λ={lam:<8g} Epoch {epoch:>3d}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        for batch in pbar:
            images, labels, groups = unpack_batch(batch, device)
            if len(torch.unique(groups)) < 2:
                continue

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels) + lam * compute_penalty_term(logits, groups)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        val_metrics = evaluate(model, val_loader, criterion, "penalty", device, use_amp)
        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= xp_params.get_early_stop_patience():
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        best_val_metrics = evaluate(model, val_loader, criterion, "penalty", device, use_amp)

    return best_model_state, best_val_loss, best_val_metrics


# ── Main ─────────────────────────────────────────────────────────────────────────

def run_cross_val(args):
    set_seed(args.seed)

    print("\n[1/3] Loading FairFace dataset...")
    splits_dir = Path(args.splits_dir)
    cached = [splits_dir / f"{s}_dataset.pt" for s in ("train", "val", "test")]

    if all(p.exists() for p in cached):
        print(f"  Loading cached splits from {splits_dir}...")
        train_dataset = torch.load(cached[0], weights_only=False)
        val_dataset = torch.load(cached[1], weights_only=False)
        test_dataset = torch.load(cached[2], weights_only=False)
    else:
        raise FileNotFoundError(f"Cached splits not found in {splits_dir}. Run load_data.py first.")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, bs_train=args.batch_size,
    )

    p_pos = train_dataset.tensors[1][:, 0].float().mean().item()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    slack = xp_params.get_slack()

    # Sort lambdas largest → smallest
    lambdas = sorted(args.lambdas, reverse=True)

    print(f"\n[2/3] Searching lambda (largest → smallest, early stop on first failure)...")
    print(f"  Slack (epsilon): {slack}")
    print(f"  Lambdas (descending): {lambdas}\n")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    suffix = args.backbone
    if args.lora:
        suffix += f"_lora_r{args.lora_rank}"
    csv_path = results_dir / f"cross_val_{suffix}.csv"

    fieldnames = ["lambda", "val_loss", "val_accuracy", "val_max_gap", "val_logit_std",
                  "constraint_satisfied", "time_sec"]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # ── Search largest → smallest ────────────────────────────────────────────────
    chosen_lam = None
    chosen_state = None
    chosen_metrics = None

    for lam in lambdas:
        print(f"  ── Lambda = {lam} {'─'*50}")
        start = time.time()

        best_state, best_loss, val_metrics = train_penalty_model(
            lam, train_loader, val_loader,
            backbone_name=args.backbone, pretrained=args.pretrained,
            head_type=args.head, p_pos=p_pos,
            freeze_backbone=args.freeze_backbone, lora=args.lora,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            device=device, use_amp=use_amp, max_epochs=args.cv_max_epochs,
        )

        elapsed = time.time() - start
        satisfied = val_metrics["max_pairwise_gap"] <= slack*1.01

        row = {
            "lambda": lam,
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_max_gap": round(val_metrics["max_pairwise_gap"], 6),
            "val_logit_std": round(val_metrics["logit_std"], 6),
            "constraint_satisfied": satisfied,
            "time_sec": round(elapsed, 1),
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        status = "✓ SATISFIED" if satisfied else "✗ violated"
        print(f"    val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"max_gap={val_metrics['max_pairwise_gap']:.4f} | "
              f"logit_std={val_metrics['logit_std']:.4f} | "
              f"{status} | {elapsed:.0f}s\n")

        if satisfied:
            # This lambda works — record it and try the next smaller one
            chosen_lam = lam
            chosen_state = best_state
            chosen_metrics = val_metrics
        else:
            # Failed — all smaller lambdas will also fail, stop searching
            print(f"  Lambda {lam} failed → stopping search (smaller lambdas would also fail)\n")
            break

    # ── Report ───────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Result...")

    if chosen_lam is not None:
        print(f"  Chosen lambda: {chosen_lam}")
        print(f"    val_loss={chosen_metrics['loss']:.4f} | "
              f"val_acc={chosen_metrics['accuracy']:.4f} | "
              f"max_gap={chosen_metrics['max_pairwise_gap']:.4f}")

        model_dir = results_dir / "trained_models"
        model_dir.mkdir(exist_ok=True)
        model_suffix = f"penalty_cv_{suffix}"
        torch.save(chosen_state, model_dir / f"{model_suffix}.pt")

        with open(results_dir / f"best_lambda_{suffix}.txt", "w") as f:
            f.write(str(chosen_lam))
        print(f"  Saved model  → {model_dir / f'{model_suffix}.pt'}")
        print(f"  Saved lambda → {results_dir / f'best_lambda_{suffix}.txt'}")
    else:
        print("  WARNING: No lambda satisfied the constraints.")
        print("  The largest lambda tested also failed. Try adding larger values.")
        # Write a fallback so the bash script doesn't break
        with open(results_dir / f"best_lambda_{suffix}.txt", "w") as f:
            f.write(str(lambdas[0]))  # largest tested
        print(f"  Fallback lambda: {lambdas[0]}")

    print(f"\n  Full results → {csv_path}")


def get_args():
    ap = argparse.ArgumentParser(description="Cross-validation for penalty lambda (largest→smallest search).")
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    ap.add_argument("--freeze_backbone", action="store_true", default=False)
    ap.add_argument("--lora", action="store_true", default=False)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--cv_max_epochs", type=int, default=None,
                    help="Max epochs per lambda during CV (default: same as fairface_xp_params).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits_dir", type=str, default="fairface_splits")
    ap.add_argument("--results_dir", type=str, default="results")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_cross_val(args)