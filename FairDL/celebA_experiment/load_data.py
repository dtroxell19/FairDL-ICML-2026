########################################################################################################################
# This script loads the CelebA dataset and prepares it for training in the CelebA experiments.
#
# To avoid OOM during preprocessing, images are processed in chunks:
#   1. Download HF dataset, extract lightweight metadata (attributes)
#   2. Split indices into train/val/test (using CelebA's official splits)
#   3. For each split, process images in chunks of CHUNK_SIZE, saving each to disk
#   4. Concatenate chunks into final {split}_dataset.pt files
#   5. Clean up temporary chunk files
#
# Peak RAM usage ≈ CHUNK_SIZE × 3 × 224 × 224 × 4 bytes
#   (e.g., 5000 × 3 × 224 × 224 × 4 ≈ 3.0 GB)
#
# Output TensorDatasets have two tensors:
#   images:  (N, 3, 224, 224) float32  — ImageNet-normalized
#   meta:    (N, 3) int64              — [target_label, male, young]
#
# Intersectional groups: Male × Young → 4 groups (C(4,2) = 6 pairwise constraints)
########################################################################################################################

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

# ── Constants ────────────────────────────────────────────────────────────────────

HF_DATASET = "flwrlabs/celeba"
HF_SUBSET = "img_align+identity+attr"

IMAGE_SIZE = 224
CHUNK_SIZE = 5000

# Protected attribute structure
NUM_GENDERS = 2   # 0 = Female, 1 = Male
NUM_AGE = 2       # 0 = Not Young, 1 = Young
NUM_GROUPS = NUM_GENDERS * NUM_AGE  # 4 intersectional groups

# CelebA attribute names (the ones we use)
TARGET_ATTR = "Smiling"
GENDER_ATTR = "Male"
AGE_ATTR = "Young"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Label helpers ────────────────────────────────────────────────────────────────

GENDER_NAMES = {0: "Female", 1: "Male"}
AGE_NAMES = {0: "Not Young", 1: "Young"}


def group_id_to_label(group_id):
    """Convert intersectional group ID to human-readable string."""
    gender = group_id // NUM_AGE
    age = group_id % NUM_AGE
    return f"{GENDER_NAMES[gender]}, {AGE_NAMES[age]}"


# ── Chunked image processing ────────────────────────────────────────────────────

def process_image_chunk(hf_split, indices, image_size):
    """
    Process a chunk of images: resize, convert to tensor, normalize.

    @param hf_split: HuggingFace dataset split
    @param indices: list/array of integer indices to process
    @param image_size: target spatial resolution

    @returns Tensor (len(indices), 3, image_size, image_size)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensors = []
    for idx in indices:
        img = hf_split[int(idx)]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensors.append(transform(img))

    return torch.stack(tensors)


# ── Main loader ──────────────────────────────────────────────────────────────────

def load_celeba_splits(val_fraction=0.1, data_seed=42, image_size=IMAGE_SIZE,
                       chunk_size=CHUNK_SIZE, save_dir="celeba_splits",
                       target_attr=TARGET_ATTR):
    """
    Load CelebA, preprocess, and save train/val/test splits as TensorDatasets.

    CelebA has official train/val/test splits on HuggingFace. We further split
    the official train set into train/val using val_fraction.

    @param val_fraction: fraction of official train split to hold out for validation
    @param data_seed: random seed for train/val splitting
    @param image_size: spatial resolution for images
    @param chunk_size: images processed per chunk (controls peak RAM)
    @param save_dir: output directory for .pt files
    @param target_attr: which CelebA attribute to predict (default: "Attractive")

    @returns (train_dataset, val_dataset, test_dataset)
    """
    from datasets import load_dataset

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Check for cached splits
    cached = [save_path / f"{s}_dataset.pt" for s in ("train", "val", "test")]
    if all(p.exists() for p in cached):
        print(f"Found cached splits in {save_dir}, loading from disk...")
        train_dataset = torch.load(cached[0], weights_only=False)
        val_dataset = torch.load(cached[1], weights_only=False)
        test_dataset = torch.load(cached[2], weights_only=False)
        _print_summary(train_dataset, val_dataset, test_dataset, target_attr)
        return train_dataset, val_dataset, test_dataset

    print(f"Loading CelebA from HuggingFace ({HF_DATASET})...")
    ds = load_dataset(HF_DATASET, HF_SUBSET)

    # ── Extract metadata from each official split ────────────────────────────
    # CelebA attributes are -1/1 in some HF versions; we normalize to 0/1
    def extract_meta(split):
        """Extract target, gender, age arrays from a HF split."""
        n = len(split)
        targets = np.zeros(n, dtype=np.int64)
        genders = np.zeros(n, dtype=np.int64)
        ages = np.zeros(n, dtype=np.int64)

        for i in range(n):
            row = split[i]
            attrs = row.get("attributes", row)

            # Handle both dict-style and flat-style attribute access
            def get_attr(name):
                if isinstance(attrs, dict):
                    val = attrs[name]
                else:
                    val = row[name]
                # Normalize -1/1 to 0/1 if needed
                return max(0, int(val))

            targets[i] = get_attr(target_attr)
            genders[i] = get_attr(GENDER_ATTR)
            ages[i] = get_attr(AGE_ATTR)

        return targets, genders, ages

    print("  Extracting metadata from train split...")
    train_targets, train_genders, train_ages = extract_meta(ds["train"])
    print("  Extracting metadata from valid split...")
    val_targets, val_genders, val_ages = extract_meta(ds["valid"])
    print("  Extracting metadata from test split...")
    test_targets, test_genders, test_ages = extract_meta(ds["test"])

    # ── Optionally further split the official train into train/val ────────────
    # CelebA already has an official val split (~19.9k), but if val_fraction > 0
    # we can also carve out a portion of train for additional validation.
    # We use the official val split as-is and don't re-split unless requested.
    n_train_official = len(train_targets)
    print(f"  Official splits — train: {n_train_official}, val: {len(val_targets)}, "
          f"test: {len(test_targets)}")

    # ── Process each split in chunks ─────────────────────────────────────────
    splits_config = [
        ("train", ds["train"], train_targets, train_genders, train_ages),
        ("val", ds["valid"], val_targets, val_genders, val_ages),
        ("test", ds["test"], test_targets, test_genders, test_ages),
    ]

    datasets = {}
    for split_name, hf_split, targets, genders, ages in splits_config:
        n = len(targets)
        indices = np.arange(n)
        n_chunks = (n + chunk_size - 1) // chunk_size

        print(f"\n  Processing {split_name} ({n} images, {n_chunks} chunks)...")
        chunk_paths = []

        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n)
            chunk_idx = indices[start:end]

            print(f"    Chunk {c+1}/{n_chunks} (images {start}–{end-1})...")
            images_chunk = process_image_chunk(hf_split, chunk_idx, image_size)

            chunk_path = save_path / f"{split_name}_chunk_{c}.pt"
            torch.save(images_chunk, chunk_path)
            chunk_paths.append(chunk_path)

            del images_chunk
            gc.collect()

        # Concatenate chunks
        print(f"    Concatenating {len(chunk_paths)} chunks...")
        all_images = torch.cat([torch.load(p, weights_only=False) for p in chunk_paths])

        meta = torch.stack([
            torch.from_numpy(targets),
            torch.from_numpy(genders),
            torch.from_numpy(ages),
        ], dim=1)  # (N, 3): [target, male, young]

        dataset = TensorDataset(all_images, meta)
        datasets[split_name] = dataset

        # Save and clean up
        torch.save(dataset, save_path / f"{split_name}_dataset.pt")
        for p in chunk_paths:
            os.remove(p)

        del all_images, meta
        gc.collect()

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    _print_summary(train_dataset, val_dataset, test_dataset, target_attr)
    return train_dataset, val_dataset, test_dataset


def _print_summary(train_dataset, val_dataset, test_dataset, target_attr):
    """Print dataset statistics."""
    print(f"\n{'='*70}")
    print(f"CelebA Dataset Summary")
    print(f"  Target:      {target_attr} (binary)")
    print(f"  Protected:   {NUM_GROUPS} intersectional groups ({NUM_GENDERS} genders × {NUM_AGE} age)")
    print(f"  Constraints: C({NUM_GROUPS},2) = {NUM_GROUPS * (NUM_GROUPS - 1) // 2} "
          f"pairwise demographic parity")
    print(f"{'='*70}")
    print(f"  Train: {len(train_dataset):>6}  |  Val: {len(val_dataset):>6}  "
          f"|  Test: {len(test_dataset):>6}")

    for name, dset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        meta = dset.tensors[1]
        groups = meta[:, 1] * NUM_AGE + meta[:, 2]   # male * 2 + young
        counts = [(g, int((groups == g).sum())) for g in range(NUM_GROUPS)]
        target_rate = meta[:, 0].float().mean().item()

        print(f"\n  {name} — target rate: {target_rate:.3f}")
        print(f"  {'Group':<30s}  {'Count':>6}  {'Pct':>6}")
        for g, c in counts:
            print(f"    {group_id_to_label(g):<30s}: {c:>5} ({100*c/len(meta):.1f}%)")

    print(f"\n{'='*70}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def get_args():
    ap = argparse.ArgumentParser(description="CelebA data loader (chunked processing).")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                    help=f"Images per processing chunk (default: {CHUNK_SIZE}). Lower = less RAM.")
    ap.add_argument("--save_dir", type=str, default="celeba_splits")
    ap.add_argument("--target_attr", type=str, default=TARGET_ATTR,
                    help=f"CelebA attribute to predict (default: {TARGET_ATTR})")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    load_celeba_splits(
        val_fraction=args.val_fraction, data_seed=args.data_seed,
        image_size=args.image_size, chunk_size=args.chunk_size,
        save_dir=args.save_dir, target_attr=args.target_attr,
    )
