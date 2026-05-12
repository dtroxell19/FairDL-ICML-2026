########################################################################################################################
# This script loads the FairFace dataset and prepares it for training in the FairFace experiments.
#
# To avoid OOM during preprocessing, images are processed in chunks:
#   1. Download HF dataset, extract lightweight metadata (labels, gender, race)
#   2. Split indices into train/val/test
#   3. For each split, process images in chunks of CHUNK_SIZE, saving each to disk
#   4. Concatenate chunks into final {split}_dataset.pt files
#   5. Clean up temporary chunk files
#
# Peak RAM usage ≈ CHUNK_SIZE × 3 × 224 × 224 × 4 bytes (e.g. 5000 images ≈ 3GB)
########################################################################################################################

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import datasets as hf_datasets
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────────

DATASET_HANDLE = "HuggingFaceM4/FairFace"
SUBSET = "0.25"
CHUNK_SIZE = 5000  # images per chunk — tune based on available RAM

AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
AGE_THRESHOLD = 4

GENDER_CLASSES = ["Male", "Female"]
RACE_CLASSES = [
    "East Asian", "Indian", "Black", "White",
    "Middle Eastern", "Latino_Hispanic", "Southeast Asian",
]
NUM_GENDERS = len(GENDER_CLASSES)
NUM_RACES = len(RACE_CLASSES)
NUM_GROUPS = NUM_GENDERS * NUM_RACES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# ── Helpers ──────────────────────────────────────────────────────────────────────

def binarize_age(age_class_index, threshold=AGE_THRESHOLD):
    return int(age_class_index >= threshold)

def intersectional_group_id(gender, race):
    return gender * NUM_RACES + race

def group_id_to_label(group_id):
    gender = group_id // NUM_RACES
    race = group_id % NUM_RACES
    return f"{GENDER_CLASSES[gender]}-{RACE_CLASSES[race]}"


# ── Image transforms ─────────────────────────────────────────────────────────────

def get_train_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── HF loading ───────────────────────────────────────────────────────────────────

def load_hf_dataset(handle=DATASET_HANDLE, subset=SUBSET):
    if not _HF_AVAILABLE:
        raise RuntimeError("The 'datasets' package is required. Install with:  pip install datasets")
    print(f"[info] Loading FairFace from HuggingFace ({handle}, subset={subset})...")
    ds = hf_datasets.load_dataset(handle, subset)
    print(f"[info] Loaded splits: {list(ds.keys())}")
    return ds


# ── Chunked processing ───────────────────────────────────────────────────────────

def process_and_save_chunks(hf_split, indices, transform, chunk_dir, image_size=IMAGE_SIZE,
                            chunk_size=CHUNK_SIZE):
    """
    Process a subset of a HuggingFace split in chunks, saving each chunk to disk.

    @param hf_split: HuggingFace Dataset (the full split, e.g. ds["train"])
    @param indices (ndarray): which rows to process
    @param transform: torchvision transform
    @param chunk_dir (Path): directory to save temporary chunk .pt files
    @param image_size (int): target resolution
    @param chunk_size (int): images per chunk

    @returns (num_chunks, meta_tensor) where meta_tensor is [N, 3] with [age, gender, race]
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = len(indices)
    num_chunks = (n + chunk_size - 1) // chunk_size

    # Meta is tiny — collect it all at once
    meta = torch.empty(n, 3, dtype=torch.long)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        batch_indices = indices[start:end]
        batch_n = len(batch_indices)

        # Pre-allocate this chunk's image tensor
        images = torch.empty(batch_n, 3, image_size, image_size, dtype=torch.float32)

        desc = f"    Chunk {chunk_idx+1}/{num_chunks}"
        for i, hf_idx in enumerate(tqdm(batch_indices, desc=desc, leave=False)):
            sample = hf_split[int(hf_idx)]
            images[i] = transform(sample["image"].convert("RGB"))
            meta[start + i, 0] = binarize_age(sample["age"])
            meta[start + i, 1] = sample["gender"]
            meta[start + i, 2] = sample["race"]

        # Save chunk to disk and free memory
        torch.save(images, chunk_dir / f"chunk_{chunk_idx:04d}.pt")
        del images

    return num_chunks, meta


def concat_chunks(chunk_dir, num_chunks):
    """
    Load and concatenate all chunk .pt files into a single tensor.

    @param chunk_dir (Path): directory containing chunk_XXXX.pt files
    @param num_chunks (int): number of chunks

    @returns (N, 3, H, W) tensor
    """
    chunks = []
    for chunk_idx in range(num_chunks):
        chunks.append(torch.load(chunk_dir / f"chunk_{chunk_idx:04d}.pt", weights_only=False))
    return torch.cat(chunks, dim=0)


# ── Main entry point ─────────────────────────────────────────────────────────────

def load_fairface_splits(
    handle=DATASET_HANDLE, subset=SUBSET, val_fraction=0.1, data_seed=42,
    image_size=IMAGE_SIZE, chunk_size=CHUNK_SIZE, save_dir="fairface_splits",
):
    """
    Load and preprocess FairFace in chunks, saving train/val/test TensorDatasets to disk.

    Each TensorDataset contains (image_tensor, meta_tensor) where:
        meta_tensor[:, 0] = binary age label
        meta_tensor[:, 1] = gender (0=Male, 1=Female)
        meta_tensor[:, 2] = race (0–6)
    """
    ds = load_hf_dataset(handle, subset)
    hf_train = ds["train"]
    hf_test = ds["validation"]
    out_dir = Path(save_dir)
    tmp_dir = out_dir / "_tmp_chunks"

    # ── Step 1: Extract labels for stratified splitting (no images, ~1 sec) ──────
    print("[info] Scanning labels for train/val split...")
    train_labels = np.array([
        binarize_age(hf_train[i]["age"])
        for i in tqdm(range(len(hf_train)), desc="  Labels", leave=True)
    ])

    all_indices = np.arange(len(hf_train))
    idx_train, idx_val = train_test_split(
        all_indices, test_size=val_fraction, random_state=data_seed, stratify=train_labels,
    )
    idx_test = np.arange(len(hf_test))

    print(f"[info] Split sizes — Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # ── Step 2: Process each split in chunks ─────────────────────────────────────
    splits = [
        ("train", hf_train, idx_train, get_train_transform(image_size)),
        ("val",   hf_train, idx_val,   get_eval_transform(image_size)),
        ("test",  hf_test,  idx_test,  get_eval_transform(image_size)),
    ]

    datasets = {}
    for name, hf_split, indices, tfm in splits:
        print(f"\n[info] Processing {name} split ({len(indices)} images, chunk_size={chunk_size})...")
        chunk_dir = tmp_dir / name
        num_chunks, meta = process_and_save_chunks(
            hf_split, indices, tfm, chunk_dir, image_size, chunk_size,
        )

        print(f"  Concatenating {num_chunks} chunks...")
        images = concat_chunks(chunk_dir, num_chunks)

        datasets[name] = TensorDataset(images, meta)
        del images  # free before processing next split

    # ── Step 3: Save final datasets and clean up ─────────────────────────────────
    out_dir.mkdir(exist_ok=True)
    for name in ("train", "val", "test"):
        torch.save(datasets[name], out_dir / f"{name}_dataset.pt")
    print(f"\n[info] Saved datasets to {out_dir.resolve()}")

    # Clean up temp chunks
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[info] Cleaned up temporary chunk files")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FairFace Dataset Summary")
    print(f"  Target:     Binary age (under 30 vs 30+)")
    print(f"  Protected:  14 intersectional groups (2 genders × 7 races)")
    print(f"  Constraints: C(14,2) = 91 pairwise demographic parity")
    print(f"{'='*70}")
    print(f"  Train: {len(train_dataset):>6}  |  Val: {len(val_dataset):>6}  |  Test: {len(test_dataset):>6}")

    for name, dset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        meta = dset.tensors[1]
        groups = meta[:, 1] * NUM_RACES + meta[:, 2]
        counts = [(g, int((groups == g).sum())) for g in range(NUM_GROUPS)]
        print(f"\n  {name} intersectional group sizes:")
        for g, c in counts:
            print(f"    {group_id_to_label(g):<30s}: {c:>5} ({100*c/len(groups):.1f}%)")

    print(f"\n{'='*70}\n")
    return train_dataset, val_dataset, test_dataset


# ── CLI ──────────────────────────────────────────────────────────────────────────

def get_args():
    ap = argparse.ArgumentParser(description="FairFace data loader (chunked processing).")
    ap.add_argument("--subset", type=str, default=SUBSET, choices=["0.25", "1.25"])
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                    help=f"Images per processing chunk (default: {CHUNK_SIZE}). Lower = less RAM.")
    ap.add_argument("--save_dir", type=str, default="fairface_splits")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    load_fairface_splits(
        subset=args.subset, val_fraction=args.val_fraction,
        data_seed=args.data_seed, image_size=args.image_size,
        chunk_size=args.chunk_size, save_dir=args.save_dir,
    )