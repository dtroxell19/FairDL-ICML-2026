#!/bin/bash
########################################################################################################################
# Run all CelebA experiments: 3 backbones × 4 model types
#
# For each backbone:
#   1. Cross-validate to find best penalty lambda
#   2. Train fair, baseline, penalty (with best lambda), strict_penalty
#   3. Aggregate all results into a single CSV
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
########################################################################################################################

set -e  # exit on error

RESULTS_DIR="training_results"
SPLITS_DIR="celeba_splits"
SEED=1

mkdir -p "$RESULTS_DIR"

# ── Ensure data is prepared ──────────────────────────────────────────────────────

if [ ! -f "$SPLITS_DIR/train_dataset.pt" ]; then
    echo "============================================"
    echo "  Preparing data..."
    echo "============================================"
    python load_data.py --save_dir "$SPLITS_DIR"
fi

# ── Helper function to run one backbone config ───────────────────────────────────

run_backbone() {
    local BACKBONE=$1
    local EXTRA_ARGS=$2  # additional flags like --lora, --no_pretrained, --batch_size
    local LABEL=$3       # human-readable label for logging

    echo ""
    echo "============================================================================"
    echo "  BACKBONE: $LABEL"
    echo "============================================================================"

    # ── Step 1: Cross-validate penalty lambda ────────────────────────────────
    echo ""
    echo "  [CV] Finding best penalty lambda..."
    python cross_val.py \
        --backbone "$BACKBONE" \
        --seed "$SEED" \
        --splits_dir "$SPLITS_DIR" \
        --results_dir "$RESULTS_DIR" \
        --cv_max_epochs 5 \
        $EXTRA_ARGS

    # Read the best lambda
    local SUFFIX="$BACKBONE"
    if echo "$EXTRA_ARGS" | grep -q "\-\-lora"; then
        local LORA_RANK=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--lora_rank )\d+' || echo "8")
        SUFFIX="${BACKBONE}_lora_r${LORA_RANK}"
    fi
    local LAMBDA_FILE="$RESULTS_DIR/best_lambda_${SUFFIX}.txt"
    if [ -f "$LAMBDA_FILE" ]; then
        BEST_LAMBDA=$(cat "$LAMBDA_FILE")
        echo "  [CV] Best lambda: $BEST_LAMBDA"
    else
        echo "  [CV] WARNING: Lambda file not found, using default 10.0"
        BEST_LAMBDA=10.0
    fi

    # ── Step 2: Train all 4 model types ──────────────────────────────────────
    for MODEL_TYPE in fair baseline penalty strict_penalty; do
        echo ""
        echo "  ── $MODEL_TYPE ($LABEL) ──────────────────────────"

        CMD="python run_celeba_experiments.py \
            --model_type $MODEL_TYPE \
            --backbone $BACKBONE \
            --seed $SEED \
            --splits_dir $SPLITS_DIR \
            --results_dir $RESULTS_DIR \
            $EXTRA_ARGS"

        # Add penalty lambda for penalty model
        if [ "$MODEL_TYPE" = "penalty" ]; then
            CMD="$CMD --penalty_lambda $BEST_LAMBDA"
        fi

        eval $CMD
    done
}


# ── Run all backbone configs ─────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        CelebA Experiments — Full Ablation Suite             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 1. ResNet-18 (default batch sizes)
run_backbone "resnet18" "" "ResNet-18 (pretrained, full fine-tune)"

# 2. Simple CNN (from scratch, no pretrained)
run_backbone "simple_cnn" "--no_pretrained" "SimpleCNN (from scratch)"

# 3. ViT-B/16 with LoRA (smaller batch for memory)
run_backbone "vit_b_16" "--lora --batch_size 64" "ViT-B/16 (LoRA, rank=8)"


# ── Aggregate results ────────────────────────────────────────────────────────────

echo ""
echo "============================================================================"
echo "  Aggregating all results..."
echo "============================================================================"
python aggregate_results.py --results_dir "$RESULTS_DIR" --output "$RESULTS_DIR/all_results.csv"

echo ""
echo "============================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Combined results: $RESULTS_DIR/all_results.csv"
echo "============================================================================"
echo ""

# Print summary table
python -c "
import csv
from pathlib import Path

csv_path = Path('$RESULTS_DIR/all_results.csv')
if not csv_path.exists():
    exit()

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

print(f\"{'Model Type':<18} {'Backbone':<14} {'Loss':>8} {'Acc':>8} {'MaxGap':>10} {'Satisfied':>10} {'Time(s)':>10}\")
print('─' * 80)

for r in rows:
    mt = r.get('model_type', 'baseline')
    bb = r.get('backbone', '?')
    loss = r.get('loss', r.get('loss_proj', '?'))
    acc = r.get('accuracy', r.get('accuracy_proj', '?'))
    gap = r.get('max_pairwise_gap', r.get('gap_proj', '?'))
    sat = r.get('constraint_satisfied', '?')
    time_s = r.get('total_training_time_sec', '?')

    try: loss = f'{float(loss):.4f}'
    except: pass
    try: acc = f'{float(acc):.4f}'
    except: pass
    try: gap = f'{float(gap):.6f}'
    except: pass
    try: time_s = f'{float(time_s):.0f}'
    except: pass

    print(f'{mt:<18} {bb:<14} {loss:>8} {acc:>8} {gap:>10} {sat:>10} {time_s:>10}')
"
