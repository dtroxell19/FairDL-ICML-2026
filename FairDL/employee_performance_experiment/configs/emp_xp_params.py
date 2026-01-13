########################################################################################################################
#This script defines hyperparameters and the evaluation criteria/function used for the employee performance experiments
########################################################################################################################

import torch
import numpy as np


def get_slack():
    """
    Max bias per group per protected attribute
    """
    return 0.0001


def get_num_batches_train():
    """
    Defines number of minibatches when training. Perform full batch training in this experiment
    """
    return 1


def evaluate_fairness_audit(
    raw_preds,
    fair_preds,
    targets,
    x,
    protected_cols,
    slack_threshold,
    flagging_percentile=90,
    top_n_flagged=50,
):
    """
    Evaluate fairness methods with marginal fairness constraints.

    For each protected column and each group (0/1), checks that bias is under threshold

    @param raw_preds: baseline model predictions (no fairness)
    @param fair_preds: fairness-adjusted predictions
    @param targets: actual wages
    @param x: input features (to extract protected attributes)
    @param protected_cols: list of protected attribute column indices
    @param slack_threshold: maximum allowed residual for each group
    @param flagging_percentile: percentile for flagging (e.g., 90 = top 10%)
    @param top_n_flagged: number of top flagged employees to analyze (default: 50)

    @return: dictionary of evaluation metrics
    """
    #Convert tensors to numpy
    if isinstance(raw_preds, torch.Tensor):
        raw_preds = raw_preds.detach().cpu().numpy()
    if isinstance(fair_preds, torch.Tensor):
        fair_preds = fair_preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    raw_preds = raw_preds.flatten()
    fair_preds = fair_preds.flatten()
    targets = targets.flatten()

    #protected columns are binary
    x_binary = x_np.copy()
    for col in protected_cols:
        x_binary[:, col] = (x_binary[:, col] > 0.5).astype(float)

    print(
        f"\nEvaluating marginal fairness across {len(protected_cols)} protected columns..."
    )
    print(f"Total constraints: {len(protected_cols) * 2} (2 per column)")
    print("=" * 70)

    #1. MSE Comparison
    raw_mse = np.mean((raw_preds - targets) ** 2)
    fair_mse = np.mean((fair_preds - targets) ** 2)
    mse_change = fair_mse - raw_mse
    mse_pct_change = (mse_change / raw_mse) * 100

    #2. marginal Constraint Violations
    raw_violations = 0
    fair_violations = 0

    print("\nCONSTRAINT SATISFACTION (Marginal Fairness)")
    print("-" * 70)

    all_raw_residuals = []
    all_fair_residuals = []

    for col_idx, col in enumerate(protected_cols):
        #Get indices for each marginal group
        mask_0 = x_binary[:, col] == 0
        mask_1 = x_binary[:, col] == 1

        n_0 = mask_0.sum()
        n_1 = mask_1.sum()

        print(f"\n Column {col} ")

        if n_0 > 0:
            #Group 0 constraint
            residual_0_raw = abs(raw_preds[mask_0].mean() - targets[mask_0].mean())
            residual_0_fair = abs(fair_preds[mask_0].mean() - targets[mask_0].mean())

            all_raw_residuals.append(residual_0_raw)
            all_fair_residuals.append(residual_0_fair)

            violated_raw = residual_0_raw > slack_threshold + 1e-5
            violated_fair = residual_0_fair > slack_threshold + 1e-5

            if violated_raw:
                raw_violations += 1
            if violated_fair:
                fair_violations += 1

            print(f"  Group 0 (n={n_0}):")
            print(
                f"    Raw:  |residual|={residual_0_raw:.6f} {'[VIOLATION]' if violated_raw else '[OK]'}"
            )
            print(
                f"    Fair: |residual|={residual_0_fair:.6f} {'[VIOLATION]' if violated_fair else '[OK]'}"
            )

        if n_1 > 0:
            #Group 1 constraint
            residual_1_raw = abs(raw_preds[mask_1].mean() - targets[mask_1].mean())
            residual_1_fair = abs(fair_preds[mask_1].mean() - targets[mask_1].mean())

            all_raw_residuals.append(residual_1_raw)
            all_fair_residuals.append(residual_1_fair)

            violated_raw = residual_1_raw > slack_threshold + 1e-5
            violated_fair = residual_1_fair > slack_threshold + 1e-5

            if violated_raw:
                raw_violations += 1
            if violated_fair:
                fair_violations += 1

            print(f"  Group 1 (n={n_1}):")
            print(
                f"    Raw:  |residual|={residual_1_raw:.6f} {'[VIOLATION]' if violated_raw else '[OK]'}"
            )
            print(
                f"    Fair: |residual|={residual_1_fair:.6f} {'[VIOLATION]' if violated_fair else '[OK]'}"
            )

    total_constraints = len(protected_cols) * 2
    max_raw_residual = max(all_raw_residuals) if all_raw_residuals else 0
    max_fair_residual = max(all_fair_residuals) if all_fair_residuals else 0

    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Raw violations:  {raw_violations}/{total_constraints} constraints")
    print(f"  Fair violations: {fair_violations}/{total_constraints} constraints")
    print(f"  Max raw residual:  {max_raw_residual:.6f}")
    print(f"  Max fair residual: {max_fair_residual:.6f}")

    #3. Individual-Level Prediction Changes
    pred_change = fair_preds - raw_preds

    #Calculate percentage changes (relative to raw predictions)
    epsilon = 1e-6
    pct_change = (pred_change / (np.abs(raw_preds) + epsilon)) * 100

    #Absolute change metrics
    total_change = np.abs(pred_change).sum()
    mean_abs_change = np.abs(pred_change).mean()
    max_abs_change = np.abs(pred_change).max()
    #Percentage change metrics
    mean_abs_pct_change = np.abs(pct_change).mean()
    max_increase_pct = pct_change.max()
    max_decrease_pct = pct_change.min()

    #Count individuals by percentage change thresholds
    n_changed_1pct = (np.abs(pct_change) > 1.0).sum()
    n_changed_5pct = (np.abs(pct_change) > 5.0).sum()
    n_changed_10pct = (np.abs(pct_change) > 10.0).sum()
    n_changed_20pct = (np.abs(pct_change) > 20.0).sum()

    #Directional counts
    n_increased = (pred_change > 0.01).sum()
    n_decreased = (pred_change < -0.01).sum()
    n_unchanged = (np.abs(pred_change) <= 0.01).sum()

    n_total = len(pred_change)

    #Calculate percentages for directional changes
    pct_increased = (n_increased / n_total) * 100
    pct_decreased = (n_decreased / n_total) * 100
    pct_unchanged = (n_unchanged / n_total) * 100

    print("\nINDIVIDUAL-LEVEL PREDICTION CHANGES")
    print("-" * 70)
    print(f"Absolute changes:")
    print(f"  Total absolute change:    {total_change:.2f}")
    print(f"  Mean absolute change:     {mean_abs_change:.6f}")
    print(f"  Max absolute change:      {max_abs_change:.6f}")

    print(f"\nPercentage changes (relative to raw prediction):")
    print(f"  Mean absolute % change:   {mean_abs_pct_change:.2f}%")
    print(f"  Largest % increase:       {max_increase_pct:.2f}%")
    print(f"  Largest % decrease:       {max_decrease_pct:.2f}%")

    print(f"\nIndividuals affected by change threshold:")
    print(f"  Changed by >1%:   {n_changed_1pct:,} ({n_changed_1pct/n_total*100:.1f}%)")
    print(f"  Changed by >5%:   {n_changed_5pct:,} ({n_changed_5pct/n_total*100:.1f}%)")
    print(
        f"  Changed by >10%:  {n_changed_10pct:,} ({n_changed_10pct/n_total*100:.1f}%)"
    )
    print(
        f"  Changed by >20%:  {n_changed_20pct:,} ({n_changed_20pct/n_total*100:.1f}%)"
    )

    print(f"\nDirectional changes:")
    print(f"  Increased (>0.01): {n_increased:,} ({pct_increased:.1f}%)")
    print(f"  Decreased (<-0.01): {n_decreased:,} ({pct_decreased:.1f}%)")
    print(f"  Unchanged (±0.01): {n_unchanged:,} ({pct_unchanged:.1f}%)")

    #4. Raise Flagging Analysis
    #"Flagging" = identifying employees for potential raises based on prediction residuals
    #Positive residual (pred > actual) suggests model thinks employee is underpaid
    print(f"\nRAISE FLAGGING ANALYSIS")
    print("-" * 70)
    print(
        f"Employees with top {100-flagging_percentile:.0f}% of residuals are flagged as potentially underpaid"
    )
    print(f"(Residual = predicted wage - actual wage; higher = more underpaid)")

    raw_residual_all = raw_preds - targets
    fair_residual_all = fair_preds - targets

    raw_threshold = np.percentile(raw_residual_all, flagging_percentile)
    fair_threshold = np.percentile(fair_residual_all, flagging_percentile)

    raw_flagged = raw_residual_all >= raw_threshold
    fair_flagged = fair_residual_all >= fair_threshold

    print(f"\nFlagging thresholds:")
    print(f"  Raw model:  residual ≥ {raw_threshold:.4f}")
    print(f"  Fair model: residual ≥ {fair_threshold:.4f}")

    #Flagging rates by marginal group
    print(f"\nFlagging rates by group (what % of each group gets flagged):")

    for col in protected_cols:
        mask_0 = x_binary[:, col] == 0
        mask_1 = x_binary[:, col] == 1

        if mask_0.sum() > 0 and mask_1.sum() > 0:
            raw_rate_0 = raw_flagged[mask_0].mean()
            raw_rate_1 = raw_flagged[mask_1].mean()
            fair_rate_0 = fair_flagged[mask_0].mean()
            fair_rate_1 = fair_flagged[mask_1].mean()

            raw_gap = abs(raw_rate_0 - raw_rate_1)
            fair_gap = abs(fair_rate_0 - fair_rate_1)

            print(f"\n  Column {col}:")
            print(
                f"    Raw:  group_0={raw_rate_0:.3f}, group_1={raw_rate_1:.3f}, |gap|={raw_gap:.3f}"
            )
            print(
                f"    Fair: group_0={fair_rate_0:.3f}, group_1={fair_rate_1:.3f}, |gap|={fair_gap:.3f}"
            )

            if fair_gap < raw_gap:
                improvement = (
                    ((raw_gap - fair_gap) / raw_gap * 100) if raw_gap > 0 else 0
                )
                print(f"    → Gap reduced by {improvement:.1f}%")

    #Overall flagging parity (max gap across all columns)
    raw_flag_gaps = []
    fair_flag_gaps = []
    for col in protected_cols:
        mask_0 = x_binary[:, col] == 0
        mask_1 = x_binary[:, col] == 1
        if mask_0.sum() > 0 and mask_1.sum() > 0:
            raw_flag_gaps.append(
                abs(raw_flagged[mask_0].mean() - raw_flagged[mask_1].mean())
            )
            fair_flag_gaps.append(
                abs(fair_flagged[mask_0].mean() - fair_flagged[mask_1].mean())
            )

    raw_flag_parity = max(raw_flag_gaps) if raw_flag_gaps else 0
    fair_flag_parity = max(fair_flag_gaps) if fair_flag_gaps else 0

    print(f"\nMax flagging disparity across columns:")
    print(f"  Raw:  {raw_flag_parity:.4f}")
    print(f"  Fair: {fair_flag_parity:.4f}")

    #5. Top N Flagged Employees Group Composition
    print(f"\n{'='*70}")
    print(f"TOP {top_n_flagged} FLAGGED EMPLOYEES - GROUP COMPOSITION")
    print("-" * 70)
    print(
        f"Analyzing group membership rates for the {top_n_flagged} employees with largest residuals"
    )

    #Get indices of top N flagged employees
    raw_top_n_idx = np.argsort(raw_residual_all)[-top_n_flagged:]
    fair_top_n_idx = np.argsort(fair_residual_all)[-top_n_flagged:]

    print(f"\nGroup membership rates (% of top {top_n_flagged} from each group):")

    top_n_composition = {}

    for col in protected_cols:
        mask_0 = x_binary[:, col] == 0
        mask_1 = x_binary[:, col] == 1

        #Overall population rates for comparison
        pop_rate_0 = mask_0.mean()
        pop_rate_1 = mask_1.mean()

        #Rates among top N flagged
        raw_top_rate_0 = mask_0[raw_top_n_idx].mean()
        raw_top_rate_1 = mask_1[raw_top_n_idx].mean()
        fair_top_rate_0 = mask_0[fair_top_n_idx].mean()
        fair_top_rate_1 = mask_1[fair_top_n_idx].mean()

        #Calculate counts
        raw_top_count_0 = mask_0[raw_top_n_idx].sum()
        raw_top_count_1 = mask_1[raw_top_n_idx].sum()
        fair_top_count_0 = mask_0[fair_top_n_idx].sum()
        fair_top_count_1 = mask_1[fair_top_n_idx].sum()

        print(f"\n  Column {col}:")
        print(
            f"    Population baseline: group_0={pop_rate_0:.1%}, group_1={pop_rate_1:.1%}"
        )
        print(
            f"    Raw top-{top_n_flagged}:  group_0={raw_top_rate_0:.1%} (n={raw_top_count_0}), "
            f"group_1={raw_top_rate_1:.1%} (n={raw_top_count_1})"
        )
        print(
            f"    Fair top-{top_n_flagged}: group_0={fair_top_rate_0:.1%} (n={fair_top_count_0}), "
            f"group_1={fair_top_rate_1:.1%} (n={fair_top_count_1})"
        )

        #Show representation gaps (difference from population baseline)
        raw_gap_0 = raw_top_rate_0 - pop_rate_0
        raw_gap_1 = raw_top_rate_1 - pop_rate_1
        fair_gap_0 = fair_top_rate_0 - pop_rate_0
        fair_gap_1 = fair_top_rate_1 - pop_rate_1

        print(f"    Representation gap vs population:")
        print(f"      Raw:  group_0={raw_gap_0:+.1%}, group_1={raw_gap_1:+.1%}")
        print(f"      Fair: group_0={fair_gap_0:+.1%}, group_1={fair_gap_1:+.1%}")

        #Store in dictionary for return
        top_n_composition[f"col_{col}"] = {
            "pop_rate_0": pop_rate_0,
            "pop_rate_1": pop_rate_1,
            "raw_top_rate_0": raw_top_rate_0,
            "raw_top_rate_1": raw_top_rate_1,
            "raw_top_count_0": int(raw_top_count_0),
            "raw_top_count_1": int(raw_top_count_1),
            "fair_top_rate_0": fair_top_rate_0,
            "fair_top_rate_1": fair_top_rate_1,
            "fair_top_count_0": int(fair_top_count_0),
            "fair_top_count_1": int(fair_top_count_1),
        }

    print("=" * 70)

    #Return summary
    return {
        #MSE
        "raw_mse": raw_mse,
        "fair_mse": fair_mse,
        "mse_change": mse_change,
        "mse_pct_change": mse_pct_change,
        #Constraints (2 per column)
        "raw_violations": raw_violations,
        "fair_violations": fair_violations,
        "total_constraints": total_constraints,
        "max_raw_residual": max_raw_residual,
        "max_fair_residual": max_fair_residual,
        #Absolute prediction changes
        "total_change": total_change,
        "mean_abs_change": mean_abs_change,
        "max_abs_change": max_abs_change,
        #Percentage prediction changes
        "mean_abs_pct_change": mean_abs_pct_change,
        "max_increase_pct": max_increase_pct,
        "max_decrease_pct": max_decrease_pct,
        #Individuals affected by threshold
        "n_changed_1pct": n_changed_1pct,
        "n_changed_5pct": n_changed_5pct,
        "n_changed_10pct": n_changed_10pct,
        "n_changed_20pct": n_changed_20pct,
        #Directional changes (counts)
        "n_increased": n_increased,
        "n_decreased": n_decreased,
        "n_unchanged": n_unchanged,
        #Directional changes (percentages)
        "pct_increased": pct_increased,
        "pct_decreased": pct_decreased,
        "pct_unchanged": pct_unchanged,
        #Flagging parity
        "raw_flag_parity": raw_flag_parity,
        "fair_flag_parity": fair_flag_parity,
        #Top N composition
        "top_n_flagged": top_n_flagged,
        "top_n_composition": top_n_composition,
    }
