################################################################################################################################################
#This script obtains lambdas (i.e. hyperparameter weighting terms) used for the Penalty model in Synthetic experimentation
################################################################################################################################################

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import pickle
from pathlib import Path

import penalty_model
from configs import synthetic_xp_params


def set_seed(seed=1):
    """
    Universally set seed for different types of random processes potentially used
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_dataset(df):
    """
    Helper function to get output from data_generation.py and prepare for training
    @param df: raw dataset
    @returns train_dataset, val_dataset, test_dataset: torch TensorDataset objects
    """

    #process dataset
    feature_cols = [
        col for col in df.columns if col.startswith("x") and col != "x_protected"
    ]
    X_features = df[feature_cols].values.astype(
        np.float32
    )  #just features and not group
    X_protected = (
        df["x_protected"].values.astype(np.float32).reshape(-1, 1)
    )  #protected attribute (i.e. group)
    X = np.concatenate(
        [X_protected, X_features], axis=1
    )  #combine: [protected, features] so then group is always first col
    y = df["y_cont"].values.astype(np.float32)  #get target

    #split into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1
    )

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    #get group assignment
    x_protected_train = X_train_tensor[:, 0]
    mask_0 = x_protected_train == 0
    mask_1 = x_protected_train == 1

    X_train_0, y_train_0 = X_train_tensor[mask_0], y_train_tensor[mask_0]
    X_train_1, y_train_1 = X_train_tensor[mask_1], y_train_tensor[mask_1]

    print(f"Group 0 training samples: {len(X_train_0)}")
    print(f"Group 1 training samples: {len(X_train_1)}")
    print(
        f"Group 0 target stats: mean={y_train_0.mean():.3f}, std={y_train_0.std():.3f}"
    )
    print(
        f"Group 1 target stats: mean={y_train_1.mean():.3f}, std={y_train_1.std():.3f}"
    )

    val_size = synthetic_xp_params.get_val_size()
    X_val_tensor = X_test_tensor[:val_size]
    y_val_tensor = y_test_tensor[:val_size]

    X_test_tensor = X_test_tensor[val_size:]
    y_test_tensor = y_test_tensor[val_size:]

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset):
    """
    Helper function to get data loaders based on TensorDataset objects
    @param train_dataset, val_dataset, test_dataset: torch TensorDataset objects
    @returns train_loader , val_loader, test_loader: torch DataLoader objects
    """
    #define number of minibatches and get DataLoaders
    batch_size = int(len(train_dataset) / synthetic_xp_params.get_num_batches_train())
    batch_size_test = int(
        len(test_dataset) / synthetic_xp_params.get_num_batches_test()
    )
    batch_size_val = int(len(val_dataset) / synthetic_xp_params.get_num_batches_test())

    print(f"batch_size: {batch_size}")
    print(f"batch_size_test: {batch_size_test}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    return train_loader, val_loader, test_loader


def batch_mean_gap(pred, xb):
    """
    Gets gap in empirical mean predictions for 2 groups for a single batch

    @param pred: predictions from model
    @param xb: input X batch

    @returns difference in empirical means
    """

    #get sensitive attribute grouping
    s = xb[:, 0]
    m0 = s == 0
    m1 = s == 1

    if m0.sum() < 1 or m1.sum() < 1:
        return pred.new_tensor(0.0)

    return pred[m1].mean() - pred[m0].mean()


def eval_mse_gap(model, loader, criterion):
    """
    Gets normal MSE loss and gap in empirical mean predictions for 2 groups across batches
    """
    model.eval()
    mse_sum, gap_sum, n = 0.0, 0.0, 0
    with torch.inference_mode():

        for xb, yb in loader:
            if (xb[:, 0] == 0).sum() < 1 or (
                xb[:, 0] == 1
            ).sum() < 1:  #skip batch if no obs in a given group
                continue

            p = model(xb).squeeze(1)
            mse_sum += criterion(p, yb.float()).item()
            gap_sum += abs(batch_mean_gap(p, xb)).item()
            n += 1

        return (float("inf"), float("inf")) if n == 0 else (mse_sum / n, gap_sum / n)


@torch.no_grad()
def compute_gap_metrics(model, loader, criterion):
    model.eval()

    #batch-averaged
    mse_sum, gap_sum, n_batches = 0.0, 0.0, 0

    #target stats for normalization
    sum_y, sumsq_y, n_total = 0.0, 0.0, 0

    #group-specific prediction stats
    sum_p_g1, sumsq_p_g1, n1 = 0.0, 0.0, 0
    sum_p_g0, sumsq_p_g0, n0 = 0.0, 0.0, 0

    for xb, yb in loader:
        s = xb[:, 0]
        if (s == 0).sum() < 1 or (s == 1).sum() < 1:
            continue

        p = model(xb).squeeze(1)

        #batch metrics
        mse_sum += criterion(p, yb.float()).item()
        gap_sum += (p[s == 1].mean() - p[s == 0].mean()).abs().item()
        n_batches += 1

        #normalize by target std
        yb = yb.float()
        sum_y += yb.sum().item()
        sumsq_y += (yb**2).sum().item()
        n_total += yb.numel()

        #group stats for global gap + SE
        p1, p0 = p[s == 1], p[s == 0]
        sum_p_g1 += p1.sum().item()
        sumsq_p_g1 += (p1**2).sum().item()
        n1 += p1.numel()
        sum_p_g0 += p0.sum().item()
        sumsq_p_g0 += (p0**2).sum().item()
        n0 += p0.numel()

    mse_batch_avg = float("inf") if n_batches == 0 else (mse_sum / n_batches)

    if n1 == 0 or n0 == 0 or n_total == 0:
        return {
            "mse_batch_avg": mse_batch_avg,
            "gap_global": float("inf"),
            "gap_norm": float("inf"),
        }

    mean1 = sum_p_g1 / n1
    mean0 = sum_p_g0 / n0
    gap_global = abs(mean1 - mean0)

    mean_y = sum_y / n_total
    var_y = max((sumsq_y / n_total) - (mean_y**2), 0.0)
    std_y = math.sqrt(var_y) if var_y > 0 else 0.0
    gap_norm = (gap_global / std_y) if std_y > 0 else float("inf")

    return {
        "mse_batch_avg": mse_batch_avg,
        "gap_global": gap_global,
        "gap_norm": gap_norm,
    }


SLACK_VALUES = [0.05]  #, 0.005]
results = {}  #dict to store slacks and corresponding lambdas

if __name__ == "__main__":
    set_seed(123456789)

    #for each of the 32 dataset scenarios
    for slack in SLACK_VALUES:
        #initialize lambda list
        lambdas_for_slack = []

        for dataset_index in range(32):

            current_dataset_index = dataset_index

            #get datasets and dataloaders. Can read in data or uncomment to generate first
            #'''
            #df, dataset_info = data_generation.generate_dataset_by_index(
            #    dataset_index, n_samples=40000, n_features=150, random_state=random.randint(1, 10000)
            #)
            #'''

            #base directory = the folder that contains datasets/ and dataset_metadata/
            base_dir = (
                Path(__file__).resolve().parent
            )  #-> scripts/synthetic_experiments

            #Load dataset
            df = pd.read_pickle(base_dir / "datasets" / f"dataset_{dataset_index}.pkl")

            #Load metadata
            with open(
                base_dir / "dataset_metadata" / f"dataset_{dataset_index}.pkl", "rb"
            ) as f:
                dataset_info = pickle.load(f)

            constraint_lower = dataset_info["constraint_info"]["lower_bound"]
            constraint_upper = dataset_info["constraint_info"]["upper_bound"]
            print(dataset_index)
            print(dataset_info)
            train_dataset, val_dataset, test_dataset = process_dataset(df)

            train_loader, val_loader, test_loader = get_dataloaders(
                train_dataset, val_dataset, test_dataset
            )

            ###################################################################################
            ########################DEFINE PENALTY MODEL (Benchmark 2) #######################
            ###################################################################################

            current_model_type = "penalty"
            criterion = nn.MSELoss()

            LAMBDA_WEIGHTS = [
                1e-4,
                1e-3,
                1e-2,
                0.1,
                0.5,
                1,
                5,
                10,
                100,
                1000,
            ]  #Will test a vast list of weights in experiments

            pm_epochs_max = 25

            best_val_mse = float("inf")
            patience = 25
            patience_counter = 0
            min_epochs = 4
            candidates = []
            all_lams_info = []
            all_train_loss, all_val_mse, all_test_mse = [], [], []
            best_epoch = 0

            for LAMBDA_WEIGHT in LAMBDA_WEIGHTS:
                #fresh model for each lambda
                pen_model = penalty_model.PenaltyRegressionModel(
                    constraint_lower, constraint_upper
                )
                optimizer = optim.SGD(pen_model.parameters(), lr=5e-4)
                scheduler = ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.66, patience=8, min_lr=8e-10
                )

                best_val_mse = float("inf")
                best_model_state = None
                patience_counter = 0

                for epoch in range(pm_epochs_max):
                    current_epoch = epoch
                    print(
                        "----------------------------------------------------------------------------------------------------------------"
                    )
                    pen_model.train()
                    epoch_loss = 0.0

                    for xb, yb in train_loader:
                        if (xb[:, 0] == 0).sum() < 1 or (xb[:, 0] == 1).sum() < 1:
                            continue

                        #get both terms in lagrangian penalized loss
                        pred = pen_model(xb).squeeze(1)
                        mse = criterion(pred, yb.float())
                        gap = batch_mean_gap(pred, xb)
                        loss = mse + LAMBDA_WEIGHT * gap.abs()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / synthetic_xp_params.get_num_batches_train()
                    all_train_loss.append(avg_loss)

                    if epoch < min_epochs:
                        all_val_mse.append(float("inf"))
                        all_test_mse.append(float("inf"))
                        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")
                        continue

                    #Validate on MSE (keeps selection comparable to baseline)
                    val_mse, val_gap = eval_mse_gap(pen_model, val_loader, criterion)
                    scheduler.step(val_mse)
                    all_val_mse.append(val_mse)

                    test_mse, test_gap = eval_mse_gap(pen_model, test_loader, criterion)
                    all_test_mse.append(test_mse)

                    metrics = compute_gap_metrics(pen_model, val_loader, criterion)
                    val_mse_batch_avg = metrics["mse_batch_avg"]
                    val_gap_global = metrics["gap_global"]
                    val_gap_norm = metrics["gap_norm"]

                    print(
                        f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val MSE: {val_mse:.4f}, "
                        f"Gap: {val_gap:.4f}, Test MSE: {test_mse:.4f}, Test Gap: {test_gap:.4f}"
                    )

                    metrics_test = compute_gap_metrics(
                        pen_model, test_loader, criterion
                    )
                    final_test_mse = metrics_test["mse_batch_avg"]
                    final_test_gap = metrics_test["gap_global"]
                    final_test_gn = metrics_test["gap_norm"]

                    #early stopping check
                    if val_mse < best_val_mse - 5e-5:
                        best_val_mse = val_mse
                        best_epoch = epoch
                        patience_counter = 0
                        best_model_state = pen_model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        pen_model.load_state_dict(best_model_state)
                        break
                if best_model_state is not None:
                    pen_model.load_state_dict(best_model_state)

                val_metrics = compute_gap_metrics(pen_model, val_loader, criterion)
                test_metrics = compute_gap_metrics(pen_model, test_loader, criterion)

                val_gap = val_metrics["gap_global"]
                val_mse = val_metrics["mse_batch_avg"]

                #cache for fallback (so we don't retrain)
                all_lams_info.append(
                    (val_gap, val_mse, LAMBDA_WEIGHT, best_model_state, test_metrics)
                )

                if val_gap <= slack:
                    candidates.append(
                        (LAMBDA_WEIGHT, best_val_mse, best_model_state, test_metrics)
                    )

            #choose lambda
            if candidates:
                candidates.sort(key=lambda x: x[1])  #sort by MSE in validation set
                chosen_lambda, _, chosen_state, _ = candidates[0]
                lambdas_for_slack.append(chosen_lambda)  #add each lambda for dataset
            else:
                print("No candidates met the fairness constraint")
                break

            if chosen_state is None:
                print(
                    f"[Dataset {dataset_index}] Chosen lambda={chosen_lambda} has no saved state (unexpected). Skipping dataset."
                )
                continue

            pen_model = penalty_model.PenaltyRegressionModel(
                constraint_lower, constraint_upper
            )
            pen_model.load_state_dict(chosen_state)

            #Recompute metrics so logs match saved checkpoint
            val_metrics_final = compute_gap_metrics(pen_model, val_loader, criterion)
            test_metrics_final = compute_gap_metrics(pen_model, test_loader, criterion)

            #logging the dataset lambda
            with open(f"lambda_per_dataset_{slack}.csv", "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        dataset_index,
                        chosen_lambda,
                        val_metrics_final["gap_global"],
                        val_metrics_final["mse_batch_avg"],
                        test_metrics_final["mse_batch_avg"],
                        test_metrics_final["gap_global"],
                    ]
                )
            print(
                f"[Dataset {dataset_index}] Chosen lambda: {chosen_lambda}, val_gap: {val_metrics_final['gap_global']:.4f},\
                val_mse: {val_metrics_final['mse_batch_avg']:.4f}, test_gap: {test_metrics_final['gap_global']:.4f}, test_mse: {test_metrics_final['mse_batch_avg']:.4f}"
            )

        results[slack] = lambdas_for_slack

#save each slack
for slack, lambdas in results.items():
    filename = f"lambda_results_slack_{slack}.txt"
    with open(filename, "w") as f:
        for lam in lambdas:
            f.write(f"{lam}\n")
    print(f"Saved {len(lambdas)} lambdas for slack={slack} -> {filename}")
