########################################################################################################
#This script is used to acquire synthetic Data and train all models as described in paper
########################################################################################################

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
import signal
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import pickle
from pathlib import Path
import argparse

import fair_model
import baseline_model
import penalty_model
from configs import synthetic_xp_params

#vars to store current models and dataset info for saving on interrupt
current_fair_model = None
current_baseline_model = None
current_dataset_index = None
current_epoch = None
current_model_type = None  #'fair' or 'baseline'


def signal_handler(sig, frame):
    """
    Handle Ctrl+C interrupt and save current model
    """
    print("\n\n=== Interrupt detected! Saving current model... ===")

    try:
        if current_fair_model is not None and current_model_type == "fair":
            model_name = f"fair_model_{current_dataset_index}_interrupted_epoch_{current_epoch}.pth"
            torch.save(current_fair_model.state_dict(), model_name)
            print(f"Fair model saved as: {model_name}")
        elif current_baseline_model is not None and current_model_type == "baseline":
            model_name = f"baseline_model_{current_dataset_index}_interrupted_epoch_{current_epoch}.pth"
            torch.save(current_baseline_model.state_dict(), model_name)
            print(f"Baseline model saved as: {model_name}")

        else:
            print("No model to save (training hasn't started yet)")

    except Exception as e:
        print(f"Error saving model: {e}")

    print("Exiting")
    sys.exit(0)


class GradientMonitor:
    """
    Class to help monitor exploding/vanishing gradients while training
    """

    def __init__(self, threshold=10.0, window_size=100):
        self.threshold = threshold
        self.gradient_norms = []
        self.window_size = window_size
        self.large_gradient_count = 0
        self.total_steps = 0

    def check_gradients(self, model):
        """Check gradients and update statistics"""
        total_norm = 0.0
        max_grad = 0.0

        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
                max_grad = max(max_grad, p.grad.data.abs().max().item())

        total_norm = total_norm**0.5

        #Update statistics
        self.gradient_norms.append(total_norm)
        if len(self.gradient_norms) > self.window_size:
            self.gradient_norms.pop(0)

        self.total_steps += 1
        is_large = total_norm > self.threshold

        if is_large:
            self.large_gradient_count += 1

        return {
            "is_large": is_large,
            "total_norm": total_norm,
            "max_grad": max_grad,
            "avg_norm": sum(self.gradient_norms) / len(self.gradient_norms),
            "large_grad_ratio": self.large_gradient_count / self.total_steps,
        }

    def get_stats(self):
        """
        Get current gradient statistics
        """
        if not self.gradient_norms:
            return {}

        return {
            "current_avg": sum(self.gradient_norms) / len(self.gradient_norms),
            "recent_max": max(self.gradient_norms),
            "recent_min": min(self.gradient_norms),
            "large_gradient_percentage": (self.large_gradient_count / self.total_steps)
            * 100,
        }


#Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def create_deterministic_stratified_batches(dataset, protected_attr_idx, batch_size):
    """Creates batches with exact / constant group membership ratios
    
    @param dataset: torch TensorDataset object
    @param protected_attr_idx: index of protected attribute in dataset features
    @param batch_size: desired batch size

    @returns batches: list of torch tensors with indices for each batch
    """
    X = dataset.tensors[0]
    y = dataset.tensors[1]
    protected = X[:, protected_attr_idx]
    
    idx_0 = torch.where(protected == 0)[0]
    idx_1 = torch.where(protected == 1)[0]
    
    #Calculate exact ratio from data
    n_0_total = len(idx_0)
    n_1_total = len(idx_1)
    ratio = n_0_total / (n_0_total + n_1_total)
    
    #Exact split per batch
    n_0_per_batch = round(batch_size * ratio)
    n_1_per_batch = batch_size - n_0_per_batch
    
    print(f"  Exact per-batch: {n_0_per_batch} group 0, {n_1_per_batch} group 1")
    
    #Shuffle indices
    perm_0 = torch.randperm(len(idx_0))
    perm_1 = torch.randperm(len(idx_1))
    idx_0 = idx_0[perm_0]
    idx_1 = idx_1[perm_1]
    
    #Create batches with exact ratios
    batches = []
    i_0, i_1 = 0, 0
    
    while i_0 + n_0_per_batch <= len(idx_0) and i_1 + n_1_per_batch <= len(idx_1):
        batch_idx_0 = idx_0[i_0:i_0 + n_0_per_batch]
        batch_idx_1 = idx_1[i_1:i_1 + n_1_per_batch]
        
        #Combine and shuffle within batch
        batch_indices = torch.cat([batch_idx_0, batch_idx_1])
        batch_indices = batch_indices[torch.randperm(len(batch_indices))]
        batches.append(batch_indices)
        
        i_0 += n_0_per_batch
        i_1 += n_1_per_batch
    
    return batches


def get_deterministic_stratified_dataloaders(train_dataset, val_dataset, test_dataset, protected_attr_idx=0):
    """Get dataloaders with exact / constant group ratios in each batch
    
    @param train_dataset, val_dataset, test_dataset: torch TensorDataset objects
    @param protected_attr_idx: index of protected attribute in dataset features
    @returns train_loader , val_loader, test_loader: torch DataLoader objects
    """
    
    batch_size_train = int(len(train_dataset) / synthetic_xp_params.get_num_batches_train())
    batch_size_test = int(len(test_dataset) / synthetic_xp_params.get_num_batches_test())
    batch_size_val = int(len(val_dataset) / synthetic_xp_params.get_num_batches_test())
    
    print(f"batch_size_train: {batch_size_train}")
    print(f"batch_size_test: {batch_size_test}")
    
    #Create deterministic batches
    print("\nTraining set:")
    train_batches = create_deterministic_stratified_batches(train_dataset, protected_attr_idx, batch_size_train)
    
    print("\nValidation set:")
    val_batches = create_deterministic_stratified_batches(val_dataset, protected_attr_idx, batch_size_val)
    
    print("\nTest set:")
    test_batches = create_deterministic_stratified_batches(test_dataset, protected_attr_idx, batch_size_test)
    
    #Create subset samplers
    from torch.utils.data import Subset
    
    #Flatten batch indices and create loaders
    train_indices = torch.cat(train_batches).tolist()
    train_subset = Subset(train_dataset, train_indices)
    
    val_indices = torch.cat(val_batches).tolist()
    val_subset = Subset(val_dataset, val_indices)
    
    test_indices = torch.cat(test_batches).tolist()
    test_subset = Subset(test_dataset, test_indices)
    
    #Create batch samplers
    from torch.utils.data import BatchSampler, SequentialSampler
    
    train_batch_sampler = BatchSampler(
        SequentialSampler(range(len(train_indices))),
        batch_size=batch_size_train,
        drop_last=True
    )
    
    val_batch_sampler = BatchSampler(
        SequentialSampler(range(len(val_indices))),
        batch_size=batch_size_val,
        drop_last=True
    )
    
    test_batch_sampler = BatchSampler(
        SequentialSampler(range(len(test_indices))),
        batch_size=batch_size_test,
        drop_last=False
    )
    
    train_loader = DataLoader(train_subset, batch_sampler=train_batch_sampler)
    val_loader = DataLoader(val_subset, batch_sampler=val_batch_sampler)
    test_loader = DataLoader(test_subset, batch_sampler=test_batch_sampler)
    
    return train_loader, val_loader, test_loader

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
    X_features = df[feature_cols].values.astype(np.float32)
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


def get_inference_loss(model, loader):
    """
    Helper function that gets inference MSE loss

    @param model: model to test
    @param loader: torch DataLoader that gets inference data

    @return loss: MSE loss on the data inside the loader object

    """
    criterion = nn.MSELoss()
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            if (xb[:, 0] == 0).sum() < 1 or (
                xb[:, 0] == 1
            ).sum() < 1:  #skip batch if insufficient data for a group
                continue
            pred = model(xb)
            loss += criterion(pred.squeeze(1), yb.float())

    loss = loss / synthetic_xp_params.get_num_batches_test()

    return loss


#Main training function
def train_models(model_type):
    """
    Driver function that loads data and trains model for all 32 synthetic datasets
    """

    set_seed(123456789)

    #load in models and lambdas
    lambdas = None

    if model_type == "penalty":
        with open("lambda_results_slack_0.05.txt") as f:
            lambdas = f.readlines()
        lambdas = [lam.strip() for lam in lambdas]

    elif model_type == "strict_penalty":
        lambdas = [5000 for i in range(32)]

    #sanity check
    if lambdas is not None and len(lambdas) < 32:
        raise ValueError(f"Expected 32 lambdas, got {len(lambdas)}")

    #for each of the 32 dataset scenarios
    for dataset_index in range(32):

        current_dataset_index = dataset_index

        print(
            f"########################DATASET {current_dataset_index}###################"
        )

        #get datasets and dataloaders
        base_dir = Path(__file__).resolve().parent

        df = pd.read_pickle(base_dir / "datasets" / f"dataset_{dataset_index}.pkl")

        print(list(df.columns))
        #metadata_name = f"dataset_metadata/dataset_{current_dataset_index}.pkl"

        #Load metadata
        with open(
            base_dir / "dataset_metadata" / f"dataset_{dataset_index}.pkl", "rb"
        ) as f:
            dataset_info = pickle.load(f)

        train_dataset, val_dataset, test_dataset = process_dataset(df)
        #define number of minibatches
        train_loader, val_loader, test_loader = get_deterministic_stratified_dataloaders(
            train_dataset, val_dataset, test_dataset
        )

        #train_loader, val_loader, test_loader = get_dataloaders(
        #    train_dataset, val_dataset, test_dataset
        #)

        #get constraint info
        constraint_lower = dataset_info["constraint_info"]["lower_bound"]
        constraint_upper = dataset_info["constraint_info"]["upper_bound"]

        ###################################################################################
        #########################DEFINE MODEL AND TRAIN ##################################
        ###################################################################################

        if model_type == "fair":
            model = fair_model.FairModelcvxpy(
                constraint_lower, constraint_upper, val_loader
            )

        elif model_type == "baseline":
            model = baseline_model.BaselineRegressionModel(val_loader)

        elif model_type == "penalty":
            model = penalty_model.PenaltyRegressionModel(
                constraint_lower, constraint_upper
            )

            with open("lambda_results_slack_0.05.txt") as f:
                lambdas = f.readlines()
            lambdas = [lam.strip() for lam in lambdas]
        
        elif model_type == "strict_penalty":
            model = penalty_model.PenaltyRegressionModel(
                constraint_lower, constraint_upper
            )
            lambdas = [5000 for i in range(32)]

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0005)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.66, patience=8, min_lr=8e-10
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters (including frozen): {total_params:,}")

        model_epochs_max = 15000
        best_val_loss = float("inf")
        patience = 25
        patience_counter = 0
        if model_type in ("penalty", "strict_penalty"):
            lam = float(lambdas[dataset_index])  #current lambda for dataset

        #Training loop
        all_train_loss = []
        all_val_loss = []
        
        for epoch in range(model_epochs_max):
            #if b_train < b_tau, reset dual variables at start of each epoch for primal-dual training
            if (
                int(len(train_dataset) / synthetic_xp_params.get_num_batches_train())
                < synthetic_xp_params.b_tau()
                and model_type == "fair"
            ):
                #Reset primal-dual variables for training at start of epoch
                model.lambda_dual_train = 0.0
                model.dual_update_count_train = 0
                model.slack_current = synthetic_xp_params.get_slack()
            
            print(
                "------------------------------------------------------------------------"
            )
            epoch_loss = 0.0
            for xb, yb in train_loader:
                #if insufficient obs for a group in the batch, skip
                if (xb[:, 0] == 0).sum() < 1 or (xb[:, 0] == 1).sum() < 1:
                    continue

                if model_type == "fair" or model_type == "baseline":
                    pred = model(xb)
                    loss = criterion(pred.squeeze(1), yb.float())
                elif model_type == "penalty" or model_type == "strict_penalty":
                    pred = model(xb)
                    loss = (
                        criterion(pred.squeeze(1), yb.float())
                        + lam * penalty_model.batch_mean_gap(pred, xb).abs()
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / synthetic_xp_params.get_num_batches_train()
            all_train_loss.append(avg_loss)

            #Compute and print aggregate training fairness after epoch
            if model_type == "fair":
                train_avg_violation, train_aggregate_fairness_gap, train_theoretical_bound, train_max_dual_seen, train_mean_group_0, train_mean_group_1 = \
                    model.get_aggregate_fairness_violation(train_loader)
                
                print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")
                print(f"  Train Fairness - Gap: {train_aggregate_fairness_gap:.6f}, Target: {synthetic_xp_params.get_slack():.6f}, Violation: {train_avg_violation:.6f}")
                print(f"  Train Mean Group 0: {train_mean_group_0:.6f}, Train Mean Group 1: {train_mean_group_1:.6f}")
                print(f"  Train Theoretical Bound: {train_theoretical_bound:.6f}, Train Max Dual: {train_max_dual_seen:.4f}")

            #get validation loss
            if model_type == "fair":
                #Reset inference dual variables before validation
                model.lambda_dual = 0.0
                model.dual_update_count = 0
                model.slack_current = synthetic_xp_params.get_slack()
            val_loss = get_inference_loss(model, val_loader)
            scheduler.step(val_loss)
            all_val_loss.append(val_loss)
            
            #Compute and print aggregate validation fairness violation
            if model_type == "fair":
                avg_violation, aggregate_fairness_gap, theoretical_bound, max_dual_seen, mean_group_0, mean_group_1 = \
                    model.get_aggregate_fairness_violation(val_loader)
                
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Fairness - Gap: {aggregate_fairness_gap:.6f}, Target: {synthetic_xp_params.get_slack():.6f}, Violation: {avg_violation:.6f}")
                print(f"  Val Mean Group 0: {mean_group_0:.6f}, Val Mean Group 1: {mean_group_1:.6f}")
                print(f"  Val Theoretical Bound: {theoretical_bound:.6f}, Val Max Dual: {max_dual_seen:.4f}, Adaptive η: {model.get_adaptive_eta():.6f}")
            else:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            model.train()

            #early stopping
            if val_loss < best_val_loss - 0.00005:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

            #save model now that training is done, and record test set loss
            model_name = f"model_{model_type}_"
            model_name += str(dataset_index)
            model_name += ".pth"
            torch.save(model.state_dict(), model_name)

        #save model now that training is done, and record test set loss
        model_name = f"model_{model_type}_"
        model_name += str(dataset_index)
        model_name += ".pth"
        torch.save(model.state_dict(), model_name)

        if model_type == "fair":
            model.slack_current = synthetic_xp_params.get_slack()
            fair_loss = model.get_test_loss(model, test_loader)
            avg_violation, aggregate_fairness_gap, theoretical_bound, max_dual_seen, mean_group_0, mean_group_1 = \
                    model.get_aggregate_fairness_violation(test_loader)
            print(f"  Test Fairness - Gap: {aggregate_fairness_gap:.6f}, Target: {synthetic_xp_params.get_slack():.6f}, Violation: {avg_violation:.6f}")
            print(f"  Mean Group 0: {mean_group_0:.6f}, Mean Group 1: {mean_group_1:.6f}")
            print(f"  Theoretical Bound: {theoretical_bound:.6f}, Max Dual: {max_dual_seen:.4f}, Adaptive η: {model.get_adaptive_eta():.6f}")
            with open("test_loss_log.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["fair", dataset_index, fair_loss,aggregate_fairness_gap])

        elif model_type == "baseline":
            mseloss_orig, mseloss,aggregate_fairness_satisfied, aggregate_fairness_gap = model.get_test_loss_large_batch(
                model, test_loader, constraint_lower, constraint_upper
            )
            with open("test_loss_log.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["baseline", dataset_index, mseloss_orig, mseloss,aggregate_fairness_satisfied, aggregate_fairness_gap])

        elif model_type in ("penalty", "strict_penalty"):
            unfairness, loss = penalty_model.get_test_loss(model, test_loader)
            with open("test_loss_log.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([model_type, dataset_index, unfairness, loss])


def get_args():
    """
    Parse command-line arguments for preprocessing
    """
    parser = argparse.ArgumentParser(description="Train models on synthetic data")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["fair", "baseline", "penalty", "strict_penalty"],
        default="fair",
        help="Type of model to train",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_models(args.model_type)
