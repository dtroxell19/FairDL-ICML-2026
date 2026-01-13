########################################################################################################################
#This script includes the main function that trains models used in the employee performance experiments
########################################################################################################################

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import argparse
import os

import fair_model_emp as fair_model_emp
import baseline_model_emp as baseline_model_emp
from configs import emp_xp_params

pro_cols = [18, 19, 21, 27, 29]  #columns that hold protected groups

#vars to store current models and dataset info for saving on interrupt
current_fair_model = None
current_baseline_model = None
current_dataset_index = None
current_epoch = None
current_model_type = None  #'fair' or 'baseline'


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


def process_datasets(X, y):
    """
    Convert pandas DataFrames/Series into PyTorch TensorDatasets.
    Automatically handles categorical/string targets by mapping to numeric codes.

    Args:
        X_train, X_val, X_test : pd.DataFrame
        y_train, y_val, y_test : pd.Series (can be categorical or string)

    Returns:
        train_dataset, val_dataset, test_dataset : TensorDataset objects
    """

    #Convert categorical/string targets to numeric codes
    def to_numeric(y):
        if hasattr(y, "cat"):  #categorical dtype
            return y.cat.codes
        elif y.dtype == "O":  #object/string dtype
            return y.astype("category").cat.codes
        else:
            return y

    y = to_numeric(y)
    X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    #Extract protected attribute from training set
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    return train_dataset


def get_dataloaders(train_dataset):
    """
    Helper function to get data loaders based on TensorDataset objects

    @param train_dataset, val_dataset, test_dataset: torch TensorDataset objects

    @returns train_loader , val_loader, test_loader: torch DataLoader objects

    """
    #define number of minibatches and get DataLoaders
    batch_size = int(len(train_dataset) / emp_xp_params.get_num_batches_train())

    print(f"batch_size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def compute_constraint_violations(predictions, x, y, protected_cols):
    """
    Compute sum of constraint violations for Lagrangian penalty model

    @param predictions: model predictions (tensor, shape [batch_size] or [batch_size, 1])
    @param x: input features (tensor, shape [batch_size, n_features])
    @param y: ground truth labels (tensor, shape [batch_size] or [batch_size, 1])
    @param protected_cols: list of protected attribute column indices
    @param slack: (unused in this version, kept for compatibility)

    @return: total gap penalty (scalar tensor, differentiable)
    """
    #Ensure tensors are properly shaped
    predictions = predictions.squeeze()
    y = y.squeeze()

    #protected cols are binary
    x_binary = x.clone()
    for col in protected_cols:
        x_binary[:, col] = (x_binary[:, col] > 0.5).float()

    total_violation = torch.tensor(0.0, device=predictions.device, requires_grad=True)

    #For each protected column, penalize the gap
    for col_idx, col in enumerate(protected_cols):
        #Get mask for the two groups in this column
        mask_0 = x_binary[:, col] == 0
        mask_1 = x_binary[:, col] == 1

        #Compute mean residual for each group (marginal over other attributes)
        if mask_0.sum() > 0 and mask_1.sum() > 0:
            residual_0 = torch.abs(predictions[mask_0].mean() - y[mask_0].mean())
            residual_1 = torch.abs(predictions[mask_1].mean() - y[mask_1].mean())

            total_violation = total_violation + (residual_0 + residual_1)

    return total_violation


def save_results_to_csv(loss_dict, model_type, csv_path="fairness_results.csv"):
    """
    Save evaluation results to CSV, appending if file exist

    @param loss_dict: dictionary of results from evaluate_fairness_audit
    @param model_type: string identifier for the model (e.g., 'baseline', 'cvxpy_layer')
    @param csv_path: path to CSV file
    """
    row_data = {"model_type": model_type}

    #Add all metrics except nested dictionaries
    for key, value in loss_dict.items():
        if key not in [
            "raw_flagging_rates",
            "fair_flagging_rates",
            "group_raw_residuals",
            "group_fair_residuals",
        ]:
            row_data[key] = value

    #Convert to DataFrame with single row
    df_new = pd.DataFrame([row_data])

    #Append to existing CSV or create new one
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)

    print(f"\nResults saved to {csv_path}")


def train_models(model_type, penalty_lambda=None):
    """
    Driver function that loads data and trains model of specified type

    @param model_type (str): type of model to train ('fair' or 'baseline' or 'penalty')
    @param penalty_lambda (float | None): value of hyperparameter weighting term if penalty model used
    """
    set_seed(1234)

    folder = "./"
    names = ["X", "y"]

    X, y = [pickle.load(open(os.path.join(folder, f"{n}.pkl"), "rb")) for n in names]
    set_seed(seed=1234)

    train_dataset = process_datasets(X, y)
    train_loader = get_dataloaders(train_dataset)

    #get lower and upper bounds for predictions. Since y were normalized to 0-1, make it 0 to 1
    constraint_lower = 0
    constraint_upper = 1

    ###################################################################################
    #########################DEFINE MODEL AND TRAIN ##################################
    ###################################################################################
    if model_type == "fair":
        model = fair_model_emp.FairModelcvxpy(
            constraint_lower, constraint_upper, pro_cols
        )

    elif model_type == "baseline":
        model = baseline_model_emp.BaselineRegressionModel()

    else:
        model = baseline_model_emp.BaselineRegressionModel(type="penalty")

    #different learning rate starting points based on size of hyperparameter weighting term found to help stabilize training
    criterion = nn.MSELoss()
    if model_type == "penalty" and penalty_lambda > 0.01:
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.66, patience=10, min_lr=8e-16
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.66, patience=10, min_lr=8e-10
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (including frozen): {total_params:,}")

    model_epochs_max = 30000
    if model_type == "penalty":
        patience = 200  #penalty model needed higher to finish training
    else:
        patience = 20
    patience_counter = 0
    #Training loop
    best_loss = float("inf")
    all_train_loss = []
    for epoch in range(model_epochs_max):
        epoch_loss = 0.0
        for xb, yb in train_loader:

            if model_type == "fair":
                pred = model(xb, yb)
                loss = criterion(pred.squeeze(1), yb.float().view(-1))
            elif model_type == "baseline":
                pred = model(xb)
                loss = criterion(pred.squeeze(1), yb.float().view(-1))                
            elif model_type == "penalty":
                pred = model(xb)
                loss = criterion(
                    pred.squeeze(1), yb.float().view(-1)
                ) + penalty_lambda * compute_constraint_violations(
                    pred.squeeze(1), xb, yb.float().view(-1), pro_cols
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / emp_xp_params.get_num_batches_train()
        all_train_loss.append(avg_loss)

        scheduler.step(avg_loss)
        model.train()

        #early stopping check
        if avg_loss < best_loss - 0.0001:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(best_model_state)
            break

    #save model now that training is done, and record test set loss
    if model_type == "penalty":
        model_name = f"model_employee_{model_type}_{penalty_lambda}_"
    else:
        model_name = f"model_employee_{model_type}_"
    model_name += ".pth"
    torch.save(model.state_dict(), model_name)

    if model_type == "fair":
        loss_dict = fair_model_emp.get_test_loss_residuals(
            model, train_loader, protected_cols=pro_cols
        )

    elif model_type == "baseline":
        loss_dict = baseline_model_emp.get_test_loss_residuals(
            model,
            train_loader,
            constraint_lower,
            constraint_upper,
            protected_cols=pro_cols,
        )

    else:
        loss_dict = fair_model_emp.get_test_loss_residuals(
            model,
            train_loader,
            protected_cols=pro_cols,
            type="penalty",
            pred_filename="predictions_penalty.pkl",
        )

    save_results_to_csv(loss_dict, model_type)


def get_args():
    """Parse command-line arguments for credit data preprocessing."""
    parser = argparse.ArgumentParser(description="Train models on synthetic data")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["fair", "baseline", "penalty"],
        default="fair",
        help="Type of model to train",
    )

    return parser.parse_args()


if __name__ == "__main__":

    lambdas = [1, 10, 100, 0.01]
    args = get_args()
    if args.model_type == "penalty":
        for lam in lambdas:
            train_models(args.model_type, lam)
    else:
        train_models(args.model_type)
