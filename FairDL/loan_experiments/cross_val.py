########################################################################################################################
# This script performs cross validation for the Penalty/Strict Penalty models explored for the loan experiments
########################################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import csv, os

import baseline_model_loan
from load_data import load_sba_splits
from configs import loan_xp_params
from run_loan_experiments import (
    get_inference_loss,
    get_dataloaders,
    process_datasets,
    set_seed,
)
from penalty_model_loan import batch_mean_gap_logits


#vars to store current models and dataset info for saving on interrupt
current_fair_model = None
current_baseline_model = None
current_dataset_index = None
current_epoch = None
current_model_type = None


#csv for tracking lambda results
RESULTS_CSV = "./results/penalty_lambda_metrics_large.csv"

#create directory if needed
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

#write header once
with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda", "data_seed", "best_val_loss", "best_gap1", "best_gap2"])


#Main training function
def train_models(model_type, data_seed):
    """
    Driver function that loads/splits dataset and trains models of specified type

    @param model_type (str): model keyword to train
    @param data_seed (int): seed when splitting data
    """

    ###################################################################################
    ###################################LOAD DATA #####################################
    ###################################################################################

    set_seed(123456789)

    #Get data
    X_train, y_train, X_val, y_val, X_test, y_test = load_sba_splits(
        data_seed=data_seed
    )
    X_train = X_train.drop(columns=["xx", "UrbanRural"])
    X_val = X_val.drop(columns=["xx", "UrbanRural"])
    X_test = X_test.drop(columns=["xx", "UrbanRural"])

    #get positive class rate for model weight's bias initialization
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    p_pos = (y_train.sum() / len(y_train)).item()

    #get class imbalance ratio for BCEWithLogitsLoss
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight_ratio = neg_count / pos_count

    protected_col_1 = X_train.columns[-2]
    protected_col_2 = X_train.columns[-1]

    print(
        f"Loaded Loan Data splits -> "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )
    print(f"Protected attribute 1 (column -2): {protected_col_1}")
    print(f"Protected attribute 2 (column -1): {protected_col_2}")

    #get tensors
    train_dataset, val_dataset, test_dataset = process_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test, protected_cols=[-2, -1]
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    ###################################################################################
    #########################DEFINE MODEL AND TRAIN ##################################
    ###################################################################################

    #data splitting seed can affect #of columns (since some small classes may not appear for one-hot encoding), so pass info to init via shape[1]

    model_epochs_max = 100
    best_val_loss = float("inf")
    patience = 180
    patience_counter = 0

    #Training loop
    all_train_loss = []
    all_val_loss = []
    all_test_loss = []

    #options of lambdas for penalty models
    if model_type == "penalty":
        #lambdas = [1e-4, 1e-3, 1e-2, .1, 1, 5, 10, 100, 1000]
        lambdas = [0.01, 0.1, 1, 5, 10, 25, 100]
    elif model_type == "strict_penalty":
        lambdas = [100, 1000]

    for lam in lambdas:
        model = baseline_model_loan.BaselineRegressionModel(
            p_pos, num_cols=X_train.shape[1]
        )

        #Set up configs for training
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_ratio]))
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=60, min_lr=8e-6
        )
        total_params = sum(p.numel() for p in model.parameters())

        #reset early-stopping state per lambda
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        for epoch in range(model_epochs_max):

            print(
                "----------------------------------------------------------------------------------------------------------------"
            )
            epoch_loss = 0.0
            for xb, yb in train_loader:

                if model_type == "penalty":
                    pred = model(xb)
                    loss = (
                        criterion(pred.squeeze(1), yb.float().view(-1))
                        + (
                            batch_mean_gap_logits(pred, xb, -1)
                            + batch_mean_gap_logits(pred, xb, -2)
                        )
                        * lam
                    )  #maybe use penalty model version
                elif model_type == "strict_penalty":
                    pred = model(xb)
                    loss = (
                        criterion(pred.squeeze(1), yb.float().view(-1))
                        + (
                            batch_mean_gap_logits(pred, xb, -1)
                            + batch_mean_gap_logits(pred, xb, -2)
                        )
                        * lam
                    )  #maybe use penalty model version

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / loan_xp_params.get_num_batches_train()
            all_train_loss.append(avg_loss)

            #get inference loss
            val_loss, overall_rate, per_group_rate = get_inference_loss(
                model, val_loader, pos_weight_ratio
            )

            scheduler.step(val_loss)
            all_val_loss.append(val_loss)
            model.train()

            #early stopping check
            if val_loss < best_val_loss - 0.00005:
                best_val_loss = val_loss
                best_gap1 = batch_mean_gap_logits(pred, xb, -1)
                best_gap2 = batch_mean_gap_logits(pred, xb, -2)
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch + 1} / Seed {data_seed}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(
                    best_model_state
                )  #get best model in terms of validation set loss
                break

        #log to CSV
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    model_type,
                    data_seed,
                    lam,
                    best_val_loss,
                    float(best_gap1),
                    float(best_gap2),
                ]
            )

    #save model now that training is done, and record test set loss
    model_name = f"model_loan_{model_type}_"
    model_name += ".pth"
    torch.save(model.state_dict(), model_name)


def get_args():
    """Parse command-line arguments for credit data preprocessing."""
    parser = argparse.ArgumentParser(description="Train models for cross validation")

    parser.add_argument(
        "--model_type",
        choices=["penalty", "strict_penalty"],
        default="penalty",
        help="Type of model to train",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    #repeat experiment using different data seeds
    for data_seed in range(0, 25):
        train_models(args.model_type, data_seed)
