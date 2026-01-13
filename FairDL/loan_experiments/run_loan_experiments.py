########################################################################################################################
#This script includes the main function that trains models used in the loan experiments
########################################################################################################################

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import argparse
import numpy as np

import fair_model_loan
import baseline_model_loan
from load_data import load_sba_splits
from configs import loan_xp_params

#vars to store current models and dataset info for saving on interrupt
current_fair_model = None
current_baseline_model = None
current_dataset_index = None
current_epoch = None
current_model_type = None


def set_seed(seed=1):
    """
    Universally set seed for different types of random processes used

    @param seed (int): seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_datasets(
    X_train, y_train, X_val, y_val, X_test, y_test, protected_cols=None
):
    """
    Convert pandas DataFrames/Series/ndarrays into PyTorch TensorDatasets.
    Ensures all X columns are numeric

    @param X_train: training dataset (not including target)
    @param y_train: training target
    @param X_val: val dataset (not including target)
    @param y_val: val target
    @param X_test: test dataset (not including target)
    @param y_test: test target
    @param protected_cols: list of column indices for protected attributes (default [-2, -1])

    @returns train_dataset (TensorDataset): train dataset to use via torch DataLoader
    @returns val_dataset (TensorDataset): val dataset to use via torch DataLoader
    @returns test_dadtaset (TensorDataset): test dataset to use via torch DataLoader
    """

    if protected_cols is None:
        protected_cols = [-2, -1]  #default: last 2 columns

    def to_numeric_array(y):
        """
        Helper func to convert to numeric array for target
        """
        if hasattr(y, "dtype") and str(getattr(y, "dtype", "")) == "category":
            y = y.cat.codes
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        if isinstance(y, np.ndarray) and y.dtype == object:
            y = pd.Series(y).astype("category").cat.codes.to_numpy()
        return y

    def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper func to convert to numeric df for X
        """
        df = df.copy()

        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            print("[warn] object-typed columns in X:", obj_cols)

        for c in df.columns:
            s = df[c]
            if pd.api.types.is_categorical_dtype(s):  #convert codes in loan dataset
                df[c] = s.cat.codes
            #bool -> int
            elif pd.api.types.is_bool_dtype(s):
                df[c] = s.astype(np.int8)
            #everything else coerced to numeric
            else:
                df[c] = pd.to_numeric(s, errors="coerce")
        #handle odd vals
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        return df

    y_train_num = to_numeric_array(y_train).astype(np.float32)
    y_val_num = to_numeric_array(y_val).astype(np.float32)
    y_test_num = to_numeric_array(y_test).astype(np.float32)
    X_train_num = to_numeric_df(X_train)
    X_val_num = to_numeric_df(X_val)
    X_test_num = to_numeric_df(X_test)

    #sanity check
    assert (
        list(X_train_num.columns) == list(X_val_num.columns) == list(X_test_num.columns)
    ), "Column mismatch across splits after numeric coercion."

    X_train_tensor = torch.tensor(X_train_num.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_num.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_num.values, dtype=torch.float32)
    y_train_tensor = torch.from_numpy(y_train_num).view(-1, 1)
    y_val_tensor = torch.from_numpy(y_val_num).view(-1, 1)
    y_test_tensor = torch.from_numpy(y_test_num).view(-1, 1)

    #Print the protected attributes statistics for both protected columns
    for i, col_idx in enumerate(protected_cols):
        print(f"\nProtected Attribute {i+1} (column {col_idx}):")
        x_protected_train = X_train_tensor[:, col_idx]
        mask_0 = x_protected_train == 0
        mask_1 = x_protected_train == 1

        X_train_0, y_train_0 = X_train_tensor[mask_0], y_train_tensor[mask_0]
        X_train_1, y_train_1 = X_train_tensor[mask_1], y_train_tensor[mask_1]

        print(f"Group 0 training samples: {len(X_train_0)}")
        print(f"Group 1 training samples: {len(X_train_1)}")
        if len(y_train_0) > 0:
            print(
                f"Group 0 target stats: mean={y_train_0.mean().item():.3f}, std={y_train_0.std().item():.3f}"
            )
        if len(y_train_1) > 0:
            print(
                f"Group 1 target stats: mean={y_train_1.mean().item():.3f}, std={y_train_1.std().item():.3f}"
            )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset):
    """
    Helper function to get data loaders based on TensorDataset objects

    @param train_dataset, val_dataset, test_dataset: torch TensorDataset objects

    @returns train_loader , val_loader, test_loader: torch DataLoader objects

    """
    #define number of minibatches and get DataLoaders
    batch_size = int(len(train_dataset) / loan_xp_params.get_num_batches_train())
    batch_size_test = int(len(test_dataset) / loan_xp_params.get_num_batches_test())
    batch_size_val = int(len(val_dataset) / loan_xp_params.get_num_batches_test())

    print(f"batch_size: {batch_size}")
    print(f"batch_size_test: {batch_size_test}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    return train_loader, val_loader, test_loader


def batch_mean_gap_logits(logits, xb, protected_col: int = -1):
    """
    Helper function to return gap in mean logits (used for penalty terms)

    @param logits: predicted logits by model
    @param xb: input batch
    @param protected_col: index of protected attribute

    @returns abs value of gap in mean logits for batch
    """
    z = logits.view(-1)
    s = xb[:, protected_col]
    #masks
    m0, m1 = (s == 0), (s == 1)

    mean0 = z[m0].mean()
    mean1 = z[m1].mean()

    gap = (mean1 - mean0).abs()

    return gap


def get_inference_loss(model, loader, pos_weight: float):
    """
    Helper function that gets inference BCE loss (no projections) and prints misclassification rate
    overall and per true class (y=0 vs y=1).
    Now handles 2 protected attributes independently.

    @param model: model to test
    @param loader: torch DataLoader that gets inference data
    @param pos_weight (float): weighting value for BCEWithLogitsLoss

    @return: metrics tuple
    """

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_errors = 0

    class_errors = {0: 0, 1: 0}  #misclassification count per true class
    class_samples = {0: 0, 1: 0}  #number of samples per true class

    #get all predictions and groups for metric calculation
    all_probs, all_labels = [], []
    all_attr1, all_attr2 = [], []

    with torch.no_grad():
        total_batches = 0
        for xb, yb in loader:  #for all points in inference dataset
            total_batches += 1

            pred = model(xb).squeeze(1)
            yb_float = yb.float().view(-1)

            total_loss += criterion(pred, yb_float)
            prob = torch.sigmoid(pred)

            #BCEWithLogitsLoss takes in logits. Need sigmoid for misclassification rate analysis

            pred_labels = (prob >= 0.5).float()
            total_errors += (pred_labels != yb_float).sum().item()
            total_samples += yb_float.size(0)

            #misclassification per true class
            for cls in [0, 1]:
                mask = yb_float == cls
                class_errors[cls] += (pred_labels[mask] != yb_float[mask]).sum().item()
                class_samples[cls] += mask.sum().item()

            #global metrics
            all_probs.append(prob.cpu())
            all_labels.append(yb_float.cpu())
            all_attr1.append(xb[:, -2].cpu())
            all_attr2.append(xb[:, -1].cpu())

    avg_loss = total_loss / max(1, total_batches)
    overall_misclass = total_errors / total_samples if total_samples > 0 else 0.0

    return (
        avg_loss,
        overall_misclass,
        {cls: class_errors[cls] / class_samples[cls] for cls in [0, 1]},
    )


#Main training function
def train_models(
    model_type, data_seed, penalty_lambda=None, file_name="", size="small"
):
    """
    Driver function that loads/splits dataset and trains models of specified type

    @param model_type (str): model keyword to train
    @param data_seed (int): seed when splitting data
    @param penalty_lambda (float | None): weight of hyperparameter penalty term for Penalty or Strict Penalty models
    @param file_name (str): path to save results to
    @param size (str): Whether to use smaller or larger network architecture options
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
    if model_type == "fair":
        model = fair_model_loan.FairModelcvxpy(
            p_pos=p_pos, num_cols=X_train.shape[1], size=size
        )

    elif model_type == "baseline":
        model = baseline_model_loan.BaselineRegressionModel(
            p_pos, num_cols=X_train.shape[1], size=size
        )

    elif model_type == "penalty":
        model = baseline_model_loan.BaselineRegressionModel(
            p_pos, num_cols=X_train.shape[1], size=size
        )

    elif model_type == "strict_penalty":
        model = baseline_model_loan.BaselineRegressionModel(
            p_pos, num_cols=X_train.shape[1], size=size
        )

    #Set up configs for training
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_ratio]))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=60, min_lr=8e-6
    )

    model_epochs_max = 15000
    best_val_loss = float("inf")
    patience = 180
    patience_counter = 0

    #Training loop
    all_train_loss = []
    all_val_loss = []
    for epoch in range(model_epochs_max):

        print(
            "----------------------------------------------------------------------------------------------------------------"
        )
        epoch_loss = 0.0
        for xb, yb in train_loader:

            if model_type == "fair":
                pred = model(xb)
                loss = criterion(pred.squeeze(1), yb.float().view(-1))
            elif model_type == "baseline":
                pred = model(xb)
                loss = criterion(pred.squeeze(1), yb.float().view(-1))
            elif model_type == "penalty":
                pred = model(xb)

                #penalize gaps in predicted logits for each attribute independently
                loss = (
                    criterion(pred.squeeze(1), yb.float().view(-1))
                    + (
                        batch_mean_gap_logits(pred, xb, -1)
                        + batch_mean_gap_logits(pred, xb, -2)
                    )
                    * penalty_lambda
                )
            elif model_type == "strict_penalty":
                pred = model(xb)
                loss = (
                    criterion(pred.squeeze(1), yb.float().view(-1))
                    + (
                        batch_mean_gap_logits(pred, xb, -1)
                        + batch_mean_gap_logits(pred, xb, -2)
                    )
                    * penalty_lambda
                )

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
            torch.save(model.state_dict(), f"./loan_models/{model_type}_{data_seed}.pth")
            break

    #define all metrics that we record for each model (when applicable)
    CSV_HEADER = [
        "model_type",
        "data_seed",
        "raw_loss",
        "fair_loss",
        "overall_misclass_0.5",
        "misclass_y0",
        "misclass_y1",
        "auc_raw",
        "ap_raw",
        "auc_fair",
        "ap_fair",
        "best_thresh",
        "misclass_best_thresh",
        "mean_prob_diff_attr1",
        "mean_logit_diff_attr1",
        "mean_prob_diff_attr2",
        "mean_logit_diff_attr2",
        "best_f1_score",
        "best_thresh_fair",
        "misclass_best_thresh_fair",
        "mean_prob_diff_attr1_fair",
        "mean_logit_diff_attr1_fair",
        "mean_prob_diff_attr2_fair",
        "mean_logit_diff_attr2_fair",
        "best_f1_score_fair",
    ]

    if (
        model_type != "baseline"
    ):  #Penalty and Strict Penalty procedure as fair model (no projection necessary like Baseline model)

        (
            fair_loss,
            overall_rate,
            per_group_rate,
            auc_score,
            ap_score,
            best_thresh,
            overall_misclass_best,
            mean_prob_diff_attr1,
            mean_logit_diff_attr1,
            mean_prob_diff_attr2,
            mean_logit_diff_attr2,
            best_f1_score,
        ) = fair_model_loan.get_test_loss(
            model, test_loader, pos_weight=pos_weight_ratio
        )

        log_path = Path(file_name)
        with log_path.open(mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(CSV_HEADER)

            #fill in the csv of metrics
            writer.writerow(
                [
                    model_type + "_loan",
                    data_seed,
                    None,  #raw_loss
                    fair_loss,  #fair_loss
                    overall_rate,  #overall_misclass_0.5
                    per_group_rate.get(0, None),  #misclass_y0
                    per_group_rate.get(1, None),  #misclass_y1
                    auc_score,  #auc_raw  (mirror)
                    ap_score,  #ap_raw   (mirror)
                    auc_score,  #auc_fair
                    ap_score,  #ap_fair
                    best_thresh,
                    overall_misclass_best,
                    mean_prob_diff_attr1,
                    mean_logit_diff_attr1,
                    mean_prob_diff_attr2,
                    mean_logit_diff_attr2,
                    best_f1_score,
                    None,  #best_thresh_fair (N/A)
                    None,  #misclass_best_thresh_fair (N/A)
                    None,  #mean_prob_diff_attr1_fair (N/A)
                    None,  #mean_logit_diff_attr1_fair (N/A)
                    None,  #mean_prob_diff_attr2_fair (N/A)
                    None,  #mean_logit_diff_attr2_fair (N/A)
                    None,  #best_f1_score_fair (N/A)
                ]
            )

    else:
        loss_dict = baseline_model_loan.get_test_loss(model, test_loader)

        log_path = Path(file_name)
        with log_path.open(mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:  #if csv col names dont exist, add them
                writer.writerow(CSV_HEADER)

            writer.writerow(
                [
                    "baseline_loan",
                    data_seed,
                    loss_dict.get("raw_loss"),
                    loss_dict.get("fair_loss"),
                    loss_dict.get("overall_misclass_fair"),
                    loss_dict.get("misclass_y0_fair"),
                    loss_dict.get("misclass_y1_fair"),
                    loss_dict.get("auc_raw"),
                    loss_dict.get("ap_raw"),
                    loss_dict.get("auc_fair"),
                    loss_dict.get("ap_fair"),
                    loss_dict.get("best_thresh_raw"),
                    loss_dict.get("overall_misclass_best_raw"),
                    loss_dict.get("mean_prob_diff_raw_attr1"),
                    loss_dict.get("mean_logit_diff_raw_attr1"),
                    loss_dict.get("mean_prob_diff_raw_attr2"),
                    loss_dict.get("mean_logit_diff_raw_attr2"),
                    loss_dict.get("best_f1_score_raw"),
                    loss_dict.get("best_thresh_fair"),
                    loss_dict.get("overall_misclass_best_fair"),
                    loss_dict.get("mean_prob_diff_fair_attr1"),
                    loss_dict.get("mean_logit_diff_fair_attr1"),
                    loss_dict.get("mean_prob_diff_fair_attr2"),
                    loss_dict.get("mean_logit_diff_fair_attr2"),
                    loss_dict.get("best_f1_score_fair"),
                ]
            )


def get_args():
    """Parse command-line arguments for credit data preprocessing."""
    parser = argparse.ArgumentParser(description="Train models on synthetic data")

    parser.add_argument(
        "--model_type",
        choices=["fair", "baseline", "penalty", "strict_penalty"],
        default="fair",
        help="Type of model to train",
    )

    parser.add_argument(
        "--size",
        choices=["small", "large"],
        default="small",
        help="Type of model architecture to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    #load csv to get lowest lambda that meets constraints (if using Penalty or Strict Penalty models)
    df = pd.read_csv(f"./results/penalty_lambda_metrics_{args.size}.csv")

    #clean tensor() from best_val_loss column if needed
    df["best_val_loss"] = (
        df["best_val_loss"]
        .astype(str)
        .str.replace("tensor\\(|\\)", "", regex=True)
        .astype(float)
    )

    #convert numeric columns properly
    df["best_gap1"] = df["best_gap1"].astype(float)
    df["best_gap2"] = df["best_gap2"].astype(float)
    df["lambda"] = df["lambda"].astype(float)
    df["data_seed"] = df["data_seed"].astype(int)

    #precompute best lambda per seed
    best_lambdas = {}

    for seed, group in df.groupby("data_seed"):
        valid_rows = group[(group["best_gap1"] < 0.01) & (group["best_gap2"] < 0.01)]
        if len(valid_rows) > 0:
            best_lambda = valid_rows.sort_values("lambda").iloc[0]["lambda"]
        else:
            #fallback (no lambda meets both constraints)
            best_lambda = group.sort_values("best_val_loss").iloc[0]["lambda"]
        best_lambdas[seed] = best_lambda

    #run experiments
    file_name = f"./results/test_loss_log_{args.size}.csv"
    for data_seed in range(0, 25):
        if args.model_type == "strict_penalty":
            train_models(args.model_type, data_seed, 1000, file_name)
        elif args.model_type == "penalty":
            penalty_lambda = best_lambdas.get(data_seed, 1.0)  #fallback if missing
            train_models(args.model_type, data_seed, penalty_lambda, file_name)
        else:
            train_models(args.model_type, data_seed=data_seed, file_name=file_name)
