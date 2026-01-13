########################################################################################################################
# This script loads the loan experiment data from Kaggle and prepares for training
########################################################################################################################

import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

#use kagglehub if possible
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    _KAGGLE_AVAILABLE = True
except Exception:
    _KAGGLE_AVAILABLE = False


HANDLE = "larsen0966/sba-loans-case-data-set"  #kaggle dataset handle we want to load

#options to use when loading the data in
PANDAS_KW = dict(
    low_memory=False, keep_default_na=True, na_values=["", "NA", "N/A", "null", "None"]
)

#columns from raw dataset to drop
DROP_ALIASES: Dict[str, List[str]] = {
    "Selected": ["Selected"],
    "LoanNr_ChkDgt": ["LoanNr_ChkDgt"],
    "Name": ["Name", "BorrowerName"],
    "City": ["City"],
    "State": ["State"],
    "Zip": ["Zip", "ZipCode", "Zip3"],
    "ApprovalDate": ["ApprovalDate"],
    "ChgOffDate": ["ChgOffDate"],
    "DisbursementDate": ["DisbursementDate"],
    "BalanceGross": ["BalanceGross"],
}

#columns to one hot encode where we take top 2 most frequent categories and call all other instances "Other"
TOP2_PLUS_OTHER = ["BankState", "RevLineCr"]

#columns to one-hot encode
ONE_HOT = [
    "ApprovalFY",
    "NewExist",
    "FranchiseCode",
    "RevLineCr",
    "LowDoc",
    "New",
    "RealEstate",
    "Recession",
    "BankState",
    "Bank",
    "NAICS",
]

#columns to use already numeric
CONTINUOUS_CANDIDATES = [
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
    "GrAppv",
    "SBA_Appv",
    "Portion",
    "daysterm",
    "DisbursementGross",
    "DisbursementGross_log",
]


def dataset_autopath(handle: str) -> str:
    """
    Finds and returns path to main data file if downloaded from kagglehub

    @param handle (str): kaggle dataset handle

    @returns path to dataset (str)
    """

    if not _KAGGLE_AVAILABLE:
        raise RuntimeError(
            "kagglehub not installed. pip install -U kagglehub[pandas-datasets]"
        )
    root = Path(kagglehub.dataset_download(handle))
    exts = [".parquet", ".csv", ".tsv"]
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError("No supported files found in kagglehub cache.")
    files.sort(key=lambda p: (-p.stat().st_size))
    return str(files[0].relative_to(root))


def load_raw_df(
    handle: str, rel_path: str | None, local_file: str | None
) -> pd.DataFrame:
    """
    Function to get the original Loan Dataset from Kaggle.

    @param handle (str): Kaggle dataset handle
    @param rel_path (str): if relative path used
    @param local_file (str): if file exists already

    @returns Kaggle dataset as pd.DataFrame object
    """

    if local_file:  #read in local kaggle dataset locally if it exists
        p = Path(local_file)
        if not p.exists():
            raise FileNotFoundError(f"Local file not found: {local_file}")
        if p.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, **PANDAS_KW)
        df.columns = [c.strip() for c in df.columns]
        return df

    if not _KAGGLE_AVAILABLE:
        raise RuntimeError("kagglehub requested but not installed.")
    if not rel_path:
        rel_path = dataset_autopath(handle)
        print(f"[info] Auto-detected file_path='{rel_path}'")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS, handle, rel_path, pandas_kwargs=PANDAS_KW
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def first_present(aliases: List[str], columns: Iterable[str]) -> str | None:
    """
    Helper function to search list of possible col name aliases &
    return first that actually exists in a given set of columns(matches case-insensitively)
    """
    lower_map = {c.lower(): c for c in columns}
    for cand in aliases:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_typos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper func to fix typos in cols in the Loan Dataset
    """
    if "RetainedJOb" in df.columns:
        df = df.rename(columns={"RetainedJOb": "RetainedJob"})
    for alt in ["NAICS?", "NAICSCode"]:
        if alt in df.columns and "NAICS" not in df.columns:
            df = df.rename(columns={alt: "NAICS"})
    if "DisbusrementGross" in df.columns and "DisbursementGross" not in df.columns:
        df = df.rename(columns={"DisbusrementGross": "DisbursementGross"})
    return df


def drop_listed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to safely remove columns specified
    """
    to_drop = []
    for aliases in DROP_ALIASES.values():
        hit = first_present(aliases, df.columns)
        if hit:
            to_drop.append(hit)
    return df.drop(columns=to_drop, errors="ignore")


def map_bank(series: pd.Series) -> pd.Series:
    """
    Helper function to one-hot encode top 2 recurring categories of bank col and "Other" category otherwise
    """
    s = series.astype("string").fillna("Missing").str.upper()

    def classify(x):
        if "WELLS" in x and "FARGO" in x:
            return "Wells Fargo"
        if "BANK" in x and "AMERICA" in x:
            return "Bank of America"
        return "Other"

    return s.apply(classify)


def fit_top2(series: pd.Series) -> List[str]:
    return (
        series.astype("string").fillna("Missing").value_counts().head(2).index.tolist()
    )


def apply_top2_with_map(
    df: pd.DataFrame, top2_map: Dict[str, List[str]]
) -> pd.DataFrame:
    out = df.copy()
    for col, top2 in top2_map.items():
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .fillna("Missing")
                .apply(lambda x: x if x in top2 else "Other")
            )
    return out


def one_hot_frame(df_part: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Helper function to one-hot-encode cols specified in df
    """
    out = df_part.copy()
    for c in cols:
        if c in out.columns:
            dmy = pd.get_dummies(out[c].astype("category"), prefix=c, dtype=int)
            out = pd.concat([out.drop(columns=[c]), dmy], axis=1)
    return out


def compute_scaler_stats(df: pd.DataFrame, cols: List[str]):
    """
    Hekper func top get means/std of cols for standardization
    """
    means, stds, meds = {}, {}, {}
    for c in cols:
        col = pd.to_numeric(df[c], errors="coerce")
        med = float(np.nanmedian(col))
        col = col.fillna(med)
        mean = float(col.mean())
        std = float(col.std(ddof=0) or 1.0)
        means[c], stds[c], meds[c] = mean, std, med
    return means, stds, meds


def apply_scale(df: pd.DataFrame, cols: List[str], means, stds, meds):
    """
    Helper func to scale/standardize dfs
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(meds[c])
            out[c] = (out[c] - means[c]) / stds[c]
    return out


def add_protected_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 2 binary protected attributes
    1. Protected_NewExist: based on NewExist column (1 if new business, 0 if existing)
    2. Protected_Urban: based on UrbanRural column (1 if urban, 0 otherwise)

    These will be placed as the last 2 columns in dataset
    """
    out = df.copy()

    if "NewExist" in out.columns:
        ne = pd.to_numeric(out["NewExist"], errors="coerce").fillna(1).astype(int)
        out["Protected_NewExist"] = (ne == 2).astype(
            int
        )  #1 if new (2), 0 if existing (1)
    else:
        out["Protected_NewExist"] = 0

    #Second protected attr: UrbanRural (urban = 1, rural/undefined = 0)
    if "UrbanRural" in out.columns:
        ur = pd.to_numeric(out["UrbanRural"], errors="coerce").fillna(0).astype(int)
        out["Protected_Urban"] = (ur == 1).astype(int)
    else:
        out["Protected_Urban"] = 0

    return out


def move_to_last(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Move specified columns to the end of the dataframe. Used for protected attr
    """
    remaining = [c for c in df.columns if c not in cols]
    present_cols = [c for c in cols if c in df.columns]
    return df[remaining + present_cols]


#--------------------


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures Default column (i.e. target column) matches with other columns indicating loan default status
    """
    out = df.copy()
    if "MIS_Status" in out.columns:
        s = (
            out["MIS_Status"]
            .astype(str)
            .str.upper()
            .str.replace(r"\s+", "", regex=True)
        )
        out["Default"] = s.isin(["CHGOFF", "CHARGEOFF", "DEFAULT"]).astype(int)
    elif "ChgOffPrinGr" in out.columns:
        out["Default"] = (
            pd.to_numeric(out["ChgOffPrinGr"], errors="coerce") > 0
        ).astype(int)
    else:
        raise ValueError("Cannot create target: need MIS_Status or ChgOffPrinGr.")
    return out


def clean_and_split(df: pd.DataFrame, data_seed):
    """
    Driver function that calls all helper funcs
    """
    df = normalize_typos(df)
    print(df.columns)
    df = build_target(df)
    df = df.drop(columns=[c for c in ["MIS_Status", "ChgOffPrinGr"] if c in df.columns])
    df = drop_listed(df)
    if "Bank" in df.columns:
        df["Bank"] = map_bank(df["Bank"])
    if "DisbursementGross" in df.columns:
        dg = pd.to_numeric(df["DisbursementGross"], errors="coerce").fillna(0.0)
        df["DisbursementGross"] = dg
        df["DisbursementGross_log"] = np.log1p(dg)

    if "NAICS" in df.columns:
        #convert to string and fill missing
        df["NAICS"] = (
            pd.to_numeric(df["NAICS"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .astype(str)
            .replace("-1", "Missing")
        )
        print(f"Number of unique NAICS values: {df['NAICS'].nunique()}")

    #add protected attributes BEFORE splitting (so available for stratification if needed)
    df = add_protected_attributes(df)

    for col in CONTINUOUS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    y = df["Default"].astype(int).values
    X = df.drop(columns=["Default"]).copy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=data_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=data_seed
    )

    top2_map = {
        col: fit_top2(X_train[col]) for col in TOP2_PLUS_OTHER if col in X_train.columns
    }
    X_train = apply_top2_with_map(X_train, top2_map)
    X_val = apply_top2_with_map(X_val, top2_map)
    X_test = apply_top2_with_map(X_test, top2_map)

    #One-hot encode
    one_hot_cols = [c for c in ONE_HOT if c in X_train.columns]
    X_train = one_hot_frame(X_train, one_hot_cols)
    train_cols = X_train.columns.tolist()

    X_val = one_hot_frame(X_val, one_hot_cols).reindex(columns=train_cols, fill_value=0)
    X_test = one_hot_frame(X_test, one_hot_cols).reindex(
        columns=train_cols, fill_value=0
    )

    cont_cols = [c for c in CONTINUOUS_CANDIDATES if c in train_cols]
    means, stds, meds = compute_scaler_stats(X_train, cont_cols)
    X_train = apply_scale(X_train, cont_cols, means, stds, meds)
    X_val = apply_scale(X_val, cont_cols, means, stds, meds)
    X_test = apply_scale(X_test, cont_cols, means, stds, meds)

    X_train = X_train.fillna(0.0)[train_cols]
    X_val = X_val.fillna(0.0)[train_cols]
    X_test = X_test.fillna(0.0)[train_cols]

    #move both protected attributes to the last 2 columns
    X_train = move_to_last(X_train, ["Protected_NewExist", "Protected_Urban"])
    X_val = move_to_last(X_val, ["Protected_NewExist", "Protected_Urban"])
    X_test = move_to_last(X_test, ["Protected_NewExist", "Protected_Urban"])

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print(
        "Last 2 features (protected):",
        (
            X_train.columns[-2:].tolist()
            if len(X_train.columns) >= 2
            else "<insufficient columns>"
        ),
    )

    #print stats for both protected attributes
    for i, col in enumerate(["Protected_NewExist", "Protected_Urban"]):
        if col in X_train.columns:
            print(f"\n{col} statistics:")
            print(f"  Train: {X_train[col].value_counts().to_dict()}")
            print(f"  Val: {X_val[col].value_counts().to_dict()}")
            print(f"  Test: {X_test[col].value_counts().to_dict()}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_sba_splits(handle: str = HANDLE, data_seed=1, file_path: str | None = None):
    """
    Load and clean Loan dataset

    @param handle (str): Kaggle dataset handle
    @param data_seed (int): seed to use when splitting daata
    @param file_path (str): path to load data

    @returns X_train, y_train, X_val, y_val, X_test, y_test
    """

    df_raw = load_raw_df(handle, None, file_path)
    return clean_and_split(df_raw, data_seed)


def get_args():
    ap = argparse.ArgumentParser(
        description="SBA data loader (no leakage, 2 protected attributes)."
    )
    ap.add_argument("--handle", type=str, default=HANDLE, help="Kaggle dataset handle.")
    ap.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Optional local CSV/Parquet file path.",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Run a quick LogisticRegression baseline AUC.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    X_train, y_train, X_val, y_val, X_test, y_test = load_sba_splits(
        handle=args.handle, file_path=args.file_path
    )

    if args.baseline:
        #Use all features except the protected attributes for baseline
        clf = LogisticRegression(max_iter=2000, n_jobs=None)
        clf.fit(X_train.iloc[:, :-2], y_train)  #Exclude last 2 protected columns
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val.iloc[:, :-2])[:, 1])
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test.iloc[:, :-2])[:, 1])
        print(f"Validation AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    else:
        print("Returned arrays ready for model training")
        #--- Optional: save to disk ---
        out_dir = Path("sba_splits")
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "X_train.npy", X_train)
        np.save(out_dir / "y_train.npy", y_train)
        np.save(out_dir / "X_val.npy", X_val)
        np.save(out_dir / "y_val.npy", y_val)
        np.save(out_dir / "X_test.npy", X_test)
        np.save(out_dir / "y_test.npy", y_test)
        print(f"Saved preprocessed splits to {out_dir.resolve()}")
