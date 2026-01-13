########################################################################################################################
#This script outputs the X.pkl and y.pkl data used in the employee performance experiments
########################################################################################################################


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_excel("./Employee_Performance.xls")
print("Original shape:", df.shape)

df = df.drop(columns=["EmpNumber"])

# Create binary Age column
df["Age_50plus"] = (df["Age"] >= 50).astype(int)

# Create binary variable that is 1 if any kind of manager
df["IsManager"] = (
    df["EmpJobRole"].str.contains("Manager", case=False, na=False).astype(int)
)

# Inspect mean hourly rate by Age_50plus
means = df.groupby("Age_50plus")["EmpHourlyRate"].mean()
print("Mean hourly rate by Age_50plus:")
print(means)

# cols to one-hot encode
one_hot_cols = [
    "Gender",
    "EducationBackground",
    "MaritalStatus",
    "EmpDepartment",
    "EmpJobRole",
    "BusinessTravelFrequency",
    "OverTime",
    "Attrition",
]


# One-hot encoding
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False, dtype=int)


y = df["EmpHourlyRate"].copy()
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)

# Define features
X = df.drop(columns=["EmpHourlyRate"])
one_hot_cols_after = [
    c for c in X.columns if X[c].nunique() == 2 and sorted(X[c].unique()) == [0, 1]
]
numeric_cols_to_scale = [
    c
    for c in X.select_dtypes(include=["int64", "float64"]).columns
    if c not in one_hot_cols_after
]
scaler = StandardScaler()
X[numeric_cols_to_scale] = scaler.fit_transform(X[numeric_cols_to_scale])

X.to_pickle("X.pkl")
y.to_pickle("y.pkl")
