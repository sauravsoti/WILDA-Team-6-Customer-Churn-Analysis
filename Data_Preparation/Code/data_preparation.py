"""
WILDA Team 6 - Stage 2 Data Preparation
Customer Churn Analysis for Telecommunication Company
"""
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(os.getcwd(), "Dataset_ATS_v2.csv"),
    os.path.abspath(os.path.join(HERE, "..", "..", "Dataset_ATS_v2.csv")),
]
SOURCE = next((p for p in CANDIDATES if os.path.exists(p)), None)
if SOURCE is None:
    raise FileNotFoundError("Dataset_ATS_v2.csv was not found.")

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(SOURCE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = reader.fieldnames

missing = {
    c: sum((r[c] is None) or (str(r[c]).strip() == "") for r in rows)
    for c in columns
}
assert all(v == 0 for v in missing.values())

feature_cols = [c for c in columns if c != "Churn"]
categorical_cols = [
    "gender", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "Contract"
]
binary_numeric_cols = ["SeniorCitizen"]
continuous_cols = ["tenure", "MonthlyCharges"]

X = np.array([[
    r["gender"], int(r["SeniorCitizen"]), r["Dependents"], int(r["tenure"]),
    r["PhoneService"], r["MultipleLines"], r["InternetService"], r["Contract"],
    float(r["MonthlyCharges"])
] for r in rows], dtype=object)
y = np.array([1 if r["Churn"] == "Yes" else 0 for r in rows], dtype=int)

cat_idx = [feature_cols.index(c) for c in categorical_cols]
bin_idx = [feature_cols.index(c) for c in binary_numeric_cols]
cont_idx = [feature_cols.index(c) for c in continuous_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
        ("binary", "passthrough", bin_idx),
        ("continuous", StandardScaler(), cont_idx),
    ],
    remainder="drop"
)

X_train_processed = preprocessor.fit_transform(X_train).astype(float)
X_test_processed = preprocessor.transform(X_test).astype(float)

encoder = preprocessor.named_transformers_["cat"]
feature_names = []
for col, cats in zip(categorical_cols, encoder.categories_):
    feature_names.extend([f"{col}_{cat}" for cat in cats])
feature_names += binary_numeric_cols + continuous_cols

def write_csv(path, header, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)

# Full encoded dataset for the preprocessed-dataset deliverable.
# Continuous values remain unscaled here for interpretability.
full_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
        ("binary", "passthrough", bin_idx),
        ("continuous", "passthrough", cont_idx),
    ],
    remainder="drop"
)
X_full = full_preprocessor.fit_transform(X).astype(float)

preprocessed_dir = os.path.join(OUT, "Preprocessed_Dataset")
os.makedirs(preprocessed_dir, exist_ok=True)
write_csv(
    os.path.join(preprocessed_dir, "customer_churn_preprocessed.csv"),
    feature_names + ["Churn"],
    [list(row) + [int(target)] for row, target in zip(X_full, y)]
)

split_dir = os.path.join(OUT, "Training_Testing_Sets")
os.makedirs(split_dir, exist_ok=True)
write_csv(os.path.join(split_dir, "X_train.csv"), feature_names, X_train_processed)
write_csv(os.path.join(split_dir, "X_test.csv"), feature_names, X_test_processed)
write_csv(os.path.join(split_dir, "y_train.csv"), ["Churn"], [[int(v)] for v in y_train])
write_csv(os.path.join(split_dir, "y_test.csv"), ["Churn"], [[int(v)] for v in y_test])

from collections import Counter
overall = Counter(y.tolist())
train_counts = Counter(y_train.tolist())
test_counts = Counter(y_test.tolist())
with open(os.path.join(split_dir, "split_summary.txt"), "w", encoding="utf-8") as f:
    f.write(
        "Training/Test Split Summary\n"
        "===========================\n"
        f"Total records: {len(y)}\n"
        f"Training records: {len(y_train)} (80.0%)\n"
        f"Testing records: {len(y_test)} (20.0%)\n"
        "Random state: 42\n"
        "Stratification: Churn target\n\n"
        f"Overall Churn=No (0): {overall[0]}\n"
        f"Overall Churn=Yes (1): {overall[1]}\n"
        f"Train Churn=No (0): {train_counts[0]}\n"
        f"Train Churn=Yes (1): {train_counts[1]}\n"
        f"Test Churn=No (0): {test_counts[0]}\n"
        f"Test Churn=Yes (1): {test_counts[1]}\n"
    )

obj_dir = os.path.join(OUT, "Preprocessing_Objects")
os.makedirs(obj_dir, exist_ok=True)
joblib.dump(preprocessor, os.path.join(obj_dir, "preprocessing_pipeline.joblib"))

assert len(y_train) + len(y_test) == len(y)
assert np.isfinite(X_train_processed).all()
assert np.isfinite(X_test_processed).all()

print("Data preparation completed successfully.")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)
