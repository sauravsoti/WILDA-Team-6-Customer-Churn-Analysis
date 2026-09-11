# WILDA Team 6 - Stage 2 Data Preparation

## Project
Customer Churn Analysis for Telecommunication Company

## Source dataset
- Records: 7,043
- Columns: 10
- Target: `Churn`
- Missing values: 0
- Exact duplicate-looking rows: 302 (retained because no customer identifier is available)

## Processing decisions
1. `Churn` is encoded as No = 0 and Yes = 1.
2. Nominal categorical variables are one-hot encoded.
3. `SeniorCitizen` is already a binary 0/1 indicator and is retained without scaling.
4. Continuous numerical variables `tenure` and `MonthlyCharges` are standardised with StandardScaler.
5. Data is split into 80% training and 20% testing sets using `random_state=42`.
6. The split is stratified by `Churn`.
7. The encoder/scaler is fitted only on training data before transforming the test data to avoid data leakage.

## Folder contents
### Preprocessed_Dataset
- `customer_churn_preprocessed.csv` - full encoded dataset with interpretable unscaled numeric values and Churn target.

### Training_Testing_Sets
- `X_train.csv` - model-ready encoded predictors; continuous variables scaled.
- `X_test.csv` - model-ready encoded predictors; continuous variables scaled using training-set parameters.
- `y_train.csv` - training target.
- `y_test.csv` - testing target.
- `split_summary.txt` - size and class-composition documentation.

### Code
- `data_preparation.ipynb` - documented, executable notebook.
- `data_preparation.py` - reproducible script.

### Documentation
- `Data_Preparation_and_Scaling.pdf` - methodology, rationale, code snippets and validation.

### Preprocessing_Objects
- `preprocessing_pipeline.joblib` - fitted preprocessing object.

## Split composition
- Training: 5634 records (4139 No, 1495 Yes)
- Testing: 1409 records (1035 No, 374 Yes)

## Why duplicate-looking rows were retained
The supplied reduced dataset has no customer ID. Identical values across the available fields do not prove that two records are the same customer. Removing them without evidence could delete valid observations.

## Reproducibility
Place `Dataset_ATS_v2.csv` in the repository root. Run the notebook or Python script with Python 3, NumPy, scikit-learn and joblib.
