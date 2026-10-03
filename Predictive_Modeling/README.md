# WILDA Team 6 - Stage 3 Predictive Modeling

This folder contains the Stage 3 ANN predictive modeling deliverables for the Customer Churn Analysis project.

## Contents
- `ANN_Model_Architecture.docx` - architecture, training design, and evaluation approach.
- `Code/ANN_Churn_Model.ipynb` - notebook walkthrough of the ANN workflow.
- `Code/train_ann.py` - reproducible training and evaluation script.
- `Model/trained_ann_model.pt` - trained PyTorch ANN checkpoint with input feature names and model metadata.
- `Results/` - predictions, metrics, confusion matrix, training history, feature importance, and visualisations.
- `requirements.txt` - Python dependencies.

## Model design
The model uses 16 Stage 2 prepared inputs, hidden layers of 32, 16 and 8 neurons with ReLU activation, dropout (0.20) after the first two hidden layers, and one sigmoid-equivalent binary output (trained with BCEWithLogitsLoss). Adam is used for optimisation. A class weight of 1.5 is applied to the churn class to improve sensitivity to churners. Early stopping uses a stratified validation subset of the Stage 2 training data.

## Test performance
- Accuracy: 77.29%
- Precision (Churn): 56.92%
- Recall (Churn): 59.36%
- F1-score (Churn): 58.12%
- ROC-AUC: 0.8046

The untouched Stage 2 test set contains 1,409 customers. The confusion matrix is TN=867, FP=168, FN=152, TP=222.
