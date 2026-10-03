# WILDA Team 6 - Customer Churn Analysis

## Project

**Customer Churn Analysis for Telecommunication Company**

This repository contains the completed project deliverables for customer data preparation, customer segmentation, and predictive churn modeling.

The project uses data analysis, K-Means clustering, and an Artificial Neural Network (ANN) to understand customer behaviour, identify customer segments, and predict customer churn.

## Stage 2-1: Data Preparation

- Preprocessed customer dataset
- 80/20 training and testing sets
- Missing value and data quality checks
- Categorical variable encoding
- Feature scaling
- Reproducible data-preparation code
- Saved preprocessing objects

## Stage 2-2: Clustering Analysis

- Elbow-method analysis and supporting visualisations
- Optimal number of clusters: **K = 3**
- Trained K-Means model
- Customer cluster assignment dataset
- Cluster profiles, labels, and business insights
- PCA and supporting cluster visualisations
- Reproducible clustering notebook/script
- Clustering analysis report

## Stage 3: Predictive Modeling

An Artificial Neural Network (ANN) was developed to predict customer churn using the prepared Stage 2 dataset.

### ANN Architecture

- Input layer: 16 customer features
- Hidden layer 1: 32 neurons with ReLU activation
- Hidden layer 2: 16 neurons with ReLU activation
- Hidden layer 3: 8 neurons with ReLU activation
- Dropout regularisation
- Output layer: 1 neuron with Sigmoid activation
- Optimiser: Adam
- Loss function: Binary Cross-Entropy

### Model Evaluation

The ANN was evaluated using the untouched Stage 2 testing dataset.

- Accuracy: **77.29%**
- Precision (Churn): **56.92%**
- Recall (Churn): **59.36%**
- F1-Score (Churn): **58.12%**
- ROC-AUC: **0.8046**

The repository also contains the trained ANN model, churn predictions, confusion matrix, training history, feature-importance analysis, and supporting visualisations.

## Repository Structure

- `Data_Preparation/` - Stage 2 data preparation deliverables
- `Clustering_Analysis/` - Stage 2 K-Means clustering deliverables
- `Predictive_Modeling/` - Stage 3 ANN predictive modeling deliverables
- `Dataset_ATS_v2.csv` - Original project dataset

## Key Project Outcome

The project combines customer segmentation and predictive modeling to identify customer groups and customers with higher churn risk. These findings can support targeted customer retention strategies and data-driven decision-making.

## Team

**WILDA Team 6**

ACS WIL Project - Customer Churn Analysis for Telecommunication Company
