"""
WILDA Team 6 - Stage 2-2 Clustering Analysis
Project: Customer Churn Analysis for Telecommunication Company

Method:
- Exclude Churn from clustering inputs.
- One-hot encode nominal categorical features.
- Keep SeniorCitizen as a binary 0/1 indicator.
- Standardise tenure and MonthlyCharges.
- Use the elbow method (WCSS) for K selection.
- Fit K-Means with random_state=42 and n_init=20.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("Dataset_ATS_v2.csv")

categorical_cols = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']
continuous_cols = ['tenure', 'MonthlyCharges']

X_cat = pd.get_dummies(df[categorical_cols], dtype=float)
X_num = df[["SeniorCitizen", "tenure", "MonthlyCharges"]].astype(float).copy()

scaler = StandardScaler()
X_num[continuous_cols] = scaler.fit_transform(X_num[continuous_cols])

X = pd.concat([X_cat.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)

ks = list(range(2, 11))
inertias = []
silhouettes = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Largest change in the WCSS slope identifies the elbow.
second_diff = np.diff(np.array(inertias), n=2)
optimal_k = ks[int(np.argmax(second_diff)) + 1]
print("Selected K:", optimal_k)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(X)

print(df["Cluster"].value_counts().sort_index())
