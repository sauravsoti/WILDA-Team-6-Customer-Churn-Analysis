# Stage 2-2 - Clustering Analysis

## Project
Customer Churn Analysis for Telecommunication Company  
WILDA Team 6

## Objective
Identify distinct customer segments using K-Means clustering and provide interpretable visualisations and insights.

## Method
- `Churn` excluded from clustering inputs.
- Nominal categorical variables one-hot encoded.
- `tenure` and `MonthlyCharges` standardised with StandardScaler.
- `SeniorCitizen` retained as 0/1.
- K values 2-10 evaluated using the elbow method (WCSS).
- The strongest change in the WCSS slope occurs at **K = 3**.
- Final K-Means model: `K=3`, `random_state=42`, `n_init=20`.
- PCA is used only for 2-D visualisation, not to train the K-Means model.

## Final customer segments
- **Cluster 0 - Lower-Cost Lower-Churn Customers**: 2278 customers; average tenure 29.1 months; average monthly charges $27.96; observed churn rate 13.0%.
- **Cluster 1 - Newer Higher-Churn Customers**: 2534 customers; average tenure 12.7 months; average monthly charges $75.81; observed churn rate 48.3%.
- **Cluster 2 - Established Higher-Cost Lower-Churn Customers**: 2231 customers; average tenure 58.0 months; average monthly charges $89.78; observed churn rate 15.6%.

## Folder contents
- `Optimal_Clusters/` - elbow analysis data, elbow chart, and supporting silhouette chart
- `KMeans_Model/` - trained K-Means model, scaler, PCA object, feature names, and cluster labels
- `Visualisations/` - PCA, cluster-size, churn-rate, and tenure/monthly-charge charts
- `Results/` - customer dataset with cluster assignments, cluster profile table, and business insights
- `Code/` - Jupyter notebook and Python script
- `Documentation/` - final clustering analysis report

## Important interpretation note
`Churn` was **not** used to create the clusters. It is used only after clustering to understand how observed churn differs between the customer segments.

## Reproducibility
Run the notebook/script from the repository root where `Dataset_ATS_v2.csv` is located.
