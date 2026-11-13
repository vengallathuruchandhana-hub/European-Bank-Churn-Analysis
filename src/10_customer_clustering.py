
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.4: FINAL CUSTOMER CLUSTERING
# ==========================================

import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load datasets
scaled_df = pd.read_csv(
    "outputs/customer_segmentation_scaled.csv"
)

original_df = pd.read_csv(
    "data/raw/European_Bank.csv"
)

print("\n========== DATASET INFORMATION ==========")
print("Scaled dataset shape:", scaled_df.shape)
print("Original dataset shape:", original_df.shape)

# Check row alignment
if len(scaled_df) != len(original_df):
    raise ValueError("Dataset row counts do not match!")

# Apply K-Means with 6 clusters
kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(scaled_df)

# Add cluster labels to original data
original_df["Cluster"] = cluster_labels

# Save customer segmentation results
os.makedirs("outputs", exist_ok=True)

original_df.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

# ------------------------------------------
# 1. Cluster Size Analysis
# ------------------------------------------

cluster_sizes = (
    original_df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n========== CUSTOMER DISTRIBUTION ==========")
print(cluster_sizes)

# ------------------------------------------
# 2. Cluster Churn Analysis
# ------------------------------------------

churn_analysis = original_df.groupby("Cluster").agg(
    Total_Customers=("CustomerId", "count"),
    Churned_Customers=("Exited", "sum"),
    Average_Age=("Age", "mean"),
    Average_Balance=("Balance", "mean"),
    Average_CreditScore=("CreditScore", "mean"),
    Active_Members=("IsActiveMember", "sum")
)

churn_analysis["Churn_Rate_Percent"] = (
    churn_analysis["Churned_Customers"]
    / churn_analysis["Total_Customers"]
) * 100

print("\n========== CUSTOMER SEGMENT PROFILE ==========")
print(churn_analysis.round(2))

# Save cluster profile
churn_analysis.to_csv(
    "outputs/cluster_profile.csv"
)

# ------------------------------------------
# 3. Final Evaluation
# ------------------------------------------

score = silhouette_score(
    scaled_df,
    cluster_labels,
    sample_size=2000,
    random_state=42
)

print("\n========== FINAL CLUSTERING SUMMARY ==========")
print("Number of Clusters:", 6)
print("Silhouette Score:", round(score, 4))
print("Total Customers:", len(original_df))

print("\nCustomer segmentation files saved successfully!")
print("\n========== STEP 3.4 COMPLETED ==========")

