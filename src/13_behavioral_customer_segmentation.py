
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.7: BEHAVIORAL CUSTOMER SEGMENTATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------
# 1. Load original customer data
# ------------------------------------------

df = pd.read_csv("outputs/customer_segments.csv")

print("\n========== ORIGINAL DATASET ==========")
print("Shape:", df.shape)

# ------------------------------------------
# 2. Select behavioral and financial features
# ------------------------------------------

features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

X = df[features].copy()

print("\nSelected Features:")
print(features)

print("\nPrepared Dataset Shape:", X.shape)

# Save prepared behavioral data
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)

X.to_csv(
    "outputs/behavioral_customer_prepared.csv",
    index=False
)

# ------------------------------------------
# 3. Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

scaled_df.to_csv(
    "outputs/behavioral_customer_scaled.csv",
    index=False
)

print("\nFeature scaling completed!")

# ------------------------------------------
# 4. Find Optimal Number of Clusters
# ------------------------------------------

k_values = range(2, 11)

inertia_values = []
silhouette_values = []

print("\n========== CLUSTER EVALUATION ==========")

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(scaled_df)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(
        scaled_df,
        labels,
        sample_size=2000,
        random_state=42
    )

    silhouette_values.append(score)

    print(f"\nK = {k}")
    print("Inertia:", round(kmeans.inertia_, 2))
    print("Silhouette Score:", round(score, 4))

# ------------------------------------------
# 5. Save Evaluation Charts
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Behavioral Clustering - Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/20_behavioral_elbow.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Behavioral Clustering - Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/21_behavioral_silhouette.png",
    dpi=300
)

plt.close()

# ------------------------------------------
# 6. Select Best K and Fit Final Model
# ------------------------------------------

best_index = silhouette_values.index(
    max(silhouette_values)
)

best_k = list(k_values)[best_index]

print("\n========== SELECTED MODEL ==========")
print("Best K:", best_k)
print("Best Silhouette Score:",
      round(silhouette_values[best_index], 4))

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

behavioral_labels = final_model.fit_predict(
    scaled_df
)

# ------------------------------------------
# 7. Attach Cluster Labels
# ------------------------------------------

behavioral_df = df.copy()

behavioral_df["Behavioral_Cluster"] = (
    behavioral_labels
)

behavioral_df.to_csv(
    "outputs/behavioral_customer_segments.csv",
    index=False
)

# ------------------------------------------
# 8. Create Behavioral Cluster Profile
# ------------------------------------------

profile = behavioral_df.groupby(
    "Behavioral_Cluster"
).agg(
    Total_Customers=("CustomerId", "count"),
    Churned_Customers=("Exited", "sum"),
    Average_Credit_Score=("CreditScore", "mean"),
    Average_Age=("Age", "mean"),
    Average_Tenure=("Tenure", "mean"),
    Average_Balance=("Balance", "mean"),
    Average_Products=("NumOfProducts", "mean"),
    Active_Members=("IsActiveMember", "mean"),
    Credit_Card_Holders=("HasCrCard", "mean"),
    Average_Salary=("EstimatedSalary", "mean")
)

profile["Churn_Rate_Percent"] = (
    profile["Churned_Customers"] /
    profile["Total_Customers"] * 100
)

profile["Active_Member_Percent"] = (
    profile["Active_Members"] * 100
)

profile["Credit_Card_Percent"] = (
    profile["Credit_Card_Holders"] * 100
)

profile = profile.drop(
    columns=["Active_Members", "Credit_Card_Holders"]
)

profile = profile.round(2)

profile.to_csv(
    "outputs/behavioral_cluster_profile.csv"
)

# ------------------------------------------
# 9. Display Results
# ------------------------------------------

print("\n========== BEHAVIORAL CLUSTER DISTRIBUTION ==========")

print(
    behavioral_df["Behavioral_Cluster"]
    .value_counts()
    .sort_index()
)

print("\n========== BEHAVIORAL CLUSTER PROFILE ==========")

print(profile.to_string())

print("\n========== FINAL SUMMARY ==========")
print("Total Customers:", len(behavioral_df))
print("Number of Clusters:", best_k)
print(
    "Silhouette Score:",
    round(silhouette_values[best_index], 4)
)

print("\nBehavioral segmentation files saved successfully!")

print("\n========== STEP 3.7 COMPLETED ==========")

