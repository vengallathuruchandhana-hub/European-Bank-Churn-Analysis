
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.3: OPTIMAL NUMBER OF CLUSTERS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load scaled dataset
df = pd.read_csv("outputs/customer_segmentation_scaled.csv")

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)

# Values of K to evaluate
k_values = range(2, 11)

inertia_values = []
silhouette_values = []

# Test different cluster numbers
for k in k_values:

    print(f"\nEvaluating K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(df)

    # Elbow Method
    inertia_values.append(kmeans.inertia_)

    # Silhouette Score using a representative sample
    score = silhouette_score(
        df,
        labels,
        sample_size=2000,
        random_state=42
    )

    silhouette_values.append(score)

    print("Inertia:", round(kmeans.inertia_, 2))
    print("Silhouette Score:", round(score, 4))


# Create output folder
os.makedirs("outputs/visualizations", exist_ok=True)

# ------------------------------------------
# 1. Elbow Method Plot
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/10_elbow_method.png",
    dpi=300
)

plt.close()


# ------------------------------------------
# 2. Silhouette Score Plot
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/11_silhouette_score.png",
    dpi=300
)

plt.close()


# Display best silhouette score
best_index = silhouette_values.index(max(silhouette_values))
best_k = list(k_values)[best_index]

print("\n========== CLUSTER EVALUATION SUMMARY ==========")
print("Best K based on Silhouette Score:", best_k)
print("Best Silhouette Score:", round(max(silhouette_values), 4))

print("\nBoth evaluation plots saved successfully!")

print("\n========== CLUSTER EVALUATION COMPLETED ==========")
