
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.5: CUSTOMER SEGMENT VISUALIZATIONS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load customer segments
df = pd.read_csv("outputs/customer_segments.csv")

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Create visualization folder
os.makedirs("outputs/visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------
# 1. Customer Count by Cluster
# ------------------------------------------

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Cluster")
plt.title("Customer Distribution Across Clusters")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/visualizations/12_cluster_distribution.png", dpi=300)
plt.close()

# ------------------------------------------
# 2. Churn Rate by Cluster
# ------------------------------------------

churn_profile = df.groupby("Cluster")["Exited"].mean() * 100

plt.figure(figsize=(8, 5))
sns.barplot(x=churn_profile.index, y=churn_profile.values)
plt.title("Churn Rate Across Customer Clusters")
plt.xlabel("Customer Cluster")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/13_cluster_churn_rate.png", dpi=300)
plt.close()

# ------------------------------------------
# 3. Average Age by Cluster
# ------------------------------------------

age_profile = df.groupby("Cluster")["Age"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=age_profile.index, y=age_profile.values)
plt.title("Average Customer Age by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Age")
plt.tight_layout()
plt.savefig("outputs/visualizations/14_cluster_average_age.png", dpi=300)
plt.close()

# ------------------------------------------
# 4. Average Balance by Cluster
# ------------------------------------------

balance_profile = df.groupby("Cluster")["Balance"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=balance_profile.index, y=balance_profile.values)
plt.title("Average Customer Balance by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Balance")
plt.tight_layout()
plt.savefig("outputs/visualizations/15_cluster_average_balance.png", dpi=300)
plt.close()

# ------------------------------------------
# 5. Active Member Percentage by Cluster
# ------------------------------------------

active_profile = df.groupby("Cluster")["IsActiveMember"].mean() * 100

plt.figure(figsize=(8, 5))
sns.barplot(x=active_profile.index, y=active_profile.values)
plt.title("Active Customer Percentage by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Active Members (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/16_cluster_active_members.png", dpi=300)
plt.close()

# ------------------------------------------
# 6. Average Products by Cluster
# ------------------------------------------

product_profile = df.groupby("Cluster")["NumOfProducts"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=product_profile.index, y=product_profile.values)
plt.title("Average Number of Products by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Number of Products")
plt.tight_layout()
plt.savefig("outputs/visualizations/17_cluster_average_products.png", dpi=300)
plt.close()

# ------------------------------------------
# 7. Geography Distribution by Cluster
# ------------------------------------------

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Cluster", hue="Geography")
plt.title("Geographical Distribution Across Clusters")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/visualizations/18_cluster_geography.png", dpi=300)
plt.close()

# ------------------------------------------
# 8. Gender Distribution by Cluster
# ------------------------------------------

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Cluster", hue="Gender")
plt.title("Gender Distribution Across Clusters")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/visualizations/19_cluster_gender.png", dpi=300)
plt.close()

print("\n========== SEGMENT VISUALIZATION SUMMARY ==========")
print("Customer Count:\n", df["Cluster"].value_counts().sort_index())
print("\nChurn Rate (%):\n", churn_profile.round(2))
print("\nAverage Age:\n", age_profile.round(2))
print("\nAverage Balance:\n", balance_profile.round(2))
print("\nActive Members (%):\n", active_profile.round(2))
print("\nAverage Products:\n", product_profile.round(2))

print("\nAll 8 customer segment visualizations saved successfully!")
print("\n========== STEP 3.5 COMPLETED ==========")

