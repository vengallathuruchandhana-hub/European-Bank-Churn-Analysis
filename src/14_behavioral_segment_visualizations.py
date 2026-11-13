
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.8: BEHAVIORAL SEGMENT VISUALIZATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load behavioral segments
df = pd.read_csv(
    "outputs/behavioral_customer_segments.csv"
)

os.makedirs("outputs/visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

print("\n========== BEHAVIORAL DATASET ==========")
print("Shape:", df.shape)

# ------------------------------------------
# 1. Customer Distribution
# ------------------------------------------

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Behavioral_Cluster")
plt.title("Customer Distribution by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/22_behavioral_cluster_distribution.png",
    dpi=300
)
plt.close()

# ------------------------------------------
# 2. Churn Rate
# ------------------------------------------

churn = (
    df.groupby("Behavioral_Cluster")["Exited"].mean() * 100
)

plt.figure(figsize=(9, 5))
sns.barplot(x=churn.index, y=churn.values)
plt.title("Churn Rate by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/23_behavioral_churn_rate.png",
    dpi=300
)
plt.close()

# ------------------------------------------
# 3. Average Balance
# ------------------------------------------

balance = df.groupby(
    "Behavioral_Cluster"
)["Balance"].mean()

plt.figure(figsize=(9, 5))
sns.barplot(x=balance.index, y=balance.values)
plt.title("Average Balance by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Average Balance")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/24_behavioral_average_balance.png",
    dpi=300
)
plt.close()

# ------------------------------------------
# 4. Average Age
# ------------------------------------------

age = df.groupby(
    "Behavioral_Cluster"
)["Age"].mean()

plt.figure(figsize=(9, 5))
sns.barplot(x=age.index, y=age.values)
plt.title("Average Age by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Average Age")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/25_behavioral_average_age.png",
    dpi=300
)
plt.close()

# ------------------------------------------
# 5. Average Number of Products
# ------------------------------------------

products = df.groupby(
    "Behavioral_Cluster"
)["NumOfProducts"].mean()

plt.figure(figsize=(9, 5))
sns.barplot(x=products.index, y=products.values)
plt.title("Average Products by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Average Number of Products")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/26_behavioral_average_products.png",
    dpi=300
)
plt.close()

# ------------------------------------------
# 6. Active Member Percentage
# ------------------------------------------

active = (
    df.groupby("Behavioral_Cluster")["IsActiveMember"].mean() * 100
)

plt.figure(figsize=(9, 5))
sns.barplot(x=active.index, y=active.values)
plt.title("Active Member Percentage by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Active Members (%)")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/27_behavioral_active_members.png",
    dpi=300
)
plt.close()

# ------------------------------------------
# 7. Average Credit Score
# ------------------------------------------

credit = df.groupby(
    "Behavioral_Cluster"
)["CreditScore"].mean()

plt.figure(figsize=(9, 5))
sns.barplot(x=credit.index, y=credit.values)
plt.title("Average Credit Score by Behavioral Cluster")
plt.xlabel("Behavioral Cluster")
plt.ylabel("Average Credit Score")
plt.tight_layout()
plt.savefig(
    "outputs/visualizations/28_behavioral_credit_score.png",
    dpi=300
)
plt.close()

print("\n========== BEHAVIORAL SEGMENT SUMMARY ==========")
print("Churn Rate:\n", churn.round(2))
print("\nAverage Balance:\n", balance.round(2))
print("\nAverage Age:\n", age.round(2))
print("\nAverage Products:\n", products.round(2))
print("\nActive Members (%):\n", active.round(2))
print("\nAverage Credit Score:\n", credit.round(2))

print("\nAll 7 behavioral visualizations saved successfully!")
print("\n========== STEP 3.8 COMPLETED ==========")
