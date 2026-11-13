
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 2: EDA VISUALIZATIONS
# STEP 2.7
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")

# Create output folder
os.makedirs("outputs/visualizations", exist_ok=True)

# Set chart style
sns.set_theme(style="whitegrid")

# ------------------------------------------
# 1. Customer Churn Distribution
# ------------------------------------------

churn_counts = df["Exited"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.pie(
    churn_counts,
    labels=["Retained", "Churned"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig("outputs/visualizations/01_churn_distribution.png")
plt.close()


# ------------------------------------------
# 2. Geography-wise Churn Rate
# ------------------------------------------

geo_churn = df.groupby("Geography")["Exited"].mean() * 100

plt.figure(figsize=(8, 5))
sns.barplot(x=geo_churn.index, y=geo_churn.values)
plt.title("Churn Rate by Geography")
plt.xlabel("Geography")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/02_geography_churn.png")
plt.close()


# ------------------------------------------
# 3. Gender-wise Churn Rate
# ------------------------------------------

gender_churn = df.groupby("Gender")["Exited"].mean() * 100

plt.figure(figsize=(7, 5))
sns.barplot(x=gender_churn.index, y=gender_churn.values)
plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/03_gender_churn.png")
plt.close()


# ------------------------------------------
# 4. Active vs Inactive Churn
# ------------------------------------------

active_churn = df.groupby("IsActiveMember")["Exited"].mean() * 100
active_churn.index = ["Inactive", "Active"]

plt.figure(figsize=(7, 5))
sns.barplot(x=active_churn.index, y=active_churn.values)
plt.title("Churn Rate: Active vs Inactive Customers")
plt.xlabel("Customer Status")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/04_active_member_churn.png")
plt.close()


# ------------------------------------------
# 5. Age Group Churn Rate
# ------------------------------------------

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "Above 65"]
)

age_churn = df.groupby("AgeGroup", observed=True)["Exited"].mean() * 100

plt.figure(figsize=(9, 5))
sns.barplot(x=age_churn.index, y=age_churn.values)
plt.title("Churn Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/05_age_group_churn.png")
plt.close()


# ------------------------------------------
# 6. Number of Products vs Churn Rate
# ------------------------------------------

product_churn = df.groupby("NumOfProducts")["Exited"].mean() * 100

plt.figure(figsize=(8, 5))
sns.barplot(x=product_churn.index, y=product_churn.values)
plt.title("Churn Rate by Number of Products")
plt.xlabel("Number of Products")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/06_product_churn.png")
plt.close()


# ------------------------------------------
# 7. Credit Score vs Churn Rate
# ------------------------------------------

df["CreditScoreGroup"] = pd.cut(
    df["CreditScore"],
    bins=[300, 500, 600, 700, 800, 900],
    labels=["300-500", "501-600", "601-700", "701-800", "801-900"],
    include_lowest=True
)

credit_churn = df.groupby("CreditScoreGroup", observed=True)["Exited"].mean() * 100

plt.figure(figsize=(9, 5))
sns.barplot(x=credit_churn.index, y=credit_churn.values)
plt.title("Churn Rate by Credit Score")
plt.xlabel("Credit Score Group")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("outputs/visualizations/07_credit_score_churn.png")
plt.close()


# ------------------------------------------
# 8. Balance Group vs Churn Rate
# ------------------------------------------

df["BalanceGroup"] = pd.qcut(
    df["Balance"],
    q=4,
    duplicates="drop"
)

balance_churn = df.groupby("BalanceGroup", observed=True)["Exited"].mean() * 100

plt.figure(figsize=(10, 5))
sns.barplot(
    x=balance_churn.index.astype(str),
    y=balance_churn.values
)
plt.title("Churn Rate by Balance Group")
plt.xlabel("Balance Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/visualizations/08_balance_churn.png")
plt.close()


print("\n========== EDA VISUALIZATIONS COMPLETED ==========")
print("Charts saved in outputs/visualizations/")
