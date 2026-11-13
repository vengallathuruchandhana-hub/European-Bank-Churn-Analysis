
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 2: Exploratory Data Analysis
# STEP 2.6: Numerical Feature Churn Analysis
# ==========================================

import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")


# Function to calculate churn rate
def analyze_churn(column):

    result = df.groupby(column, observed=True).agg(
        Total_Customers=("Exited", "count"),
        Churned_Customers=("Exited", "sum"),
        Churn_Rate=("Exited", "mean")
    )

    result["Churn_Rate"] = result["Churn_Rate"] * 100

    print(f"\n========== CHURN BY {column.upper()} ==========")
    print(result.round(2))


# 1. Credit Score groups
df["CreditScoreGroup"] = pd.cut(
    df["CreditScore"],
    bins=[300, 500, 600, 700, 800, 900],
    labels=[
        "300-500",
        "501-600",
        "601-700",
        "701-800",
        "801-900"
    ],
    include_lowest=True
)

analyze_churn("CreditScoreGroup")


# 2. Tenure groups
df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 2, 4, 6, 8, 10],
    labels=[
        "0-2 Years",
        "3-4 Years",
        "5-6 Years",
        "7-8 Years",
        "9-10 Years"
    ]
)

analyze_churn("TenureGroup")


# 3. Balance groups using quartiles
df["BalanceGroup"] = pd.qcut(
    df["Balance"],
    q=4,
    duplicates="drop"
)

analyze_churn("BalanceGroup")


# 4. Estimated Salary groups using quartiles
df["SalaryGroup"] = pd.qcut(
    df["EstimatedSalary"],
    q=4,
    duplicates="drop"
)

analyze_churn("SalaryGroup")


print("\n========== NUMERICAL CHURN ANALYSIS COMPLETED ==========")
