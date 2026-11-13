
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 2: Exploratory Data Analysis
# STEP 2.5: Customer Churn Patterns
# ==========================================

import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")

# Function to calculate customer count and churn rate
def analyze_churn(column):

    result = df.groupby(column).agg(
        Total_Customers=("Exited", "count"),
        Churned_Customers=("Exited", "sum"),
        Churn_Rate=("Exited", "mean")
    )

    result["Churn_Rate"] = result["Churn_Rate"] * 100

    print(f"\n========== CHURN BY {column.upper()} ==========")
    print(result.round(2))


# 1. Geography-wise churn
analyze_churn("Geography")

# 2. Gender-wise churn
analyze_churn("Gender")

# 3. Active vs inactive customer churn
analyze_churn("IsActiveMember")

# 4. Credit card ownership and churn
analyze_churn("HasCrCard")

# 5. Number of products and churn
analyze_churn("NumOfProducts")


# 6. Create age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "Above 65"
    ]
)

analyze_churn("AgeGroup")

print("\n========== CHURN PATTERN ANALYSIS COMPLETED ==========")
