
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "modeling"

# Load tuned Random Forest predictions
df = pd.read_csv(
    MODEL_DIR / "tuned_random_forest_predictions.csv"
)

# Classify customer churn risk
def classify_risk(probability):
    if probability >= 0.60:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Category"] = df["Churn_Probability"].apply(
    classify_risk
)

# Risk category summary
risk_summary = (
    df.groupby("Risk_Category")
    .agg(
        Customer_Count=("Churn_Probability", "count"),
        Average_Churn_Probability=("Churn_Probability", "mean"),
        Actual_Churned=("Actual_Churn", "sum")
    )
    .reset_index()
)

risk_summary["Average_Churn_Probability"] *= 100

# Arrange risk categories
risk_order = ["High Risk", "Medium Risk", "Low Risk"]

risk_summary["Risk_Category"] = pd.Categorical(
    risk_summary["Risk_Category"],
    categories=risk_order,
    ordered=True
)

risk_summary = risk_summary.sort_values("Risk_Category")

# Print results
print("\n========== CUSTOMER CHURN RISK ANALYSIS ==========")
print(f"Total Customers Analyzed: {len(df)}")

print("\n========== RISK CATEGORY SUMMARY ==========")
print(risk_summary.to_string(index=False))

# High-risk customer statistics
high_risk = df[df["Risk_Category"] == "High Risk"]

print("\n========== HIGH-RISK CUSTOMER INSIGHTS ==========")
print(f"High-Risk Customers: {len(high_risk)}")

if len(high_risk) > 0:
    print(
        f"Average High-Risk Probability: "
        f"{high_risk['Churn_Probability'].mean() * 100:.2f}%"
    )
    print(
        f"Actual Churned Among High-Risk Customers: "
        f"{high_risk['Actual_Churn'].sum()}"
    )

# Save outputs
df.to_csv(
    MODEL_DIR / "customer_churn_risk_predictions.csv",
    index=False
)

risk_summary.to_csv(
    MODEL_DIR / "customer_churn_risk_summary.csv",
    index=False
)

print("\nRisk analysis files saved successfully!")
print("1. outputs/modeling/customer_churn_risk_predictions.csv")
print("2. outputs/modeling/customer_churn_risk_summary.csv")

print("\n========== STEP 5.2 COMPLETED ==========")
