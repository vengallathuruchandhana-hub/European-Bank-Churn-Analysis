
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "modeling"
VISUAL_DIR = BASE_DIR / "outputs" / "visualizations"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)

# Load risk predictions
df = pd.read_csv(
    MODEL_DIR / "customer_churn_risk_predictions.csv"
)

# Set category order
risk_order = ["High Risk", "Medium Risk", "Low Risk"]

sns.set_theme(style="whitegrid")

# Chart 1: Customer count by risk category
plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Risk_Category",
    order=risk_order
)

plt.title("Customer Distribution by Churn Risk")
plt.xlabel("Risk Category")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    VISUAL_DIR / "30_customer_risk_distribution.png",
    dpi=300
)
plt.close()

# Chart 2: Actual churn by risk category
plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Risk_Category",
    hue="Actual_Churn",
    order=risk_order
)

plt.title("Actual Churn Distribution Across Risk Categories")
plt.xlabel("Risk Category")
plt.ylabel("Number of Customers")
plt.legend(title="Actual Churn", labels=["Retained", "Churned"])
plt.tight_layout()

plt.savefig(
    VISUAL_DIR / "31_actual_churn_by_risk.png",
    dpi=300
)
plt.close()

# Chart 3: Average predicted churn probability
probability_summary = (
    df.groupby("Risk_Category")["Churn_Probability"]
    .mean()
    .reindex(risk_order)
    .reset_index()
)

probability_summary["Churn_Probability"] *= 100

plt.figure(figsize=(9, 6))

sns.barplot(
    data=probability_summary,
    x="Risk_Category",
    y="Churn_Probability"
)

plt.title("Average Predicted Churn Probability by Risk")
plt.xlabel("Risk Category")
plt.ylabel("Average Churn Probability (%)")
plt.tight_layout()

plt.savefig(
    VISUAL_DIR / "32_average_churn_probability.png",
    dpi=300
)
plt.close()

print("\nAll churn risk visualizations saved successfully!")

print("1. 30_customer_risk_distribution.png")
print("2. 31_actual_churn_by_risk.png")
print("3. 32_average_churn_probability.png")

print("\n========== STEP 5.3 COMPLETED ==========")
