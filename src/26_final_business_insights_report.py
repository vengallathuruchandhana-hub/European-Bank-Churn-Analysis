
import pandas as pd
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODELING_DIR = OUTPUT_DIR / "modeling"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD PROJECT RESULTS
# ==========================================

demographic_df = pd.read_csv(
    OUTPUT_DIR / "customer_segment_business_profile.csv"
)

behavioral_df = pd.read_csv(
    OUTPUT_DIR / "behavioral_cluster_profile.csv"
)

feature_df = pd.read_csv(
    MODELING_DIR / "random_forest_feature_importance.csv"
)

risk_df = pd.read_csv(
    MODELING_DIR / "customer_churn_risk_summary.csv"
)

model_df = pd.read_csv(
    MODELING_DIR / "final_model_comparison.csv"
)

# ==========================================
# 3. OVERALL CUSTOMER CHURN ANALYSIS
# ==========================================

total_customers = 10000
total_churned = 2037

overall_churn_rate = (
    total_churned / total_customers
) * 100

# ==========================================
# 4. DEMOGRAPHIC HIGH-RISK SEGMENTS
# ==========================================

high_risk_demographic = demographic_df[
    demographic_df["Churn_Rate_Percent"] > overall_churn_rate
].sort_values(
    by="Churn_Rate_Percent",
    ascending=False
)

# ==========================================
# 5. BEHAVIORAL HIGH-RISK SEGMENTS
# ==========================================

high_risk_behavioral = behavioral_df[
    behavioral_df["Churn_Rate_Percent"] > overall_churn_rate
].sort_values(
    by="Churn_Rate_Percent",
    ascending=False
)

# ==========================================
# 6. TOP CHURN PREDICTION FEATURES
# ==========================================

top_features = feature_df.sort_values(
    by="Importance",
    ascending=False
).head(5)

# ==========================================
# 7. CUSTOMER RISK CATEGORY ANALYSIS
# ==========================================

risk_summary = risk_df.sort_values(
    by="Customer_Count",
    ascending=False
)

# ==========================================
# 8. BEST MODEL INFORMATION
# ==========================================

best_model = model_df.sort_values(
    by="ROC_AUC",
    ascending=False
).iloc[0]

# ==========================================
# 9. GENERATE BUSINESS REPORT
# ==========================================

report = []

report.append("=" * 65)
report.append("EUROPEAN BANKING CUSTOMER CHURN ANALYSIS")
report.append("FINAL BUSINESS INSIGHTS AND RECOMMENDATIONS")
report.append("=" * 65)

report.append(
    f"\nReport Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)

# Overall findings
report.append("\n\n1. OVERALL CUSTOMER CHURN ANALYSIS")
report.append("-" * 50)

report.append(f"Total Customers: {total_customers:,}")
report.append(f"Total Churned Customers: {total_churned:,}")
report.append(f"Overall Churn Rate: {overall_churn_rate:.2f}%")

report.append(
    "\nBusiness Insight: Approximately one in five "
    "customers has left the bank."
)

# Demographic segments
report.append("\n\n2. HIGH-RISK DEMOGRAPHIC SEGMENTS")
report.append("-" * 50)

for _, row in high_risk_demographic.iterrows():

    report.append(
        f"Cluster {int(row['Cluster'])}: "
        f"{int(row['Total_Customers']):,} customers | "
        f"Churn Rate: {row['Churn_Rate_Percent']:.2f}%"
    )

# Behavioral segments
report.append("\n\n3. HIGH-RISK BEHAVIORAL SEGMENTS")
report.append("-" * 50)

for _, row in high_risk_behavioral.iterrows():

    report.append(
        f"Cluster {int(row['Behavioral_Cluster'])}: "
        f"{int(row['Total_Customers']):,} customers | "
        f"Churn Rate: {row['Churn_Rate_Percent']:.2f}%"
    )

# Feature importance
report.append("\n\n4. TOP FIVE CHURN PREDICTION FEATURES")
report.append("-" * 50)

for _, row in top_features.iterrows():

    report.append(
        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"
    )

# Risk analysis
report.append("\n\n5. CUSTOMER CHURN RISK DISTRIBUTION")
report.append("-" * 50)

for _, row in risk_summary.iterrows():

    report.append(
        f"{row['Risk_Category']}: "
        f"{int(row['Customer_Count']):,} customers | "
        f"Average Probability: "
        f"{row['Average_Churn_Probability']:.2f}% | "
        f"Actual Churned: {int(row['Actual_Churned']):,}"
    )

# Model performance
report.append("\n\n6. BEST CHURN PREDICTION MODEL")
report.append("-" * 50)

report.append(f"Model: {best_model['Model']}")
report.append(f"Accuracy: {best_model['Accuracy'] * 100:.2f}%")
report.append(f"Precision: {best_model['Precision'] * 100:.2f}%")
report.append(f"Recall: {best_model['Recall'] * 100:.2f}%")
report.append(f"F1 Score: {best_model['F1_Score'] * 100:.2f}%")
report.append(f"ROC-AUC: {best_model['ROC_AUC'] * 100:.2f}%")

# Business recommendations
report.append("\n\n7. BUSINESS RECOMMENDATIONS")
report.append("-" * 50)

recommendations = [
    "Prioritize customer retention campaigns for high-risk customers.",
    "Investigate the reasons behind higher churn in Germany.",
    "Develop engagement programs for inactive customers.",
    "Review product combinations associated with higher churn.",
    "Provide personalized retention offers to older customer groups.",
    "Monitor high-balance customers with low engagement.",
    "Use churn probabilities to prioritize customer outreach.",
    "Regularly evaluate and retrain the churn prediction model."
]

for number, recommendation in enumerate(recommendations, start=1):
    report.append(f"{number}. {recommendation}")

# Conclusion
report.append("\n\n8. PROJECT CONCLUSION")
report.append("-" * 50)

report.append(
    "This project combined exploratory data analysis, "
    "customer segmentation, churn prediction and business insights "
    "to understand customer attrition in European banking."
)

report.append(
    "The Tuned Random Forest model achieved the best ROC-AUC "
    "among the evaluated models. Customer segmentation identified "
    "groups with different churn patterns."
)

report.append(
    "These findings can help the bank prioritize retention efforts "
    "and make data-driven customer relationship decisions."
)

report.append("\n" + "=" * 65)
report.append("END OF FINAL BUSINESS INSIGHTS REPORT")
report.append("=" * 65)

# ==========================================
# 10. SAVE REPORT
# ==========================================

report_text = "\n".join(report)

report_path = REPORT_DIR / "final_business_insights_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_text)

print(report_text)

print("\n\nFinal business insights report saved successfully!")
print(f"Report Location: {report_path}")

print("\n========== STEP 5.4 COMPLETED ==========")
