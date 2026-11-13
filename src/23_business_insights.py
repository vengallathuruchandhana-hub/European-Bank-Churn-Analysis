
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "modeling"

# Load datasets
df = pd.read_csv(BASE_DIR / "data" / "raw" / "European_Bank.csv")

demographic = pd.read_csv(
    OUTPUT_DIR / "customer_segment_business_profile.csv"
)

behavioral = pd.read_csv(
    OUTPUT_DIR / "behavioral_cluster_profile.csv"
)

feature_importance = pd.read_csv(
    MODEL_DIR / "random_forest_feature_importance.csv"
)

# 1. Overall churn analysis
total_customers = len(df)
churned_customers = int(df["Exited"].sum())
overall_churn_rate = df["Exited"].mean() * 100

print("\n========== OVERALL BUSINESS INSIGHTS ==========")
print(f"Total Customers: {total_customers}")
print(f"Churned Customers: {churned_customers}")
print(f"Overall Churn Rate: {overall_churn_rate:.2f}%")

# 2. High-risk demographic segments
print("\n========== HIGH-RISK DEMOGRAPHIC SEGMENTS ==========")

demographic = demographic.sort_values(
    by="Churn_Rate_Percent", ascending=False
)

print(
    demographic[
        [
            "Cluster",
            "Total_Customers",
            "Female_Percentage",
            "Germany_Percentage",
            "France_Percentage",
            "Spain_Percentage",
            "Churn_Rate_Percent"
        ]
    ].head(3).to_string(index=False)
)

# 3. High-risk behavioral segments
print("\n========== HIGH-RISK BEHAVIORAL SEGMENTS ==========")

behavioral = behavioral.sort_values(
    by="Churn_Rate_Percent", ascending=False
)

print(
    behavioral[
        [
            "Behavioral_Cluster",
            "Total_Customers",
            "Churn_Rate_Percent",
            "Average_Age",
            "Average_Balance",
            "Average_Products",
            "Active_Member_Percent"
        ]
    ].head(3).to_string(index=False)
)

# 4. Top important features
print("\n========== TOP 5 IMPORTANT FEATURES ==========")

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
)

print(
    feature_importance[
        ["Feature", "Importance"]
    ].head(5).to_string(index=False)
)

# 5. Business recommendations
recommendations = [
    {
        "Area": "Germany",
        "Insight": "Germany has a higher churn rate than France and Spain.",
        "Recommendation": "Investigate customer complaints, pricing, and service experience in Germany."
    },
    {
        "Area": "Inactive Customers",
        "Insight": "Inactive customers show higher churn than active customers.",
        "Recommendation": "Use personalized engagement campaigns and service reminders."
    },
    {
        "Area": "Product Usage",
        "Insight": "Customers with 3 or 4 products show very high observed churn rates, but these groups are small.",
        "Recommendation": "Review product suitability and customer satisfaction before cross-selling."
    },
    {
        "Area": "Customer Age",
        "Insight": "Age is the most important feature in the Random Forest model.",
        "Recommendation": "Study age-group needs and provide suitable banking services."
    },
    {
        "Area": "Behavioral Segments",
        "Insight": "Some behavioral clusters have substantially higher churn rates.",
        "Recommendation": "Prioritize retention outreach for high-churn clusters and monitor campaign outcomes."
    }
]

recommendations_df = pd.DataFrame(recommendations)

recommendations_df.to_csv(
    OUTPUT_DIR / "business_recommendations.csv",
    index=False
)

print("\n========== BUSINESS RECOMMENDATIONS ==========")
print(recommendations_df.to_string(index=False))

print("\nBusiness insights saved successfully!")
print("File: outputs/business_recommendations.csv")

print("\n========== STEP 5.1 COMPLETED ==========")
