
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.9: MODEL COMPARISON
# ==========================================

import pandas as pd

# Load both segmentation datasets
demographic = pd.read_csv(
    "outputs/customer_segments.csv"
)

behavioral = pd.read_csv(
    "outputs/behavioral_customer_segments.csv"
)

overall_churn = demographic["Exited"].mean() * 100

# ------------------------------------------
# 1. Demographic Model Summary
# ------------------------------------------

demo_profile = demographic.groupby("Cluster").agg(
    Customers=("CustomerId", "count"),
    Churned=("Exited", "sum")
)

demo_profile["Churn_Rate"] = (
    demo_profile["Churned"] /
    demo_profile["Customers"] * 100
)

demo_high_risk = demo_profile[
    demo_profile["Churn_Rate"] > overall_churn
]["Customers"].sum()

# ------------------------------------------
# 2. Behavioral Model Summary
# ------------------------------------------

behavior_profile = behavioral.groupby(
    "Behavioral_Cluster"
).agg(
    Customers=("CustomerId", "count"),
    Churned=("Exited", "sum")
)

behavior_profile["Churn_Rate"] = (
    behavior_profile["Churned"] /
    behavior_profile["Customers"] * 100
)

behavior_high_risk = behavior_profile[
    behavior_profile["Churn_Rate"] > overall_churn
]["Customers"].sum()

# ------------------------------------------
# 3. Model Comparison
# ------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Number of Clusters",
        "Features Used",
        "Best Silhouette Score",
        "Highest Cluster Churn Rate (%)",
        "Lowest Cluster Churn Rate (%)",
        "Customers in Above-Average-Churn Clusters"
    ],

    "Demographic Model": [
        len(demographic),
        demographic["Cluster"].nunique(),
        13,
        0.2004,
        demo_profile["Churn_Rate"].max(),
        demo_profile["Churn_Rate"].min(),
        demo_high_risk
    ],

    "Behavioral Model": [
        len(behavioral),
        behavioral["Behavioral_Cluster"].nunique(),
        8,
        0.1526,
        behavior_profile["Churn_Rate"].max(),
        behavior_profile["Churn_Rate"].min(),
        behavior_high_risk
    ]
})

# ------------------------------------------
# 4. Display Comparison
# ------------------------------------------

print("\n========== SEGMENTATION MODEL COMPARISON ==========")

print(comparison.to_string(index=False))

print("\n========== DEMOGRAPHIC MODEL ==========")
print(demo_profile.round(2).to_string())

print("\n========== BEHAVIORAL MODEL ==========")
print(behavior_profile.round(2).to_string())

print("\nOverall Bank Churn Rate:",
      round(overall_churn, 2), "%")

# ------------------------------------------
# 5. Save Comparison
# ------------------------------------------

comparison.to_csv(
    "outputs/segmentation_model_comparison.csv",
    index=False
)

demo_profile.to_csv(
    "outputs/demographic_cluster_comparison.csv"
)

behavior_profile.to_csv(
    "outputs/behavioral_cluster_comparison.csv"
)

print("\nAll comparison files saved successfully!")

print("\n========== STEP 3.9 COMPLETED ==========")
