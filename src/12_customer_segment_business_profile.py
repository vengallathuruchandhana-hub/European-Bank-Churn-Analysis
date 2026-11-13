
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.6: BUSINESS PROFILING
# ==========================================

import pandas as pd

# Load customer segments
df = pd.read_csv("outputs/customer_segments.csv")

# Build detailed cluster profile
profile = df.groupby("Cluster").agg(
    Total_Customers=("CustomerId", "count"),
    Churned_Customers=("Exited", "sum"),
    Average_Age=("Age", "mean"),
    Average_Credit_Score=("CreditScore", "mean"),
    Average_Balance=("Balance", "mean"),
    Average_Salary=("EstimatedSalary", "mean"),
    Average_Tenure=("Tenure", "mean"),
    Average_Products=("NumOfProducts", "mean"),
    Active_Member_Percentage=("IsActiveMember",
                              lambda x: x.mean() * 100),
    Credit_Card_Percentage=("HasCrCard",
                           lambda x: x.mean() * 100),
    Female_Percentage=("Gender",
                       lambda x: (x == "Female").mean() * 100),
    Germany_Percentage=("Geography",
                        lambda x: (x == "Germany").mean() * 100),
    France_Percentage=("Geography",
                       lambda x: (x == "France").mean() * 100),
    Spain_Percentage=("Geography",
                      lambda x: (x == "Spain").mean() * 100)
)

# Calculate churn percentage
profile["Churn_Rate_Percent"] = (
    profile["Churned_Customers"] /
    profile["Total_Customers"] * 100
)

# Calculate customer share
profile["Customer_Share_Percent"] = (
    profile["Total_Customers"] / len(df) * 100
)

# Round numerical values
profile = profile.round(2)

# Save business profile
profile.to_csv(
    "outputs/customer_segment_business_profile.csv"
)

# Display results
print("\n========== CUSTOMER SEGMENT BUSINESS PROFILE ==========")
print(profile.to_string())

print("\n========== OVERALL BUSINESS SUMMARY ==========")
print("Total Customers:", len(df))
print("Overall Churn Rate:",
      round(df["Exited"].mean() * 100, 2), "%")

print("\nHighest Churn Cluster:",
      profile["Churn_Rate_Percent"].idxmax())

print("Lowest Churn Cluster:",
      profile["Churn_Rate_Percent"].idxmin())

print("\nBusiness profile saved successfully!")
print("\n========== STEP 3.6 COMPLETED ==========")

