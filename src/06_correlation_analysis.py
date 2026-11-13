
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 2: EXPLORATORY DATA ANALYSIS
# STEP 2.8: CORRELATION ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")

# Select numerical columns
# Exclude Year and CustomerId because they are not useful
# for this customer-behaviour correlation analysis.

numeric_df = df.select_dtypes(include="number").drop(
    columns=["Year", "CustomerId"],
    errors="ignore"
)

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Print correlation with churn
churn_correlation = (
    correlation_matrix["Exited"]
    .drop("Exited")
    .sort_values(ascending=False)
)

print("\n========== CORRELATION WITH CUSTOMER CHURN ==========")
print(churn_correlation.round(3))

# Create output folder
os.makedirs("outputs/visualizations", exist_ok=True)

# Plot correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix - European Banking Customers")
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/09_correlation_heatmap.png",
    dpi=300
)

plt.close()

print("\nCorrelation heatmap saved successfully!")
print("\n========== CORRELATION ANALYSIS COMPLETED ==========")
