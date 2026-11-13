
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 2: Exploratory Data Analysis
# STEP 2.3: Initial Statistical Analysis
# ==========================================

import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")

# 1. Statistical summary
print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe())

# 2. Customer distribution by geography
print("\n========== GEOGRAPHY DISTRIBUTION ==========")
print(df["Geography"].value_counts())

# 3. Gender distribution
print("\n========== GENDER DISTRIBUTION ==========")
print(df["Gender"].value_counts())

# 4. Churn distribution
print("\n========== CUSTOMER CHURN DISTRIBUTION ==========")
print(df["Exited"].value_counts())

# 5. Churn percentage
print("\n========== CHURN PERCENTAGE ==========")

churn_percentage = df["Exited"].value_counts(normalize=True) * 100

print(churn_percentage)

# 6. Active customer distribution
print("\n========== ACTIVE CUSTOMER DISTRIBUTION ==========")
print(df["IsActiveMember"].value_counts())

# 7. Credit card distribution
print("\n========== CREDIT CARD DISTRIBUTION ==========")
print(df["HasCrCard"].value_counts())

# 8. Unique years
print("\n========== YEAR DISTRIBUTION ==========")
print(df["Year"].value_counts())

print("\n========== INITIAL ANALYSIS COMPLETED ==========")
