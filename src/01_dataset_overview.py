
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 2: Dataset Understanding
# STEP 2.1: Dataset Overview
# ==========================================

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/raw/European_Bank.csv")

# 2. Display dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

# 3. Display first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 4. Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 5. Display dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 6. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 7. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Total duplicates:", df.duplicated().sum())

print("\n========== DATASET OVERVIEW COMPLETED ==========")
