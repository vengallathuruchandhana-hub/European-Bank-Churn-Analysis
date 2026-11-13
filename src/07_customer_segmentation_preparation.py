
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.1: DATA PREPARATION
# ==========================================

import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")

print("\n========== ORIGINAL DATASET ==========")
print("Shape:", df.shape)

# Remove unnecessary columns
# CustomerId and Surname are identifiers.
# Year is constant in this dataset.
# Exited is the target outcome, not a clustering feature.

segmentation_df = df.drop(
    columns=["CustomerId", "Surname", "Year", "Exited"]
)

# Convert categorical columns into numerical columns
segmentation_df = pd.get_dummies(
    segmentation_df,
    columns=["Geography", "Gender"],
    drop_first=False,
    dtype=int
)

print("\n========== SEGMENTATION DATA ==========")
print("Shape:", segmentation_df.shape)

print("\n========== FIRST FIVE RECORDS ==========")
print(segmentation_df.head())

print("\n========== DATA TYPES ==========")
print(segmentation_df.dtypes)

print("\n========== MISSING VALUES ==========")
print(segmentation_df.isnull().sum())

# Save prepared dataset
os.makedirs("outputs", exist_ok=True)

segmentation_df.to_csv(
    "outputs/customer_segmentation_prepared.csv",
    index=False
)

print("\nPrepared dataset saved successfully!")

print("\n========== DATA PREPARATION COMPLETED ==========")
