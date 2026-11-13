
# ==========================================
# PROJECT: European Banking Churn Analysis
# PHASE 3: CUSTOMER SEGMENTATION
# STEP 3.2: FEATURE SCALING
# ==========================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load prepared dataset
df = pd.read_csv("outputs/customer_segmentation_prepared.csv")

print("\n========== ORIGINAL DATA ==========")
print("Shape:", df.shape)

# Initialize StandardScaler
scaler = StandardScaler()

# Scale all segmentation features
scaled_data = scaler.fit_transform(df)

# Convert scaled data into DataFrame
scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\n========== SCALED DATA ==========")
print(scaled_df.head())

print("\n========== SCALED FEATURE MEANS ==========")
print(scaled_df.mean().round(3))

print("\n========== SCALED FEATURE STANDARD DEVIATIONS ==========")
print(scaled_df.std(ddof=0).round(3))

# Save scaled dataset
os.makedirs("outputs", exist_ok=True)

scaled_df.to_csv(
    "outputs/customer_segmentation_scaled.csv",
    index=False
)

print("\nScaled dataset saved successfully!")

print("\n========== FEATURE SCALING COMPLETED ==========")

