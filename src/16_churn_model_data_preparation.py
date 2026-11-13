import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/European_Bank.csv")

print("\n========== ORIGINAL DATASET ==========")
print("Shape:", df.shape)

# Remove unnecessary columns
df = df.drop(columns=["CustomerId", "Surname", "Year"])

# Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create output folder
os.makedirs("outputs/modeling", exist_ok=True)

# Save datasets
X_train.to_csv("outputs/modeling/X_train.csv", index=False)
X_test.to_csv("outputs/modeling/X_test.csv", index=False)
y_train.to_csv("outputs/modeling/y_train.csv", index=False)
y_test.to_csv("outputs/modeling/y_test.csv", index=False)

# Display results
print("\n========== TRAIN-TEST SPLIT ==========")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())
print("\nTraining churn percentage:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts())
print("\nTesting churn percentage:")
print((y_test.value_counts(normalize=True) * 100).round(2))

print("\nAll modeling datasets saved successfully!")
print("\n========== STEP 4.1 COMPLETED ==========")