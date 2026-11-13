
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Load trained Random Forest model
model = joblib.load(
    "outputs/models/random_forest_model.pkl"
)

# Get transformed feature names
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

# Get Random Forest feature importance
classifier = model.named_steps["classifier"]

importance_values = classifier.feature_importances_

# Create feature importance dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Clean feature names
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

# Save results
os.makedirs("outputs/visualizations", exist_ok=True)

importance_df.to_csv(
    "outputs/modeling/random_forest_feature_importance.csv",
    index=False
)

# Visualize top 10 features
top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Features Influencing Customer Churn Predictions")
plt.xlabel("Feature Importance")
plt.ylabel("Customer Features")
plt.tight_layout()

plt.savefig(
    "outputs/visualizations/29_random_forest_feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis saved successfully!")
print("\n========== STEP 4.5 COMPLETED ==========")
