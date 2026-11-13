
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load predictions
logistic = pd.read_csv(
    "outputs/modeling/logistic_regression_predictions.csv"
)

random_forest = pd.read_csv(
    "outputs/modeling/random_forest_predictions.csv"
)

# Verify actual values match
assert logistic["Actual_Churn"].equals(
    random_forest["Actual_Churn"]
), "Actual target values do not match!"

y_true = logistic["Actual_Churn"]

# Calculate performance metrics
models = {
    "Logistic Regression": logistic,
    "Random Forest": random_forest
}

results = []

for model_name, predictions in models.items():

    y_pred = predictions["Predicted_Churn"]
    y_probability = predictions["Churn_Probability"]

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_probability)
    })

comparison = pd.DataFrame(results)

# Round values
comparison.iloc[:, 1:] = comparison.iloc[:, 1:].round(4)

# Identify best model for each metric
print("\n========== MODEL PERFORMANCE COMPARISON ==========")
print(comparison.to_string(index=False))

print("\n========== BEST MODEL BY METRIC ==========")

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]:
    best_model = comparison.loc[
        comparison[metric].idxmax(), "Model"
    ]

    print(f"{metric}: {best_model}")

# Save comparison
os.makedirs("outputs/modeling", exist_ok=True)

comparison.to_csv(
    "outputs/modeling/model_performance_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")
print("\n========== STEP 4.4 COMPLETED ==========")
