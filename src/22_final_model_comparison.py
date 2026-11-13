
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load all three model predictions
prediction_files = {
    "Logistic Regression":
        "outputs/modeling/logistic_regression_predictions.csv",

    "Random Forest":
        "outputs/modeling/random_forest_predictions.csv",

    "Tuned Random Forest":
        "outputs/modeling/tuned_random_forest_predictions.csv"
}

results = []
actual_values = None

for model_name, file_path in prediction_files.items():

    predictions = pd.read_csv(file_path)

    if actual_values is None:
        actual_values = predictions["Actual_Churn"]
    else:
        assert actual_values.equals(
            predictions["Actual_Churn"]
        ), "Actual values do not match!"

    y_true = predictions["Actual_Churn"]
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

# Create comparison dataframe
comparison = pd.DataFrame(results)

# Round metrics
metric_columns = [
    "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"
]

comparison[metric_columns] = comparison[metric_columns].round(4)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# Identify best model for each metric
print("\n========== BEST MODEL BY METRIC ==========")

for metric in metric_columns:

    best_model = comparison.loc[
        comparison[metric].idxmax(), "Model"
    ]

    print(f"{metric}: {best_model}")

# Identify overall model based on ROC-AUC
best_index = comparison["ROC_AUC"].idxmax()
best_model = comparison.loc[best_index, "Model"]

print("\nBest model based on ROC-AUC:", best_model)

# Save report
os.makedirs("outputs/modeling", exist_ok=True)

comparison.to_csv(
    "outputs/modeling/final_model_comparison.csv",
    index=False
)

print("\nFinal model comparison saved successfully!")
print("\n========== STEP 4.7 COMPLETED ==========")
