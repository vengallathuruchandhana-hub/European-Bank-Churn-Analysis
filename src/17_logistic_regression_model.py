
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load training and testing datasets
X_train = pd.read_csv("outputs/modeling/X_train.csv")
X_test = pd.read_csv("outputs/modeling/X_test.csv")
y_train = pd.read_csv("outputs/modeling/y_train.csv").squeeze()
y_test = pd.read_csv("outputs/modeling/y_test.csv").squeeze()

# Identify numerical and categorical columns
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Build Logistic Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_probability), 4))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("outputs/models", exist_ok=True)
joblib.dump(model, "outputs/models/logistic_regression_model.pkl")

# Save predictions
predictions = pd.DataFrame({
    "Actual_Churn": y_test,
    "Predicted_Churn": y_pred,
    "Churn_Probability": y_probability
})

predictions.to_csv(
    "outputs/modeling/logistic_regression_predictions.csv",
    index=False
)

print("\nModel and predictions saved successfully!")
print("\n========== STEP 4.2 COMPLETED ==========")
