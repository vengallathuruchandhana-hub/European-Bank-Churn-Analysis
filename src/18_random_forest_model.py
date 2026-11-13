
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load datasets
X_train = pd.read_csv("outputs/modeling/X_train.csv")
X_test = pd.read_csv("outputs/modeling/X_test.csv")
y_train = pd.read_csv("outputs/modeling/y_train.csv").squeeze()
y_test = pd.read_csv("outputs/modeling/y_test.csv").squeeze()

# Identify feature types
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Build Random Forest pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n========== RANDOM FOREST RESULTS ==========")

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

joblib.dump(
    model,
    "outputs/models/random_forest_model.pkl"
)

# Save predictions
predictions = pd.DataFrame({
    "Actual_Churn": y_test,
    "Predicted_Churn": y_pred,
    "Churn_Probability": y_probability
})

predictions.to_csv(
    "outputs/modeling/random_forest_predictions.csv",
    index=False
)

print("\nModel and predictions saved successfully!")
print("\n========== STEP 4.3 COMPLETED ==========")
