
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ))
    ]
)

# Hyperparameter search space
param_grid = {
    "classifier__n_estimators": [200, 300, 400, 500],
    "classifier__max_depth": [8, 10, 12, 15, None],
    "classifier__min_samples_split": [2, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 4, 6],
    "classifier__max_features": ["sqrt", "log2"]
}

# Randomized search
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=6,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=1,
    verbose=2,
    pre_dispatch=1
)

print("\nTraining and tuning Random Forest...")
search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

print("\n========== BEST PARAMETERS ==========")
print(search.best_params_)

print("\nBest Cross-Validation ROC-AUC:",
      round(search.best_score_, 4))

# Test predictions
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n========== TUNED RANDOM FOREST RESULTS ==========")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_probability), 4))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# Save tuned model
os.makedirs("outputs/models", exist_ok=True)

joblib.dump(
    best_model,
    "outputs/models/tuned_random_forest_model.pkl"
)

# Save predictions
predictions = pd.DataFrame({
    "Actual_Churn": y_test,
    "Predicted_Churn": y_pred,
    "Churn_Probability": y_probability
})

predictions.to_csv(
    "outputs/modeling/tuned_random_forest_predictions.csv",
    index=False
)

# Save search results
pd.DataFrame(search.cv_results_).to_csv(
    "outputs/modeling/random_forest_tuning_results.csv",
    index=False
)

print("\nTuned model and results saved successfully!")
print("\n========== STEP 4.6 COMPLETED ==========")
