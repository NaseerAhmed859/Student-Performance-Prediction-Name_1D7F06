"""
model.py
Core AI logic for the Student Performance Predictor (regression version).
Kept separate from the UI (app.py).

Functions:
    load_data(path)
    preprocess_data(data)
    run_model_or_algorithm(X, y, params)
    predict_single(model, student_input)
    grade_category(score)
    generate_explanation(...)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = ["hours_studied_per_week", "attendance", "gpa", "assignment_score"]
TARGET_COLUMN = "final_grade"


def load_data(path):
    """Load the student dataset from a CSV path."""
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def preprocess_data(data):
    """Clean the raw dataframe and split into features/target."""
    df = data.copy()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Basic range validation / clipping to keep things sane
    df["attendance"] = df["attendance"].clip(0, 100)
    df["gpa"] = df["gpa"].clip(0, 4.0)
    df["assignment_score"] = df["assignment_score"].clip(0, 100)
    df["hours_studied_per_week"] = df["hours_studied_per_week"].clip(0, 24 * 7)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def run_model_or_algorithm(X, y, params=None):
    """
    Train a regressor (Random Forest by default) and return the trained
    model + evaluation metrics + train/test split for downstream use.

    params: dict with keys:
        model_type: "random_forest" | "linear_regression"
        n_estimators: int (for random forest)
        test_size: float
        random_state: int
    """
    params = params or {}
    model_type = params.get("model_type", "random_forest")
    n_estimators = params.get("n_estimators", 200)
    test_size = params.get("test_size", 0.25)
    random_state = params.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
    }

    if hasattr(model, "feature_importances_"):
        importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    elif hasattr(model, "coef_"):
        importances = dict(zip(FEATURE_COLUMNS, np.abs(model.coef_)))
    else:
        importances = {}

    return {
        "model": model,
        "model_type": model_type,
        "metrics": metrics,
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def grade_category(score):
    """Map a numeric predicted grade to a category label, per the UI spec."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Poor"


def letter_grade(score):
    """Map a numeric grade to a letter band (used for distribution chart)."""
    if score >= 80:
        return "A (80-100)"
    elif score >= 60:
        return "B (60-79)"
    elif score >= 40:
        return "C (40-59)"
    else:
        return "D (<40)"


def predict_single(model, student_input):
    """
    Predict the final grade for a single new student.
    student_input: dict with keys matching FEATURE_COLUMNS
    """
    X_new = pd.DataFrame([student_input])[FEATURE_COLUMNS]
    pred = float(model.predict(X_new)[0])
    pred = max(0.0, min(100.0, pred))
    return pred


def generate_explanation(student_input=None, predicted_score=None, feature_importances=None):
    """
    Generate short natural-language / checklist-style explanation points.
    Returns a list of (is_positive: bool, text: str) tuples for a single
    prediction, or a summary sentence when explaining the model overall.
    """
    if student_input is not None and predicted_score is not None:
        points = []

        if student_input["attendance"] >= 85:
            points.append((True, "Attendance is high, which positively impacts performance."))
        elif student_input["attendance"] < 60:
            points.append((False, "Attendance is low, which is holding performance back."))
        else:
            points.append((True, "Attendance is moderate."))

        avg_hours = 5.0
        if student_input["hours_studied_per_week"] >= avg_hours:
            points.append((True, "Study hours are above average."))
        else:
            points.append((False, "Study hours are below average."))

        if student_input["gpa"] >= 3.2:
            points.append((True, "Previous GPA is strong."))
        elif student_input["gpa"] < 2.3:
            points.append((False, "Previous GPA is on the lower side."))
        else:
            points.append((True, "Previous GPA is decent."))

        if student_input["assignment_score"] >= 80:
            points.append((True, "Assignment score is excellent."))
        elif student_input["assignment_score"] < 50:
            points.append((False, "Assignment score needs improvement."))
        else:
            points.append((True, "Assignment score is reasonable."))

        return points

    # Global model-level explanation
    if not feature_importances:
        return "Feature importance is not available for this model type."
    sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_two = sorted_feats[:2]
    names = " and ".join([f[0].replace("_", " ") for f in top_two])
    return f"The model relies most heavily on {names} when predicting a student's final grade."
