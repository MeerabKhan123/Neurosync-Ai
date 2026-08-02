
import os
import glob
import joblib
import numpy as np
import pandas as pd

from feature_engineering import add_engineered_features
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS

SAVE_DIR = "saved_models"

RAW_INPUT_COLUMNS = [
    "Age", "Gender", "Occupation", "Work_Hours", "Sleep_Hours", "Screen_Time",
    "Exercise_Frequency", "Daily_Steps", "Water_Intake", "Caffeine_Intake",
    "BMI", "Heart_Rate", "BP_Systolic", "BP_Diastolic", "Stress_Score",
    "Mood_Score", "Social_Interaction_Hours", "Meditation_Minutes",
    "Weekend_Rest_Hours", "Productivity_Score", "Wellness_Score",
]


def load_artifacts(save_dir=SAVE_DIR):
    scaler = joblib.load(os.path.join(save_dir, "scaler.joblib"))
    target_encoder = joblib.load(os.path.join(save_dir, "target_encoder.joblib"))
    selected_features = joblib.load(os.path.join(save_dir, "selected_features.joblib"))
    return scaler, target_encoder, selected_features


def list_available_models(save_dir=SAVE_DIR):
    """Return {display_name: (path, kind)} for every trained model found on disk."""
    models = {}
    for path in glob.glob(os.path.join(save_dir, "*.joblib")):
        name = os.path.basename(path).replace(".joblib", "")
        if name in ("scaler", "target_encoder", "selected_features"):
            continue
        models[name.replace("_", " ")] = (path, "sklearn")
    for path in glob.glob(os.path.join(save_dir, "*.keras")):
        name = os.path.basename(path).replace(".keras", "")
        models[name.replace("_", " ")] = (path, "keras")
    return models


def load_model(path, kind):
    if kind == "keras":
        from tensorflow import keras
        return keras.models.load_model(path)
    return joblib.load(path)


def transform_raw_dataframe(raw_df, scaler, selected_features):
    """raw_df: DataFrame with RAW_INPUT_COLUMNS (unscaled, un-encoded).
    Returns a DataFrame ready to feed directly into any trained model,
    with columns matching `selected_features` in the correct order.
    """
    df = raw_df.copy()

    missing_cols = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required input columns: {missing_cols}")

    # 1. impute any missing numeric values with column median (fallback: 0)
    numeric_raw = [c for c in RAW_INPUT_COLUMNS if c not in CATEGORICAL_COLS]
    for col in numeric_raw:
        if df[col].isna().any():
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val if not np.isnan(fill_val) else 0)

    # 2. feature engineering (same function used at training time)
    df = add_engineered_features(df)

    # 3. one-hot encode categoricals (drop_first=False, same as training)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # 4. scale numeric columns with the fitted scaler (exact column order)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # 5. align to the exact selected-feature set/order used in training
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0.0  # e.g. a dummy category not present in this batch

    return df[selected_features]


def predict(raw_df, model, model_kind, scaler, target_encoder, selected_features):
    X = transform_raw_dataframe(raw_df, scaler, selected_features)

    if model_kind == "keras":
        proba = model.predict(X.to_numpy(dtype="float32"), verbose=0)
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        proba = None

    if model_kind == "keras":
        pred_idx = np.argmax(proba, axis=1)
    else:
        pred_idx = model.predict(X)

    labels = target_encoder.inverse_transform(pred_idx)

    result = pd.DataFrame({"Burnout_Risk_Prediction": labels})
    if proba is not None:
        for i, cls in enumerate(target_encoder.classes_):
            result[f"Probability_{cls}"] = proba[:, i]

    return result


if __name__ == "__main__":
    # quick smoke test with one manual row
    scaler, target_encoder, selected_features = load_artifacts()
    sample = pd.DataFrame([{
        "Age": 29, "Gender": "Female", "Occupation": "IT/Software",
        "Work_Hours": 10.5, "Sleep_Hours": 5.2, "Screen_Time": 9.0,
        "Exercise_Frequency": 1, "Daily_Steps": 3200, "Water_Intake": 1.2,
        "Caffeine_Intake": 5, "BMI": 24.5, "Heart_Rate": 88,
        "BP_Systolic": 122, "BP_Diastolic": 80, "Stress_Score": 78,
        "Mood_Score": 38, "Social_Interaction_Hours": 1.5,
        "Meditation_Minutes": 0, "Weekend_Rest_Hours": 6.0,
        "Productivity_Score": 42, "Wellness_Score": 35,
    }])

    models = list_available_models()
    print(f"Available models: {list(models.keys())}")
    name = list(models.keys())[0]
    path, kind = models[name]
    model = load_model(path, kind)

    result = predict(sample, model, kind, scaler, target_encoder, selected_features)
    print(f"\nPrediction using {name}:")
    print(result.to_string(index=False))
