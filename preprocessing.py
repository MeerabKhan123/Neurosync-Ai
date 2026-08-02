"""
NeuroSync AI - Preprocessing Pipeline
--------------------------------------
Handles: missing values, duplicate removal, outlier capping, encoding,
feature scaling, correlation analysis, feature selection, and
train/test split. Saves fitted encoders/scaler to saved_models/ so the
same transforms can be replayed at prediction time.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from feature_engineering import add_engineered_features, ENGINEERED_COLS

RANDOM_SEED = 42

BASE_NUMERIC_COLS = [
    "Age", "Work_Hours", "Sleep_Hours", "Screen_Time", "Exercise_Frequency",
    "Daily_Steps", "Water_Intake", "Caffeine_Intake", "BMI", "Heart_Rate",
    "BP_Systolic", "BP_Diastolic", "Stress_Score", "Mood_Score",
    "Social_Interaction_Hours", "Meditation_Minutes", "Weekend_Rest_Hours",
    "Productivity_Score", "Wellness_Score",
]
NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_COLS
CATEGORICAL_COLS = ["Gender", "Occupation"]
TARGET_COL = "Burnout_Risk"


def load_raw(path="dataset/neurosync_dataset.csv"):
    return pd.read_csv(path)


def handle_missing_values(df):
    """Impute numeric columns with median (robust to outliers)."""
    df = df.copy()
    for col in BASE_NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows.")
    return df


def cap_outliers_iqr(df, cols, factor=1.5):
    """Cap outliers using the IQR rule instead of dropping rows,
    to preserve dataset size for model training."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def encode_features(df, save_dir="saved_models"):
    """One-hot encode nominal categoricals, label-encode the target."""
    os.makedirs(save_dir, exist_ok=True)
    df = df.copy()

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    target_encoder = LabelEncoder()
    df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL])
    joblib.dump(target_encoder, os.path.join(save_dir, "target_encoder.joblib"))
    print(f"Target classes: {list(target_encoder.classes_)} -> {list(range(len(target_encoder.classes_)))}")

    return df, target_encoder


def scale_features(df, feature_cols, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    scaler = StandardScaler()
    df = df.copy()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    return df, scaler


def correlation_analysis(df, target_col=TARGET_COL, top_n=15):
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print(f"\nTop {top_n} features correlated with {target_col}:")
    print(corr.head(top_n).round(3))
    return corr


def select_features(corr_series, min_abs_corr=0.02):
    """Drop features with negligible correlation with the target."""
    selected = corr_series[corr_series.abs() >= min_abs_corr].index.tolist()
    dropped = corr_series[corr_series.abs() < min_abs_corr].index.tolist()
    if dropped:
        print(f"Dropping low-signal features: {dropped}")
    return selected


def run_pipeline(raw_path="dataset/neurosync_dataset.csv", save_dir="saved_models", test_size=0.2):
    print("Step 1/8: Loading raw dataset...")
    df = load_raw(raw_path)
    print(f"  Raw shape: {df.shape}")

    print("Step 2/8: Handling missing values...")
    df = handle_missing_values(df)

    print("Step 3/8: Removing duplicates...")
    df = remove_duplicates(df)

    print("Step 4/8: Engineering derived features...")
    df = add_engineered_features(df)

    print("Step 5/8: Capping outliers (IQR method)...")
    df = cap_outliers_iqr(df, NUMERIC_COLS)

    print("Step 6/8: Encoding categorical features + target...")
    df, target_encoder = encode_features(df, save_dir)

    feature_cols_for_scaling = NUMERIC_COLS  # scale only true numeric features

    print("Step 7/8: Scaling numeric features...")
    df, scaler = scale_features(df, feature_cols_for_scaling, save_dir)

    print("Step 8/8: Correlation analysis + feature selection...")
    corr = correlation_analysis(df)
    selected_features = select_features(corr)

    X = df[selected_features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True).round(3)}")

    # persist processed splits + selected feature list for the training phase
    os.makedirs("dataset/processed", exist_ok=True)
    X_train.to_csv("dataset/processed/X_train.csv", index=False)
    X_test.to_csv("dataset/processed/X_test.csv", index=False)
    y_train.to_csv("dataset/processed/y_train.csv", index=False)
    y_test.to_csv("dataset/processed/y_test.csv", index=False)
    joblib.dump(selected_features, os.path.join(save_dir, "selected_features.joblib"))

    print("\nProcessed splits saved to dataset/processed/")
    return X_train, X_test, y_train, y_test, selected_features


if __name__ == "__main__":
    run_pipeline()
