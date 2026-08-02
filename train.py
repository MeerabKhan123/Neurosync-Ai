"""
NeuroSync AI - Model Training
---------------------------------
Trains and evaluates:
  Machine Learning : Logistic Regression, Decision Tree, Random Forest,
                      KNN, SVM, Gradient Boosting, XGBoost
  Deep Learning    : ANN, LSTM

Requires preprocessing.py to have been run first (needs
dataset/processed/*.csv and saved_models/selected_features.joblib).

Saves:
  saved_models/<model_name>.joblib   (ML models)
  saved_models/<model_name>.keras    (DL models)
  reports/model_comparison_table.csv
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from evaluation import evaluate_model, print_report, Timer

RANDOM_SEED = 42
SAVE_DIR = "saved_models"
REPORT_DIR = "reports"

# SVM and KNN scale poorly with 90k+ training rows; cap their training size
# so the demo trains in a reasonable time on a normal laptop. Raise/remove
# this if you have the time/hardware for the full dataset.
SLOW_MODEL_SAMPLE_SIZE = 15_000


def load_processed_data(processed_dir="dataset/processed"):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def subsample(X, y, n, seed=RANDOM_SEED):
    if len(X) <= n:
        return X, y
    idx = X.sample(n=n, random_state=seed).index
    return X.loc[idx], y.loc[idx]


def get_ml_models():
    return {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision_Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_SEED),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_SEED
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, subsample=0.8,
            learning_rate=0.15, random_state=RANDOM_SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def train_ml_models(X_train, X_test, y_train, y_test):
    results = []
    models = get_ml_models()
    os.makedirs(SAVE_DIR, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name in ("SVM", "KNN"):
            Xt, yt = subsample(X_train, y_train, SLOW_MODEL_SAMPLE_SIZE)
        else:
            Xt, yt = X_train, y_train

        with Timer() as t:
            model.fit(Xt, yt)
        train_time = t.elapsed

        with Timer() as t:
            y_pred = model.predict(X_test)
        predict_time = t.elapsed

        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = evaluate_model(name, y_test, y_pred, y_proba, train_time, predict_time)
        results.append(metrics)
        print_report(name, y_test, y_pred)

        joblib.dump(model, os.path.join(SAVE_DIR, f"{name}.joblib"))

    return results


def build_ann(input_dim, n_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(n_timesteps, n_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(n_timesteps, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_dl_models(X_train, X_test, y_train, y_test):
    from tensorflow import keras

    results = []
    n_classes = int(pd.concat([y_train, y_test]).nunique())
    X_train_np = X_train.to_numpy(dtype="float32")
    X_test_np = X_test.to_numpy(dtype="float32")
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # --- ANN ---
    print("\nTraining ANN...")
    ann = build_ann(X_train_np.shape[1], n_classes)
    with Timer() as t:
        ann.fit(
            X_train_np, y_train_np, validation_split=0.15,
            epochs=50, batch_size=256, callbacks=[early_stop], verbose=0,
        )
    train_time = t.elapsed

    with Timer() as t:
        ann_proba = ann.predict(X_test_np, verbose=0)
    predict_time = t.elapsed
    ann_pred = np.argmax(ann_proba, axis=1)

    metrics = evaluate_model("ANN", y_test_np, ann_pred, ann_proba, train_time, predict_time)
    results.append(metrics)
    print_report("ANN", y_test_np, ann_pred)
    ann.save(os.path.join(SAVE_DIR, "ANN.keras"))

    # --- LSTM ---
    # Reshape tabular features into a pseudo-sequence: (samples, n_features, 1)
    print("\nTraining LSTM...")
    X_train_seq = X_train_np.reshape(X_train_np.shape[0], X_train_np.shape[1], 1)
    X_test_seq = X_test_np.reshape(X_test_np.shape[0], X_test_np.shape[1], 1)

    lstm = build_lstm(X_train_np.shape[1], n_classes)
    with Timer() as t:
        lstm.fit(
            X_train_seq, y_train_np, validation_split=0.15,
            epochs=30, batch_size=256, callbacks=[early_stop], verbose=0,
        )
    train_time = t.elapsed

    with Timer() as t:
        lstm_proba = lstm.predict(X_test_seq, verbose=0)
    predict_time = t.elapsed
    lstm_pred = np.argmax(lstm_proba, axis=1)

    metrics = evaluate_model("LSTM", y_test_np, lstm_pred, lstm_proba, train_time, predict_time)
    results.append(metrics)
    print_report("LSTM", y_test_np, lstm_pred)
    lstm.save(os.path.join(SAVE_DIR, "LSTM.keras"))

    return results


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading processed train/test splits...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    ml_results = train_ml_models(X_train, X_test, y_train, y_test)
    dl_results = train_dl_models(X_train, X_test, y_train, y_test)

    comparison = pd.DataFrame(ml_results + dl_results).sort_values(
        "Accuracy", ascending=False
    ).reset_index(drop=True)

    comparison_path = os.path.join(REPORT_DIR, "model_comparison_table.csv")
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print(f"\nBest model: {comparison.iloc[0]['Model']} "
          f"(Accuracy: {comparison.iloc[0]['Accuracy']:.4f})")
    print(f"\nSaved comparison table to {comparison_path}")


if __name__ == "__main__":
    main()
