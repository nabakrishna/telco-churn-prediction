
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import optuna
import mlflow
import mlflow.sklearn

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
COLUMN_INFO_PATH = os.path.join(MODELS_DIR, "column_info.pkl")
MODEL_COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.pkl")
RAW_COLUMNS_PATH = os.path.join(MODELS_DIR, "raw_columns.pkl")


def save_artifact(artifact, path):
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Class 0: {(y_resampled == 0).sum()}, Class 1: {(y_resampled == 1).sum()}")
    return X_resampled, y_resampled


def tune_random_forest(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print(f"Best RF params: {study.best_params}")
    best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    return best_model, study.best_params


def compare_models(X_train, y_train, X_test, y_test, best_rf_params):
    candidates = {
        "RandomForest (Tuned)": RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    comparison_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        comparison_results[model_name] = {
            "model": model,
            "test_auc": roc_auc_score(y_test, y_prob),
            "test_f1": f1_score(y_test, y_pred),
            "cv_auc": cv_auc,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        print(f"{model_name} — Test AUC: {comparison_results[model_name]['test_auc']:.4f}, F1: {comparison_results[model_name]['test_f1']:.4f}")

    return comparison_results


def find_optimal_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.2, 0.8, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_test, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    print(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== Loading & Preprocessing Data ===")
    X, y, preprocessor, feature_names, categorical_cols, numeric_cols, binary_cols, X_raw = preprocess_pipeline(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n=== Applying SMOTE ===")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    print("\n=== Tuning Random Forest with Optuna (30 trials) ===")
    best_rf_model, best_rf_params = tune_random_forest(X_train_resampled, y_train_resampled)

    print("\n=== Comparing Models ===")
    comparison_results = compare_models(X_train_resampled, y_train_resampled, X_test, y_test, best_rf_params)

    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]["test_auc"])
    best_model = comparison_results[best_model_name]["model"]
    print(f"\nBest Model: {best_model_name}")

    print("\n=== Optimizing Classification Threshold ===")
    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test)

    print("\n=== Final Evaluation ===")
    y_prob_final = best_model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_prob_final >= optimal_threshold).astype(int)
    print(classification_report(y_test, y_pred_final, target_names=["No Churn", "Churn"]))

    column_info = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
    }

    raw_columns = {
        "categorical_options": {col: sorted(X_raw[col].unique().tolist()) for col in categorical_cols},
        "numeric_ranges": {col: {"min": float(X_raw[col].min()), "max": float(X_raw[col].max()), "mean": float(X_raw[col].mean())} for col in numeric_cols},
    }

    comparison_summary = {
        name: {k: v for k, v in res.items() if k != "model"}
        for name, res in comparison_results.items()
    }

    save_artifact(best_model, MODEL_PATH)
    save_artifact(preprocessor, PREPROCESSOR_PATH)
    save_artifact(feature_names, FEATURE_NAMES_PATH)
    save_artifact(column_info, COLUMN_INFO_PATH)
    save_artifact(comparison_summary, MODEL_COMPARISON_PATH)
    save_artifact(raw_columns, RAW_COLUMNS_PATH)
    save_artifact({"threshold": optimal_threshold, "best_model_name": best_model_name},
                  os.path.join(MODELS_DIR, "meta.pkl"))

    print(f"\nAll artifacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train_and_evaluate()
