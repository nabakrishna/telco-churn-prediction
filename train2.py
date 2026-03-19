import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_pipeline

from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.linear_model     import LogisticRegression
from sklearn.neural_network   import MLPClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling   import SMOTE
from imblearn.combine         import SMOTETomek
import xgboost  as xgb
import lightgbm as lgb
import optuna

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
N_FOLDS    = 5
RANDOM     = 42


def save(obj, name):
    with open(os.path.join(MODELS_DIR, name), "wb") as f:
        pickle.dump(obj, f)


def check_gpu():
    xgb_device = "cpu"
    lgb_device  = "cpu"
    try:
        t = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=5)
        t.fit(np.random.rand(100, 5), np.random.randint(0, 2, 100))
        xgb_device = "cuda"
        print("  XGBoost  GPU ✅")
    except Exception:
        print("  XGBoost  GPU ❌")
    try:
        t = lgb.LGBMClassifier(device="gpu", n_estimators=5, verbose=-1)
        t.fit(np.random.rand(100, 5), np.random.randint(0, 2, 100))
        lgb_device = "gpu"
        print("  LightGBM GPU ✅")
    except Exception:
        print("  LightGBM GPU ❌")
    print(f"  CatBoost     {'✅' if CATBOOST_OK else '❌ pip install catboost'}")
    return xgb_device, lgb_device


def val_split(X, y, size=0.15):
    return train_test_split(X, y, test_size=size, stratify=y, random_state=RANDOM)


def apply_smotetomek(X, y):
    smt = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.75, random_state=RANDOM, k_neighbors=5),
        random_state=RANDOM
    )
    X_res, y_res = smt.fit_resample(X, y)
    print(f"  No Churn: {(y_res==0).sum():,}  Churn: {(y_res==1).sum():,}")
    return X_res, y_res


def tune_lightgbm(X_train, y_train, device):
    # ✅ Tune on ORIGINAL data — no synthetic samples
    X_tr, X_val, y_tr, y_val = val_split(X_train, y_train)
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 2000),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            num_leaves        = trial.suggest_int("num_leaves", 20, 150),
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq    = trial.suggest_int("subsample_freq", 1, 5),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 5.0),
            scale_pos_weight  = scale_pos,
            device            = device,
            random_state      = RANDOM,
            verbose           = -1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=80, show_progress_bar=True)
    print(f"\n  LightGBM best Val AUC : {study.best_value:.4f}")
    return dict(**study.best_params,
                scale_pos_weight=scale_pos, device=device,
                random_state=RANDOM, verbose=-1)


import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_pipeline

from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.linear_model     import LogisticRegression
from sklearn.neural_network   import MLPClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling   import SMOTE
from imblearn.combine         import SMOTETomek
import xgboost  as xgb
import lightgbm as lgb
import optuna

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
N_FOLDS    = 5
RANDOM     = 42


def save(obj, name):
    with open(os.path.join(MODELS_DIR, name), "wb") as f:
        pickle.dump(obj, f)


def check_gpu():
    xgb_device = "cpu"
    lgb_device  = "cpu"
    try:
        t = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=5)
        t.fit(np.random.rand(100, 5), np.random.randint(0, 2, 100))
        xgb_device = "cuda"
        print("  XGBoost  GPU ✅")
    except Exception:
        print("  XGBoost  GPU ❌")
    try:
        t = lgb.LGBMClassifier(device="gpu", n_estimators=5, verbose=-1)
        t.fit(np.random.rand(100, 5), np.random.randint(0, 2, 100))
        lgb_device = "gpu"
        print("  LightGBM GPU ✅")
    except Exception:
        print("  LightGBM GPU ❌")
    print(f"  CatBoost     {'✅' if CATBOOST_OK else '❌ pip install catboost'}")
    return xgb_device, lgb_device


def val_split(X, y, size=0.15):
    return train_test_split(X, y, test_size=size, stratify=y, random_state=RANDOM)



    print("\n  Tuning LightGBM  (80 trials)...")
    lgb_params = tune_lightgbm(X_train, y_train, lgb_device)

    print("\n  Tuning XGBoost  (60 trials)...")
    xgb_params = tune_xgboost(X_train, y_train, xgb_device)

    cat_params = None
    if CATBOOST_OK:
        print("\n  Tuning CatBoost  (50 trials)...")
        cat_params = tune_catboost(X_train, y_train)

    print("\n  Tuning MLP  (30 trials)...")
    mlp_params = tune_mlp(X_train, y_train)

    print("\n" + "=" * 60)
    print("  STEP 5 — SMOTETomek on Train Set")
    print("           (applied AFTER tuning — no leakage)")
    print("=" * 60)
    X_train_res, y_train_res = apply_smotetomek(X_tr_orig, y_tr_orig)

    print("\n" + "=" * 60)
    print("  STEP 6 — Fitting Final Base Models on Resampled Data")
    print("=" * 60)
    models = build_final_models(
        lgb_params, xgb_params, cat_params, mlp_params,
        X_train_res, y_train_res,
        X_tr_orig, X_val_orig, y_tr_orig, y_val_orig
    )

    print("\n" + "=" * 60)
    print(f"  STEP 7 — OOF Stacking ({N_FOLDS} folds, SMOTETomek inside each fold)")
    print("=" * 60)
    oof_train, oof_test = build_oof_stack(models, X_train, y_train, X_test)

    print("\n" + "=" * 60)
    print("  STEP 8 — Calibrated Meta-Learner")
    print("=" * 60)
    meta_model, y_prob = fit_meta_learner(oof_train, oof_test, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("  STEP 9 — Threshold Optimisation")
    print("=" * 60)
    threshold = find_threshold(y_test, y_prob)

    print("\n" + "=" * 60)
    print("  STEP 10 — Final Evaluation")
    print("=" * 60)
    y_pred = (y_prob >= threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    metrics = print_metrics(y_test, y_pred, y_prob)

    print("\n  Individual Test AUC:")
    comparison = {}
    Xt = X_test.values if hasattr(X_test, "values") else X_test
    for name, model in models.items():
        prob = model.predict_proba(Xt)[:, 1]
        auc  = roc_auc_score(y_test, prob)
        f1   = f1_score(y_test, (prob >= 0.5).astype(int))
        comparison[name] = {"test_auc": auc, "test_f1": f1}
        print(f"  {name:<20} AUC={auc:.4f}  F1={f1:.4f}")

    comparison["OOF Stack"] = {
        "test_auc": metrics["ROC-AUC"],
        "test_f1" : metrics["F1 (Churn)"],
    }

    save({"base_models": models, "meta_model": meta_model}, "best_model.pkl")
    save(preprocessor,   "preprocessor.pkl")
    save(feature_names,  "feature_names.pkl")
    save(column_info,    "column_info.pkl")
    save(comparison,     "model_comparison.pkl")
    save(raw_columns,    "raw_columns.pkl")
    save({
        "threshold"       : threshold,
        "best_model_name" : "OOF Stack (LGB+XGB+CatBoost+MLP) Calibrated",
    }, "meta.pkl")

    print("\n" + "=" * 60)
    print("  All artifacts saved to  models/")
    print("=" * 60)

