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


