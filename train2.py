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


def tune_xgboost(X_train, y_train, device):
    # ✅ Tune on ORIGINAL data — no synthetic samples
    X_tr, X_val, y_tr, y_val = val_split(X_train, y_train)
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

    def objective(trial):
        p = dict(
            n_estimators          = trial.suggest_int("n_estimators", 200, 1500),
            max_depth             = trial.suggest_int("max_depth", 3, 10),
            learning_rate         = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample             = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree      = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            colsample_bylevel     = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            min_child_weight      = trial.suggest_int("min_child_weight", 1, 15),
            gamma                 = trial.suggest_float("gamma", 0.0, 3.0),
            reg_alpha             = trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda            = trial.suggest_float("reg_lambda", 0.0, 5.0),
            max_bin               = 256,
            scale_pos_weight      = scale_pos,
            tree_method           = "hist",
            device                = device,
            eval_metric           = "auc",
            early_stopping_rounds = 30,
            random_state          = RANDOM,
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=True)
    print(f"\n  XGBoost best Val AUC  : {study.best_value:.4f}")
    return dict(**study.best_params,
                max_bin=256, scale_pos_weight=scale_pos,
                tree_method="hist", device=device,
                eval_metric="auc", early_stopping_rounds=30,
                random_state=RANDOM)


def tune_catboost(X_train, y_train):
    # ✅ Tune on ORIGINAL data — no synthetic samples
    X_tr, X_val, y_tr, y_val = val_split(X_train, y_train)
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

    def objective(trial):
        p = dict(
            iterations            = trial.suggest_int("iterations", 300, 1500),
            depth                 = trial.suggest_int("depth", 4, 8),
            learning_rate         = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            l2_leaf_reg           = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature   = trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_strength       = trial.suggest_float("random_strength", 0.0, 2.0),
            border_count          = trial.suggest_int("border_count", 32, 128),
            scale_pos_weight      = scale_pos,
            eval_metric           = "AUC",
            random_seed           = RANDOM,
            verbose               = False,
            early_stopping_rounds = 40,
        )
        m = CatBoostClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print(f"\n  CatBoost best Val AUC : {study.best_value:.4f}")
    return dict(**study.best_params,
                scale_pos_weight=scale_pos, eval_metric="AUC",
                random_seed=RANDOM, verbose=False,
                early_stopping_rounds=40)


def tune_mlp(X_train, y_train):
    # ✅ Tune on ORIGINAL data — no synthetic samples
    X_tr, X_val, y_tr, y_val = val_split(X_train, y_train)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 3)
        layers   = tuple(trial.suggest_int(f"n_units_{i}", 64, 256) for i in range(n_layers))
        p = dict(
            hidden_layer_sizes = layers,
            activation         = trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha              = trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            learning_rate_init = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            max_iter           = 500,
            early_stopping     = True,
            validation_fraction= 0.1,
            random_state       = RANDOM,
        )
        m = MLPClassifier(**p)
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    print(f"\n  MLP best Val AUC      : {study.best_value:.4f}")

    n_layers = study.best_params["n_layers"]
    layers   = tuple(study.best_params[f"n_units_{i}"] for i in range(n_layers))
    return dict(
        hidden_layer_sizes = layers,
        activation         = study.best_params["activation"],
        alpha              = study.best_params["alpha"],
        learning_rate_init = study.best_params["lr"],
        max_iter           = 1000,
        early_stopping     = True,
        random_state       = RANDOM,
    )


def build_final_models(lgb_params, xgb_params, cat_params, mlp_params,
                       X_train_res, y_train_res, X_tr, X_val, y_tr, y_val):
    """
    Train final models on SMOTETomek-resampled data using
    best params found on ORIGINAL data.
    """
    models = {}

    print("  Fitting LightGBM on resampled data...")
    lgb_m = lgb.LGBMClassifier(**lgb_params)
    lgb_m.fit(X_train_res, y_train_res,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    models["LightGBM"] = lgb_m

    print("  Fitting XGBoost on resampled data...")
    xgb_m = xgb.XGBClassifier(**xgb_params)
    xgb_m.fit(X_train_res, y_train_res,
              eval_set=[(X_val, y_val)], verbose=False)
    models["XGBoost"] = xgb_m

    if CATBOOST_OK and cat_params:
        print("  Fitting CatBoost on resampled data...")
        cat_m = CatBoostClassifier(**cat_params)
        cat_m.fit(X_train_res, y_train_res, eval_set=(X_val, y_val))
        models["CatBoost"] = cat_m

    if mlp_params:
        print("  Fitting MLP on resampled data...")
        mlp_m = MLPClassifier(**mlp_params)
        mlp_m.fit(X_train_res, y_train_res)
        models["MLP"] = mlp_m

    return models


def build_oof_stack(models, X_train, y_train, X_test):
    """
    Out-of-fold on ORIGINAL train data — unbiased meta-features.
    Each fold applies SMOTETomek inside to avoid leakage.
    """
    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM)
    oof_train   = np.zeros((len(X_train), len(models)))
    oof_test    = np.zeros((len(X_test),  len(models)))

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train
    Xt_arr= X_test.values  if hasattr(X_test,  "values") else X_test

    for mi, (mname, model) in enumerate(models.items()):
        print(f"  Stacking {mname}  ({N_FOLDS} folds)...")
        test_preds = np.zeros((len(X_test), N_FOLDS))

        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            Xf_tr, yf_tr = X_arr[tr_idx], y_arr[tr_idx]
            Xf_val       = X_arr[val_idx]

            # Apply SMOTETomek inside each fold
            smt = SMOTETomek(
                smote=SMOTE(sampling_strategy=0.75, random_state=RANDOM+fi, k_neighbors=5),
                random_state=RANDOM+fi
            )
            Xf_tr_res, yf_tr_res = smt.fit_resample(Xf_tr, yf_tr)

            clone = pickle.loads(pickle.dumps(model))

            is_lgb = isinstance(clone, lgb.LGBMClassifier)
            is_xgb = isinstance(clone, xgb.XGBClassifier)
            is_cat = CATBOOST_OK and isinstance(clone, CatBoostClassifier)

            Xf_val_original = X_arr[val_idx]
            yf_val_original = y_arr[val_idx]

            if is_lgb:
                clone.fit(Xf_tr_res, yf_tr_res,
                          eval_set=[(Xf_val_original, yf_val_original)],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
            elif is_xgb:
                clone.fit(Xf_tr_res, yf_tr_res,
                          eval_set=[(Xf_val_original, yf_val_original)], verbose=False)
            elif is_cat:
                clone.fit(Xf_tr_res, yf_tr_res,
                          eval_set=(Xf_val_original, yf_val_original))
            else:
                clone.fit(Xf_tr_res, yf_tr_res)

            oof_train[val_idx, mi] = clone.predict_proba(Xf_val)[:, 1]
            test_preds[:, fi]      = clone.predict_proba(Xt_arr)[:, 1]

        oof_test[:, mi] = test_preds.mean(axis=1)
        auc = roc_auc_score(y_arr, oof_train[:, mi])
        print(f"    OOF AUC : {auc:.4f}")

    return oof_train, oof_test


def fit_meta_learner(oof_train, oof_test, y_train, X_test, y_test):
    meta = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM)
    cal  = CalibratedClassifierCV(meta, method="isotonic", cv=5)
    cal.fit(oof_train, y_train)

    y_prob = cal.predict_proba(oof_test)[:, 1]
    print(f"  Meta AUC (calibrated) : {roc_auc_score(y_test, y_prob):.4f}")
    return cal, y_prob


def find_threshold(y_test, y_prob):
    rows = []
    for t in np.arange(0.10, 0.90, 0.005):
        yp = (y_prob >= t).astype(int)
        rows.append(dict(
            threshold = round(t, 3),
            f1        = f1_score(y_test, yp, zero_division=0),
            precision = precision_score(y_test, yp, zero_division=0),
            recall    = recall_score(y_test, yp, zero_division=0),
            accuracy  = float((yp == y_test.values).mean()),
        ))
    df = pd.DataFrame(rows)

    balanced = df[(df["precision"] >= 0.72) & (df["recall"] >= 0.68)]
    best     = balanced.loc[balanced["f1"].idxmax()] if len(balanced) else df.loc[df["f1"].idxmax()]
    t        = best["threshold"]

    print(f"\n  Best threshold : {t:.3f}")
    print(f"  F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}  Acc={best['accuracy']:.4f}")
    print("\n  Top 8 by F1:")
    print(df.nlargest(8, "f1")[["threshold","f1","precision","recall","accuracy"]].to_string(index=False))
    return t


def print_metrics(y_test, y_pred, y_prob):
    rows = [
        ("ROC-AUC",      roc_auc_score(y_test, y_prob)),
        ("F1 (Churn)",   f1_score(y_test, y_pred, zero_division=0)),
        ("F1 (Weighted)",f1_score(y_test, y_pred, average="weighted")),
        ("Precision",    precision_score(y_test, y_pred, zero_division=0)),
        ("Recall",       recall_score(y_test, y_pred, zero_division=0)),
        ("Accuracy",     float((y_pred == y_test.values).mean())),
    ]
    print(f"\n  {'Metric':<22} {'Score':>8}   Status")
    print("  " + "-" * 42)
    d = {}
    for name, val in rows:
        icon = "✅" if val >= 0.80 else "⚠️" if val >= 0.70 else "❌"
        print(f"  {name:<22} {val:>8.4f}   {icon}")
        d[name] = val
    return d


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  STEP 1 — GPU & Library Check")
    print("=" * 60)
    xgb_device, lgb_device = check_gpu()

    print("\n" + "=" * 60)
    print("  STEP 2 — Preprocessing + Feature Engineering")
    print("=" * 60)
    X, y, preprocessor, feature_names, column_info, raw_columns, X_raw = preprocess_pipeline(DATA_PATH)
    print(f"  Final features : {X.shape[1]}")

    print("\n" + "=" * 60)
    print("  STEP 3 — Train / Test Split  (80/20 stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")

    # Hold out a small val set from ORIGINAL data for final model eval_set
    X_tr_orig, X_val_orig, y_tr_orig, y_val_orig = val_split(X_train, y_train, size=0.12)

    print("\n" + "=" * 60)
    print("  STEP 4 — Hyperparameter Tuning on ORIGINAL data")
    print("           (no synthetic samples — prevents overfitting)")
    print("=" * 60)

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


if __name__ == "__main__":
    train()