# src/tuning/optuna_tuner.py
from __future__ import annotations
import json
from typing import Dict, Tuple
import numpy as np

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.models.heads import build_base, predict_proba
from src.utils.metrics import tune_threshold

DEFAULT_TRIALS = 30

def _nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)

def _space(trial, name: str) -> Dict:
    name = name.lower()
    if name in ("lr", "logreg", "logistic_regression"):
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear","lbfgs"]),
                "max_iter": 1000, "class_weight": "balanced", "n_jobs": -1}
    if name in ("svm","svm_linear","svc"):
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "max_iter": 5000, "class_weight": "balanced"}
    if name in ("knn",):
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 101, step=2),
                "weights": trial.suggest_categorical("weights", ["uniform","distance"]),
                "metric": trial.suggest_categorical("metric", ["minkowski","cosine"])}
    if name in ("nb","gnb","gaussian_nb"):
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}
    if name in ("xgb","xgboost"):
        return {"n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                "tree_method": "gpu_hist", "predictor": "gpu_predictor", "eval_metric":"auc"}
    if name in ("mlp","mlp_torch","mlp_sklearn"):
        return {"hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes",
                    [(128,), (256,), (256,128), (256,128,64)]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
                "max_iter": trial.suggest_int("max_iter", 120, 300)}
    return {}

def tune_base(name: str, X: np.ndarray, y: np.ndarray, n_trials: int = DEFAULT_TRIALS,
             cv_folds: int = 5, random_state: int = 42, objective: str = "auc") -> Tuple[Dict, float]:
    """
    Optuna cho 1 base learner.
    - objective="auc" (khuyên dùng) hoặc "f1" (tune thr per-fold).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int).reshape(-1)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    objective = (objective or "auc").lower()

    def _fold_score(params: Dict) -> float:
        scores = []
        for tr, va in skf.split(X, y):
            clf = build_base(name, params)
            clf.fit(X[tr], y[tr])
            prob = predict_proba(clf, X[va])
            prob = _nan_to_num(prob)
            if objective == "auc":
                try:
                    scores.append(float(roc_auc_score(y[va], prob)))
                except Exception:
                    scores.append(0.0)
            else:
                thr, _ = tune_threshold(y[va], prob, metric="f1")
                pred = (prob >= thr).astype(int)
                scores.append(float(f1_score(y[va], pred, zero_division=0)))
        return float(np.mean(scores)) if scores else 0.0

    def _objective(trial):
        return _fold_score(_space(trial, name))

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)

def save_params(path: str, params: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
