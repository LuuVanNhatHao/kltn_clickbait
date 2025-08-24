# src/tuning/optuna_tuner.py
from __future__ import annotations
import json
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

import optuna

from src.models.heads import build_base, predict_proba
from src.utils.metrics import tune_threshold


def _nan_to_num(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)


def _space(trial: optuna.Trial, name: str) -> Dict:
    """
    Search space cho từng base learner.
    Bổ sung/điều chỉnh theo heads bạn có.
    """
    name = name.lower()
    if name in ("lr", "logreg", "logistic_regression"):
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": 2000,
            "n_jobs": -1,
        }
    if name in ("svm_linear", "linear_svm"):
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "probability": True,
        }
    if name in ("rf", "random_forest"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "n_jobs": -1,
        }
    if name in ("gbdt", "gradient_boosting"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
        }
    if name in ("extratrees", "extra_trees"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "n_jobs": -1,
        }
    if name in ("mlp", "mlp_head"):
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 1024, step=128),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "epochs": trial.suggest_int("epochs", 5, 30),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    # fallback: không có space đặc thù → không tune gì
    return {}


def tune_base(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
    objective: str = "auc",
) -> Tuple[Dict, float]:
    """
    Tối ưu hyper cho 1 base learner bằng CV.
    - objective: "auc" (khuyên dùng) hoặc "f1" (có tune threshold per-fold).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int).reshape(-1)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def _fold_score(params: Dict) -> float:
        scores = []
        for tr, va in skf.split(X, y):
            clf = build_base(name, **params)
            clf.fit(X[tr], y[tr])

            prob = predict_proba(clf, X[va])  # có thể (N,), (N,1) hoặc (N,2)
            if prob.ndim == 2:
                if prob.shape[1] == 2:
                    prob = prob[:, 1]
                else:
                    prob = prob.squeeze(-1)
            prob = _nan_to_num(prob)

            if objective == "auc":
                # AUC không phụ thuộc ngưỡng
                try:
                    score = roc_auc_score(y[va], prob)
                except Exception:
                    score = 0.0
            else:
                # F1 cần tune threshold per-fold
                thr, _ = tune_threshold(y[va], prob, metric="f1")
                pred = (prob >= thr).astype(int)
                score = f1_score(y[va], pred, zero_division=0)

            scores.append(float(score))
        return float(np.mean(scores)) if scores else 0.0

    def _objective(trial: optuna.Trial) -> float:
        params = _space(trial, name)
        return _fold_score(params)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials)

    return study.best_params, float(study.best_value)


def save_params(path: str, params: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
