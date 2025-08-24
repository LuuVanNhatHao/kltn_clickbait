# src/tuning/optuna_tuner.py
from __future__ import annotations
import json
from typing import Dict, Tuple, Optional
import numpy as np

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from ..models.heads import build_base
from ..utils.metrics import tune_threshold


DEFAULT_TRIALS = 30


def _nan_to_num(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)


def _get_prob(model, X: np.ndarray) -> np.ndarray:
    """
    Lấy xác suất lớp dương (shape (N,)).
    Ưu tiên predict_proba; fallback decision_function -> sigmoid.
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        else:
            p = p.squeeze()
        return _nan_to_num(p)
    # fallback: decision function
    if hasattr(model, "decision_function"):
        z = model.decision_function(X).astype(np.float32)
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-z))
        return _nan_to_num(p)
    # last resort: use predict (0/1) and add small jitter to avoid degenerate AUC
    yhat = getattr(model, "predict")(X).astype(np.float32).reshape(-1)
    p = 0.01 + 0.98 * yhat  # 0 -> 0.01, 1 -> 0.99
    return _nan_to_num(p)


def _space(trial, name: str) -> Dict:
    name = name.lower()
    if name == "lr":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": 1000,
        }
    if name == "svm_linear":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "probability": True,  # để dùng predict_proba nếu model hỗ trợ
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "n_jobs": -1,
        }
    if name == "gbdt":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
        }
    if name == "extratrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 48),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "n_jobs": -1,
        }
    if name == "mlp":
        hl = trial.suggest_categorical(
            "hidden_layer_sizes",
            [(128,), (256,), (128, 64), (256, 128), (256, 128, 64)]
        )
        return {
            "hidden_layer_sizes": hl,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 300),
        }
    raise ValueError(f"Unsupported base learner: {name}")


def tune_base(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = DEFAULT_TRIALS,
    cv_folds: int = 5,
    random_state: int = 42,
    objective: str = "auc",  # "auc" (khuyên dùng) hoặc "f1"
) -> Tuple[Dict, float]:
    """
    Tối ưu hyper cho 1 base learner bằng CV.
    - objective="auc": tính AUC per-fold (ổn định, không phụ thuộc ngưỡng).
    - objective="f1":   tune threshold per-fold rồi tính F1 (không dùng 0.5 cố định).
    """
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna is not installed. Please install optuna to use hyperparameter tuning.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int).reshape(-1)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    objective = (objective or "auc").lower()

    def _fold_score(params: Dict) -> float:
        scores = []
        for tr, va in skf.split(X, y):
            model = build_base(name, params)   # lưu ý: build_base(name, params) theo code của bạn
            model.fit(X[tr], y[tr])

            prob = _get_prob(model, X[va])     # (N,)
            if objective == "auc":
                try:
                    score = roc_auc_score(y[va], prob)
                except Exception:
                    score = 0.0
            else:  # F1 với tune threshold per-fold
                thr, _ = tune_threshold(y[va], prob, metric="f1")
                pred = (prob >= thr).astype(int)
                score = f1_score(y[va], pred, zero_division=0)

            scores.append(float(score))
        return float(np.mean(scores)) if scores else 0.0

    def _objective(trial):
        params = _space(trial, name)
        return _fold_score(params)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def save_params(path: str, params: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
