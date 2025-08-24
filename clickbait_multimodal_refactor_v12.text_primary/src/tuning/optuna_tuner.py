
from __future__ import annotations
import json
from typing import Dict, Tuple
import numpy as np

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

from ..models.heads import build_base

DEFAULT_TRIALS = 30

def _space(trial, name: str) -> Dict:
    name = name.lower()
    if name == "lr":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": 400
        }
    if name == "svm_linear":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True)
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if name == "gbdt":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }
    if name == "extratrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 48),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if name == "mlp":
        hl = trial.suggest_categorical("hidden_layer_sizes", [(128,), (256,), (128,64), (256,128), (256,128,64)])
        return {
            "hidden_layer_sizes": hl,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "max_iter": 200
        }
    raise ValueError(f"Unsupported base learner: {name}")

def tune_base(name: str, X: np.ndarray, y: np.ndarray, n_trials: int = DEFAULT_TRIALS, cv_folds: int = 5, random_state: int = 42) -> Tuple[Dict, float]:
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna is not installed. Please install optuna to use hyperparameter tuning.")
    name = name.lower()
    f1 = make_scorer(f1_score)

    def objective(trial):
        params = _space(trial, name)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        model = build_base(name, params)
        scores = cross_val_score(model, X, y, cv=skf, scoring=f1, n_jobs=None)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def save_params(path: str, params: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
