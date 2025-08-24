
import numpy as np
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

def build_base(name: str, params: dict | None = None) -> Any:
    name = name.lower()
    params = params or {}
    if name == "lr":
        return LogisticRegression(
            max_iter=params.get("max_iter", 400),
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs")
        )
    if name == "svm_linear":
        base = LinearSVC(C=params.get("C", 1.0))
        return CalibratedClassifierCV(base, method="sigmoid", cv=5)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            n_jobs=-1,
            random_state=42
        )
    if name == "gbdt":
        return GradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (256,)),
            alpha=params.get("alpha", 1e-4),
            learning_rate_init=params.get("learning_rate_init", 1e-3),
            max_iter=params.get("max_iter", 200)
        )
    raise ValueError(f"Unknown base learner: {name}")

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    # Return probability of positive class
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X).astype(np.float32)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    # fallback
    preds = model.predict(X)
    return preds.astype(np.float32)
