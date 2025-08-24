
import numpy as np
from typing import List, Dict, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from .heads import build_base, predict_proba

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

class StackingOOF:
    def __init__(self, base_names: List[str], base_params: Optional[Dict[str, dict]] = None,
                 meta_learner: str = "logreg", cv_folds: int = 5, random_state: int = 42):
        self.base_names = base_names
        self.base_params = base_params or {}
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.base_models_: List[List] = []
        self.meta_model_ = None
        self.oof_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        n = len(y)
        oof = np.zeros((n, len(self.base_names)), dtype=np.float32)
        self.base_models_ = [[] for _ in self.base_names]

        for bi, name in enumerate(tqdm(self.base_names, desc="Base learners", leave=False)):
            fold_models = []
            for fold, (tr, va) in enumerate(tqdm(skf.split(X, y), total=self.cv_folds, leave=False, desc=f"{name} folds")):
                params = self.base_params.get(name, None)
                model = build_base(name, params)
                model.fit(X[tr], y[tr])
                oof[va, bi] = predict_proba(model, X[va])
                fold_models.append(model)
            self.base_models_[bi] = fold_models

        # Train meta learner
        if self.meta_learner == "lightgbm" and HAS_LGBM:
            self.meta_model_ = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.8,
                colsample_bytree=0.8, random_state=self.random_state
            )
        else:
            self.meta_model_ = LogisticRegression(max_iter=500)

        self.meta_model_.fit(oof, y)
        self.oof_train_ = oof
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Average base probs across fold-models for each base
        base_probs = []
        for fold_models in self.base_models_:
            # fold_models is a list (k fold models) of the same base
            probs = np.stack([predict_proba(m, X) for m in fold_models], axis=1).mean(axis=1)
            base_probs.append(probs)
        base_probs = np.stack(base_probs, axis=1)  # [N, B]

        if hasattr(self.meta_model_, "predict_proba"):
            return self.meta_model_.predict_proba(base_probs)[:, 1]
        scores = self.meta_model_.decision_function(base_probs)
        # scale to [0,1]
        smin, smax = scores.min(), scores.max()
        return (scores - smin) / (smax - smin + 1e-9)
