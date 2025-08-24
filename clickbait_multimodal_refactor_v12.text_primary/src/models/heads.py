# src/models/heads.py
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

# ===== Optional GPU libs =====
_HAS_XGB = False
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    pass

_HAS_CUML = False
try:
    # RAPIDS cuML (nếu môi trường có)
    from cuml.svm import SVC as cuSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.naive_bayes import GaussianNB as cuGNB
    from cuml.linear_model import LogisticRegression as cuLR
    _HAS_CUML = True
except Exception:
    pass

# ===== CPU (sklearn) =====
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

# ===== PyTorch MLP (GPU) =====
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    pass


def _std(with_mean: bool = False) -> StandardScaler:
    # with_mean=False để an toàn với vector thưa / concat khác thang đo
    return StandardScaler(with_mean=with_mean)


class TorchMLPWrapper:
    """
    MLP PyTorch đơn giản để chạy GPU. API tương tự sklearn: fit/predict_proba/predict.
    """
    def __init__(self, input_dim: int, hidden=(256, 128), dropout=0.1, lr=1e-3,
                 epochs=20, batch_size=256, seed=42):
        assert _HAS_TORCH, "PyTorch not installed"
        self.input_dim = int(input_dim)
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers, prev = [], self.input_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self._mean = None
        self._std = None  # standardize đơn giản

    def _fit_scaler(self, X):
        Xc = X.astype(np.float32)
        self._mean = Xc.mean(axis=0, keepdims=True)
        self._std = Xc.std(axis=0, keepdims=True) + 1e-8

    def _transform(self, X):
        Xc = X.astype(np.float32)
        if self._mean is not None:
            Xc = (Xc - self._mean) / self._std
        return Xc

    def fit(self, X, y):
        if self._mean is None:
            self._fit_scaler(X)
        X = self._transform(X)
        y = y.astype(np.float32).reshape(-1, 1)

        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device); yb = yb.to(self.device)
                self.opt.zero_grad()
                logits = self.net(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        X = self._transform(X)
        tens = torch.from_numpy(X).to(self.device)
        logits = self.net(tens).squeeze(1)
        prob1 = torch.sigmoid(logits).float().cpu().numpy()
        prob0 = 1.0 - prob1
        return np.stack([prob0, prob1], axis=1)

    @torch.no_grad()
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_base(name: str, params: Optional[Dict] = None):
    """
    name ∈ {"knn","svm","xgb","nb","lr","mlp"} — tự dùng GPU khi có (XGB, MLP Torch, cuML).
    Lưu ý: API gọi là build_base(name, params) (params là dict).
    """
    name = (name or "").lower()
    params = params or {}

    # ---- XGBoost (GPU nếu có) ----
    if name in ("xgb", "xgboost"):
        if not _HAS_XGB:
            raise RuntimeError("xgboost chưa được cài. `pip install xgboost`")
        return XGBClassifier(
            tree_method=params.get("tree_method", "gpu_hist"),
            predictor=params.get("predictor", "gpu_predictor"),
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            eval_metric=params.get("eval_metric", "auc"),
            n_jobs=params.get("n_jobs", -1),
        )

    # ---- MLP (GPU qua PyTorch) ----
    if name in ("mlp", "mlp_torch"):
        if _HAS_TORCH:
            class _Factory:
                def __init__(self, params):
                    self.params = params; self.impl = None
                def fit(self, X, y):
                    if self.impl is None:
                        self.impl = TorchMLPWrapper(
                            input_dim=X.shape[1],
                            hidden=self.params.get("hidden", (256, 128)),
                            dropout=self.params.get("dropout", 0.1),
                            lr=self.params.get("lr", 1e-3),
                            epochs=self.params.get("epochs", 20),
                            batch_size=self.params.get("batch_size", 256),
                            seed=self.params.get("seed", 42),
                        )
                    return self.impl.fit(X, y)
                def predict_proba(self, X): return self.impl.predict_proba(X)
                def predict(self, X): return self.impl.predict(X)
            return _Factory(params)
        # fallback: sklearn MLP (CPU)
        mlp = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (256, 128)),
            alpha=params.get("alpha", 1e-4),
            learning_rate_init=params.get("learning_rate_init", 1e-3),
            max_iter=params.get("max_iter", 200),
            batch_size=params.get("batch_size", "auto"),
            early_stopping=params.get("early_stopping", True),
            n_iter_no_change=params.get("n_iter_no_change", 10),
            random_state=params.get("random_state", 42),
        )
        return make_pipeline(_std(False), mlp)

    # ---- SVM ----
    if name in ("svm", "svm_linear", "svc"):
        if _HAS_CUML:
            return cuSVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "linear"),  # hoặc 'rbf'
                probability=True,
            )
        # CPU fallback: LinearSVC + calibrate (ổn định & nhanh)
        base = LinearSVC(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            dual=params.get("dual", "auto"),
            max_iter=params.get("max_iter", 5000),
        )
        calibrated = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)
        return make_pipeline(_std(False), calibrated)

    # ---- KNN ----
    if name in ("knn",):
        if _HAS_CUML:
            return cuKNN(
                n_neighbors=params.get("n_neighbors", 31),
                weights=params.get("weights", "distance"),
                metric=params.get("metric", "minkowski"),
            )
        knn = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 31),
            weights=params.get("weights", "distance"),
            metric=params.get("metric", "minkowski"),
            n_jobs=params.get("n_jobs", None),
        )
        return make_pipeline(_std(False), knn)

    # ---- Naive Bayes ----
    if name in ("nb", "gnb", "gaussian_nb"):
        if _HAS_CUML:
            return cuGNB(var_smoothing=params.get("var_smoothing", 1e-9))
        return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))

    # ---- Logistic Regression ----
    if name in ("lr", "logreg", "logistic_regression"):
        if _HAS_CUML:
            return cuLR(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                fit_intercept=True,
                tol=params.get("tol", 1e-4),
            )
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            solver=params.get("solver", "liblinear"),
            max_iter=params.get("max_iter", 1000),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=params.get("n_jobs", -1),
        )
        return make_pipeline(_std(False), clf)

    raise ValueError(f"Unsupported base learner: {name}")


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Trả về p(y=1) dạng (N,). Tương thích mọi model/pipeline/wrapper.
    """
    # pipeline sklearn?
    clf = model[-1] if hasattr(model, "steps") else model

    # PyTorch wrapper factory
    if hasattr(model, "impl") or isinstance(clf, TorchMLPWrapper):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1].astype(np.float32)
        return p.squeeze().astype(np.float32)

    # predict_proba
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2:
            if p.shape[1] == 2:
                return p[:, 1].astype(np.float32)
            return p.squeeze().astype(np.float32)
        return p.astype(np.float32)

    # decision_function → sigmoid
    if hasattr(model, "decision_function"):
        z = model.decision_function(X).astype(np.float32).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-z))
        return p.astype(np.float32)

    # fallback: hard label → pseudo-prob
    yhat = model.predict(X).astype(np.float32).reshape(-1)
    return (0.01 + 0.98 * yhat).astype(np.float32)
