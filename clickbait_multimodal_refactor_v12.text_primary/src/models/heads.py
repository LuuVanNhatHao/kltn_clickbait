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
    # RAPIDS cuML (nếu đã cài): SVM/KNN/NB/LR trên GPU
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

# ===== PyTorch cho MLP GPU =====
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    pass


def _std(with_mean: bool = False) -> StandardScaler:
    return StandardScaler(with_mean=with_mean)


# ---------- PyTorch MLP (GPU) ----------
class TorchMLPWrapper:
    """MLP PyTorch để train/infer GPU. API: fit(X,y), predict_proba(X), predict(X)."""
    def __init__(self, input_dim: int, hidden=(256, 128), dropout=0.1,
                 lr=1e-3, epochs=20, batch_size=256, seed=42):
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
        self._std = None  # per-feature standardization

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
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad()
                loss = self.loss_fn(self.net(xb), yb)
                loss.backward()
                self.opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        X = self._transform(X)
        p1 = torch.sigmoid(self.net(torch.from_numpy(X).to(self.device)).squeeze(1))
        p1 = p1.float().cpu().numpy()
        return np.stack([1.0 - p1, p1], axis=1)

    @torch.no_grad()
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_base(name: str, params: Optional[Dict] = None):
    """
    name ∈ {"knn","svm","xgb","nb","lr","mlp"} (không phân biệt hoa thường)
    Tự dùng GPU khi có (XGB/Torch; cuML nếu môi trường hỗ trợ).
    """
    name = (name or "").lower()
    params = params or {}

    # ---- XGBoost (GPU native) ----
    if name in ("xgb", "xgboost"):
        if not _HAS_XGB:
            raise RuntimeError("xgboost chưa được cài. Hãy `pip install xgboost`.")
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

    # ---- MLP ----
    if name in ("mlp", "mlp_torch"):
        if _HAS_TORCH:
            class _Factory:
                def __init__(self, p): self.p, self.impl = p, None
                def fit(self, X, y):
                    if self.impl is None:
                        self.impl = TorchMLPWrapper(
                            input_dim=X.shape[1],
                            hidden=self.p.get("hidden", (256, 128)),
                            dropout=self.p.get("dropout", 0.1),
                            lr=self.p.get("lr", 1e-3),
                            epochs=self.p.get("epochs", 20),
                            batch_size=self.p.get("batch_size", 256),
                            seed=self.p.get("seed", 42),
                        )
                    return self.impl.fit(X, y)
                def predict_proba(self, X): return self.impl.predict_proba(X)
                def predict(self, X): return self.impl.predict(X)
            return _Factory(params)
        # fallback: sklearn (CPU)
        mlp = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (256, 128)),
            alpha=params.get("alpha", 1e-4),
            learning_rate_init=params.get("learning_rate_init", 1e-3),
            max_iter=params.get("max_iter", 200),
            early_stopping=True,
            random_state=42,
        )
        return make_pipeline(_std(False), mlp)

    # ---- SVM ----
    if name in ("svm", "svm_linear", "svc"):
        if _HAS_CUML:
            return cuSVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "linear"),
                probability=True,
            )

        kernel = params.get("kernel", "linear")
        if kernel != "linear" or name == "svc":
            # Dùng libsvm (ổn định hơn liblinear) + xác suất gốc, không cần calibrate
            svc = SVC(
                kernel=kernel,
                C=params.get("C", 1.0),
                gamma=params.get("gamma", "scale"),
                probability=True,
                class_weight=params.get("class_weight", "balanced"),
            )
            return make_pipeline(_std(False), svc)

        # Linear SVM (liblinear) + calibration
        base = LinearSVC(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            dual=params.get("dual", "auto"),  # auto tốt cho tỉ lệ n_samples/n_features
            max_iter=params.get("max_iter", 20000),  # tăng lên cho chắc
            tol=params.get("tol", 1e-3),  # nới tol giúp hội tụ dễ hơn
            loss="squared_hinge"
        )
        try:
            calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
        except TypeError:
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
            return cuLR(C=params.get("C", 1.0), max_iter=params.get("max_iter", 1000))
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
    """Trả về p(y=1) dạng (N,) cho mọi model/pipeline/wrapper."""
    # pipeline?
    clf = model[-1] if hasattr(model, "steps") else model

    # PyTorch wrapper / factory
    if isinstance(clf, TorchMLPWrapper) or hasattr(model, "impl"):
        try:
            p = model.predict_proba(X)
        except Exception:
            p = model.impl.predict_proba(X)
        return p[:, 1].astype(np.float32) if p.ndim == 2 else p.astype(np.float32)

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1].astype(np.float32) if p.ndim == 2 else p.astype(np.float32)

    if hasattr(model, "decision_function"):
        z = model.decision_function(X).astype(np.float32).reshape(-1)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    yhat = model.predict(X).astype(np.float32).reshape(-1)
    return (0.01 + 0.98 * yhat).astype(np.float32)
