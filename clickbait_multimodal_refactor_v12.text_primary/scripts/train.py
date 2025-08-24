import sys, os, numpy as np
from src.utils.io import load_yaml, ensure_dir, save_json
from src.utils.seed import set_seed
from src.utils.metrics import binary_metrics, tune_threshold
from src.models.stacking import StackingOOF

def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg.get("seed", 42))
    name = cfg["dataset_name"]
    paths = cfg["paths"]
    train_dir = os.path.join(paths["artifacts_dir"], "features", name)
    data = np.load(os.path.join(train_dir, "train_dev_test.npz"), allow_pickle=True)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xdv, ydv = data["Xdv"], data["ydv"]

    learners = cfg["training"]["base_learners"]
    meta     = cfg["training"]["meta_learner"]
    folds    = cfg["training"]["cv_folds"]
    cal      = cfg["training"].get("calibrate", True)

    model = StackingOOF(base_names=learners, meta_learner=meta, cv_folds=folds, random_state=cfg.get("seed", 42))
    model.fit(Xtr, ytr)
    prob_dv = model.predict_proba(Xdv)
    thr, best = tune_threshold(ydv, prob_dv, metric=cfg["training"].get("threshold_metric","f1"))
    m = binary_metrics(ydv, prob_dv, thr=thr)
    m["best_thr"] = float(thr)

    # save
    out_dir = os.path.join(paths["artifacts_dir"], "models", name)
    ensure_dir(out_dir)
    import joblib
    joblib.dump(model, os.path.join(out_dir, "stacking.joblib"))
    save_json(m, os.path.join(paths["artifacts_dir"], "metrics", name + "_dev_metrics.json"))
    print("Dev metrics:", m)

if __name__ == "__main__":
    main(sys.argv[1])
