# scripts/evaluate.py
import os, json, yaml, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from tqdm.auto import tqdm

from src.models.stacking import StackingOOF
from src.tuning.optuna_tuner import tune_base, save_params
from src.utils.metrics import compute_all, tune_threshold
from src.utils.io import ensure_dir
from src.utils.seed import set_seed
from src.models.heads import build_base, predict_proba  # dùng lại build_base/predict_proba


def load_feats(artifacts_dir, dataset_name, split):
    base = Path(artifacts_dir) / dataset_name / split
    if not base.exists():
        raise FileNotFoundError(f"Split folder not found: {base}")
    y = np.load(base/"y.npy") if (base/"y.npy").exists() else None
    feats = {}
    for npy in sorted(base.glob("X_*.npy")):
        key = npy.stem.replace("X_", "")
        arr = np.load(npy)
        feats[key] = (arr[:, None] if arr.ndim == 1 else arr).astype(np.float32)
    if not feats:
        raise RuntimeError(f"No X_*.npy under {base}")
    return y, feats


def build_tasks(dataset_name: str, feats: dict) -> dict:
    tasks, ds = {}, dataset_name.lower()
    if 'text' in feats: tasks['text_only'] = feats['text']
    if 'caption' in feats: tasks['caption_only'] = feats['caption']
    if 'text' in feats and 'caption' in feats and ds in ['vcc','wcc']:
        tasks['text_caption'] = np.concatenate([feats['text'], feats['caption']], axis=1)
    if 'image_effnet' in feats:
        tasks['image_only'] = feats['image_effnet']
    elif 'image_precomp' in feats:
        tasks['image_only'] = feats['image_precomp']
    if 'text_clip' in feats and 'image_clip' in feats:
        tasks['text_image'] = np.concatenate([feats['text_clip'], feats['image_clip']], axis=1)
    if ds == 'cldi' and 'text' in feats and 'image_precomp' in feats:
        tasks['text_image'] = np.concatenate([feats['text'], feats['image_precomp']], axis=1)
    return tasks


def intersect_task_names(tasks_tr, tasks_dv, tasks_te):
    names = set(tasks_tr.keys()) & set(tasks_dv.keys())
    if tasks_te is not None: names &= set(tasks_te.keys())
    return sorted(names)


def _sanity_diag(tasks_tr: dict, tasks_dv: dict):
    import numpy as np
    def diag(name, X):
        X = np.asarray(X)
        uniq_rows = np.unique(X, axis=0).shape[0]
        zero_var = int((X.std(axis=0) < 1e-12).sum())
        print(f"[SANITY] {name}: shape={X.shape}, unique_rows={uniq_rows}, zero-var-cols={zero_var}")
    for nm, X in tasks_tr.items():
        diag("TRAIN::" + nm, X)
    for nm, X in tasks_dv.items():
        diag("DEV::" + nm, X)


def _pack_split(y, prob, thr, threshold_metric):
    if y is None or prob is None:
        return None
    pred = (prob >= thr).astype(int)
    return {
        "report": classification_report(y, pred, digits=4, output_dict=True),
        "metrics": {**compute_all(y, prob, threshold=thr), "best_thr_metric": threshold_metric}
    }


def _eval_estimator(model, Xtr, ytr, Xdv, ydv, Xte, yte, threshold_metric):
    model.fit(Xtr, ytr)
    prob_tr = predict_proba(model, Xtr)
    prob_dv = predict_proba(model, Xdv)
    prob_te = predict_proba(model, Xte) if Xte is not None else None
    thr, _ = tune_threshold(ydv, prob_dv, metric=threshold_metric)
    return {
        "train": _pack_split(ytr, prob_tr, thr, threshold_metric),
        "dev":   _pack_split(ydv, prob_dv, thr, threshold_metric),
        "test":  _pack_split(yte, prob_te, thr, threshold_metric),
        "threshold": float(thr)
    }


def train_and_eval_single_task(
    Xtr, ytr, Xdv, ydv, Xte, yte, base_learners, meta_learner, cv_folds,
    threshold_metric="f1", do_optuna=False, optuna_trials=0, optuna_objective="auc",
    outdir: Path | None = None, seed: int = 42, optuna_cfg: dict | None = None,
    bag_cfg: dict | None = None
):
    if outdir is not None:
        ensure_dir(outdir)

    # ===== Optuna: số trial riêng theo learner =====
    optuna_cfg = optuna_cfg or {}
    trials_default = int(optuna_cfg.get("trials_default", optuna_trials))
    trials_map = optuna_cfg.get("trials_per_learner", {}) or {}
    objective = optuna_cfg.get("objective", optuna_objective)

    base_params = {}
    if do_optuna:
        for name in tqdm(base_learners, desc="Optuna (per base)", leave=False):
            n_trials = int(trials_map.get(name, trials_default or 0))
            if n_trials <= 0:
                print(f"[Optuna] Skip tuning {name} (n_trials={n_trials})")
                continue
            print(f"[Optuna] Tuning {name} with {n_trials} trials (objective={objective})")
            params, best = tune_base(
                name, Xtr, ytr,
                n_trials=n_trials,
                cv_folds=cv_folds,
                random_state=seed,
                objective=objective,
            )
            base_params[name] = params
            if outdir is not None:
                (outdir/"optuna").mkdir(parents=True, exist_ok=True)
                save_params(str(outdir/"optuna"/f"{name}.json"), params)

    # ===== 6 BASE LEARNERS (đánh giá riêng lẻ) =====
    base_results = {}
    for name in base_learners:
        print(f"[Base] Training {name} ...")
        mdl = build_base(name, base_params.get(name, {}))
        r = _eval_estimator(mdl, Xtr, ytr, Xdv, ydv, Xte, yte, threshold_metric)
        base_results[name] = {
            "dev": r["dev"]["metrics"] if r["dev"] else None,
            "test": r["test"]["metrics"] if r["test"] else None
        }
        # Lưu preds base (dev/test)
        if outdir is not None:
            base_dir = outdir / "preds" / f"base_{name}"
            ensure_dir(base_dir)
            # re-fit to get probs again for saving
            mdl.fit(Xtr, ytr)
            prob_dv = predict_proba(mdl, Xdv)
            pd.DataFrame({"prob": prob_dv}).to_csv(base_dir/"dev_prob.csv", index=False)
            if Xte is not None:
                prob_te = predict_proba(mdl, Xte)
                pd.DataFrame({"prob": prob_te}).to_csv(base_dir/"test_prob.csv", index=False)

    # ===== BAGGING =====
    bag_cfg = bag_cfg or {}
    bag_enabled = bag_cfg.get("enabled", True)
    bag_base = bag_cfg.get("base", "lr")
    bag_n = int(bag_cfg.get("n_estimators", 15))
    bag_ms = float(bag_cfg.get("max_samples", 0.8))
    bag_mf = float(bag_cfg.get("max_features", 1.0))
    bag_boot = bool(bag_cfg.get("bootstrap", True))
    bag_random = int(bag_cfg.get("random_state", seed))

    bag_result = None
    if bag_enabled:
        print(f"[Bagging] base={bag_base}, n_estimators={bag_n}")
        base_est = build_base(bag_base, base_params.get(bag_base, {}))
        # sklearn>=1.4 dùng estimator=..., fallback base_estimator
        try:
            bag = BaggingClassifier(
                estimator=base_est,
                n_estimators=bag_n,
                max_samples=bag_ms,
                max_features=bag_mf,
                bootstrap=bag_boot,
                random_state=bag_random,
                n_jobs=-1,
            )
        except TypeError:
            bag = BaggingClassifier(
                base_estimator=base_est,
                n_estimators=bag_n,
                max_samples=bag_ms,
                max_features=bag_mf,
                bootstrap=bag_boot,
                random_state=bag_random,
                n_jobs=-1,
            )
        bag_result = _eval_estimator(bag, Xtr, ytr, Xdv, ydv, Xte, yte, threshold_metric)

    # ===== STACKING (OOF + meta=XGB/LR/LGBM) =====
    stack = StackingOOF(base_learners, base_params=base_params,
                        meta_learner=meta_learner, cv_folds=cv_folds,
                        random_state=seed)
    stack.fit(Xtr, ytr)
    prob_tr = stack.predict_proba(Xtr)
    prob_dv = stack.predict_proba(Xdv)
    prob_te = stack.predict_proba(Xte) if Xte is not None else None
    thr, _ = tune_threshold(ydv, prob_dv, metric=threshold_metric)

    res_stacking = {
        "train": _pack_split(ytr, prob_tr, thr, threshold_metric),
        "dev":   _pack_split(ydv, prob_dv, thr, threshold_metric),
        "test":  _pack_split(yte, prob_te, thr, threshold_metric),
        "threshold": float(thr)
    }

    # ===== Đóng gói báo cáo theo format yêu cầu =====
    res = {
        # giữ compatibility cũ: các block default trỏ về stacking
        "train": res_stacking["train"],
        "dev":   res_stacking["dev"],
        "test":  res_stacking["test"],
        "threshold": res_stacking["threshold"],
        "base_learners": base_results,
        "bagging": {
            "dev": bag_result["dev"]["metrics"] if (bag_result and bag_result["dev"]) else None,
            "test": bag_result["test"]["metrics"] if (bag_result and bag_result["test"]) else None,
            "threshold": bag_result["threshold"] if bag_result else None
        },
        "stacking": {
            "dev": res_stacking["dev"]["metrics"] if res_stacking["dev"] else None,
            "test": res_stacking["test"]["metrics"] if res_stacking["test"] else None,
            "threshold": res_stacking["threshold"]
        }
    }

    # ===== Lưu model stacking + dự đoán tập trung =====
    if outdir is not None:
        preds_dir, models_dir = outdir/"preds", outdir/"models"
        ensure_dir(preds_dir); ensure_dir(models_dir)
        import joblib
        joblib.dump(stack, models_dir/"stacking.joblib")
        (models_dir/"best_threshold.txt").write_text(str(float(thr)), encoding="utf-8")
        pd.DataFrame({"prob": prob_dv, "pred": (prob_dv>=thr).astype(int), "label": ydv}).to_csv(preds_dir/"dev.csv", index=False)
        if prob_te is not None:
            d = {"prob": prob_te, "pred": (prob_te>=thr).astype(int)}
            if yte is not None: d["label"] = yte
            pd.DataFrame(d).to_csv(preds_dir/"test.csv", index=False)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["text_only","image_only","text_image","text_caption","caption_only","tabular_only","all"], default="all")
    parser.add_argument("--optuna_trials", type=int, default=0)
    parser.add_argument("--optuna_objective", choices=["auc","f1"], default="auc")
    parser.add_argument("--sanity", action="store_true", help="in chuan doan feature truoc khi train")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ds = cfg['dataset']['name']
    art = cfg['dataset'].get('artifacts_dir','artifacts')
    base_learners = [x.strip() for x in cfg['models']['base_learners']]
    meta_learner = cfg['models']['stacking'].get('meta','logreg')
    cv_folds = int(cfg.get('runtime', {}).get('cv_folds', 5))
    thr_metric = cfg.get('models', {}).get('threshold_metric', 'f1')
    seed = cfg.get('runtime', {}).get('seed', cfg.get('seed', 42))
    set_seed(seed)

    # Optuna config
    optuna_cfg = (cfg.get('models', {}).get('optuna', {}) or {})
    if args.optuna_objective:
        optuna_cfg['objective'] = args.optuna_objective
    if 'trials_default' not in optuna_cfg and args.optuna_trials:
        optuna_cfg['trials_default'] = args.optuna_trials

    # Bagging config
    bag_cfg = (cfg.get('models', {}).get('ensemble', {}).get('bagging', {}) or {})

    y_tr, feats_tr = load_feats(art, ds, 'train')
    y_dv, feats_dv = load_feats(art, ds, 'dev')
    try:
        y_te, feats_te = load_feats(art, ds, 'test')
    except Exception:
        y_te, feats_te = None, None

    tasks_tr = build_tasks(ds, feats_tr)
    tasks_dv = build_tasks(ds, feats_dv)
    tasks_te = build_tasks(ds, feats_te) if feats_te is not None else None

    if args.sanity:
        _sanity_diag(tasks_tr, tasks_dv)

    if args.task == "all":
        to_run = intersect_task_names(tasks_tr, tasks_dv, tasks_te)
        if not to_run:
            raise RuntimeError("No common tasks across splits.")
    else:
        need = [args.task]
        common = intersect_task_names(tasks_tr, tasks_dv, tasks_te)
        if set(need) - set(common):
            raise RuntimeError(f"Requested task(s) not available across splits: {set(need)-set(common)}")
        to_run = need

    results = {}
    out_root = Path(art)/ds/"results"
    ensure_dir(out_root)

    for t in tqdm(to_run, desc="Tasks"):
        print(f"\n=== Running task: {t} ===")
        Xtr, Xdv = tasks_tr[t], tasks_dv[t]
        Xte = tasks_te[t] if tasks_te is not None else None
        res = train_and_eval_single_task(
            Xtr, y_tr, Xdv, y_dv, Xte, y_te,
            base_learners, meta_learner, cv_folds,
            threshold_metric=thr_metric,
            do_optuna=bool(optuna_cfg) or (args.optuna_trials>0),
            optuna_trials=args.optuna_trials,
            optuna_objective=optuna_cfg.get('objective', args.optuna_objective),
            outdir=out_root/t,
            seed=seed,
            optuna_cfg=optuna_cfg,
            bag_cfg=bag_cfg,
        )
        results[t] = res
        with open(out_root/t/"metrics.json","w",encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    with open(out_root/"all_results.json","w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved:", out_root/"all_results.json")


if __name__ == "__main__":
    main()
