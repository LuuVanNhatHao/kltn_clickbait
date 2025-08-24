# scripts/evaluate.py
import os, json, yaml, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from src.models.stacking import StackingOOF
from src.tuning.optuna_tuner import tune_base, save_params
from src.utils.metrics import compute_all, tune_threshold
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


def load_feats(artifacts_dir, dataset_name, split):
    """
    Đọc y.npy (nếu có) và mọi X_*.npy trong artifacts/<dataset>/<split>/
    Trả về: y (np.ndarray hoặc None), feats (dict: name->np.ndarray)
    """
    base = Path(artifacts_dir) / dataset_name / split
    if not base.exists():
        raise FileNotFoundError(f"Split folder not found: {base}")

    y_path = base / "y.npy"
    y = np.load(y_path) if y_path.exists() else None

    feats = {}
    for npy in sorted(base.glob("X_*.npy")):
        key = npy.stem.replace("X_", "")
        arr = np.load(npy)
        if arr.ndim == 1:
            arr = arr[:, None]
        feats[key] = arr.astype(np.float32)
    if not feats:
        raise RuntimeError(f"No X_*.npy features under {base}")
    return y, feats


def build_tasks(dataset_name: str, feats: dict) -> dict:
    """
    Tạo ma trận đặc trưng theo từng 'task' (kịch bản) từ dict feats.
    Chỉ ghép những gì thật sự tồn tại.
    """
    tasks = {}
    ds = dataset_name.lower()

    # text/caption (HF)
    if 'text' in feats:
        tasks['text_only'] = feats['text']
    if 'caption' in feats:
        tasks['caption_only'] = feats['caption']
    if 'text' in feats and 'caption' in feats and ds in ['vcc', 'wcc']:
        tasks['text_caption'] = np.concatenate([feats['text'], feats['caption']], axis=1)

    # image only (ưu tiên effnet; nếu không có, dùng precomputed)
    if 'image_effnet' in feats:
        tasks['image_only'] = feats['image_effnet']
    elif 'image_multi' in feats:
        tasks['image_only'] = feats['image_multi']
    elif 'image_precomp' in feats:
        tasks['image_only'] = feats['image_precomp']

    # text + image
    # (A) CLIP cho VCC/WCC
    if 'text_clip' in feats and 'image_clip' in feats:
        tasks['text_image'] = np.concatenate([feats['text_clip'], feats['image_clip']], axis=1)
    # (B) CLDI: HF text + precomputed image
    if ds == 'cldi' and 'text' in feats and 'image_precomp' in feats:
        tasks['text_image'] = np.concatenate([feats['text'], feats['image_precomp']], axis=1)

    return tasks


def intersect_task_names(tasks_tr, tasks_dv, tasks_te):
    names = set(tasks_tr.keys()) & set(tasks_dv.keys())
    if tasks_te is not None:
        names &= set(tasks_te.keys())
    return sorted(names)


def train_and_eval_single_task(
    Xtr, ytr, Xdv, ydv, Xte, yte,
    base_learners, meta_learner, cv_folds,
    threshold_metric="f1",
    do_optuna=False, optuna_trials=0, outdir: Path | None = None, seed: int = 42
):
    """
    Train StackingOOF cho 1 task, tune threshold trên dev, trả metrics + reports + preds.
    """
    ensure_dir(outdir) if outdir is not None else None
    base_params = {}

    if do_optuna and optuna_trials > 0:
        for name in tqdm(base_learners, desc="Optuna (per base)", leave=False):
            params, best = tune_base(name, Xtr, ytr,
                                     n_trials=optuna_trials,
                                     cv_folds=cv_folds,
                                     random_state=seed,
                                     objective="f1")
            base_params[name] = params
            if outdir is not None:
                (outdir / "optuna").mkdir(parents=True, exist_ok=True)
                save_params(str(outdir / "optuna" / f"{name}.json"), params)

    model = StackingOOF(
        base_names=base_learners,
        base_params=base_params,
        meta_learner=meta_learner,
        cv_folds=cv_folds,
        random_state=seed
    )
    model.fit(Xtr, ytr)

    # Probabilities
    prob_tr = model.predict_proba(Xtr)
    prob_dv = model.predict_proba(Xdv)
    prob_te = model.predict_proba(Xte) if Xte is not None else None

    # Tune threshold trên dev
    thr, best = tune_threshold(ydv, prob_dv, metric=threshold_metric)

    # Metrics + reports
    def pack_set(name, y, prob):
        if y is None or prob is None:
            return None
        from sklearn.metrics import classification_report
        pred = (prob >= thr).astype(int)
        report = classification_report(y, pred, digits=4, output_dict=True)
        metrics = compute_all(y, prob, thr=thr)
        metrics["best_thr"] = float(thr)
        metrics["best_thr_metric"] = threshold_metric
        metrics["best_thr_score_dev"] = float(best)
        return {"report": report, "metrics": metrics}

    res = {
        "train": pack_set("train", ytr, prob_tr),
        "dev":   pack_set("dev",   ydv, prob_dv),
        "test":  pack_set("test",  yte, prob_te),
        "threshold": float(thr)
    }

    # Save preds CSV
    if outdir is not None:
        preds_dir = outdir / "preds"
        ensure_dir(preds_dir)
        pd.DataFrame({"prob": prob_tr, "pred": (prob_tr >= thr).astype(int), "label": ytr}).to_csv(preds_dir / "train.csv", index=False)
        pd.DataFrame({"prob": prob_dv, "pred": (prob_dv >= thr).astype(int), "label": ydv}).to_csv(preds_dir / "dev.csv", index=False)
        if prob_te is not None:
            d = {"prob": prob_te, "pred": (prob_te >= thr).astype(int)}
            if yte is not None:
                d["label"] = yte
            pd.DataFrame(d).to_csv(preds_dir / "test.csv", index=False)

        # Save model
        import joblib
        models_dir = outdir / "models"
        ensure_dir(models_dir)
        joblib.dump(model, models_dir / "stacking.joblib")
        with open(models_dir / "best_threshold.txt", "w", encoding="utf-8") as f:
            f.write(str(float(thr)))

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["text_only", "image_only", "text_image", "text_caption", "caption_only", "tabular_only", "all"], default="all")
    parser.add_argument("--optuna_trials", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ds = cfg['dataset']['name']
    art = cfg['dataset'].get('artifacts_dir', 'artifacts')

    base_learners = [x.strip() for x in cfg['models']['base_learners']]
    meta_learner = cfg['models']['stacking'].get('meta', 'logreg')
    cv_folds = int(cfg.get('runtime', {}).get('cv_folds', 5))
    thr_metric = cfg.get('models', {}).get('threshold_metric', 'f1')
    seed = cfg.get('runtime', {}).get('seed', cfg.get('seed', 42))
    set_seed(seed)

    y_tr, feats_tr = load_feats(art, ds, 'train')
    y_dv, feats_dv = load_feats(art, ds, 'dev')
    # test có thể không có y
    try:
        y_te, feats_te = load_feats(art, ds, 'test')
    except Exception:
        y_te, feats_te = None, None

    tasks_tr = build_tasks(ds, feats_tr)
    tasks_dv = build_tasks(ds, feats_dv)
    tasks_te = build_tasks(ds, feats_te) if feats_te is not None else None

    if args.task == "all":
        candidate_tasks = intersect_task_names(tasks_tr, tasks_dv, tasks_te)
        if not candidate_tasks:
            raise RuntimeError("No common tasks across splits. Check which X_*.npy exist per split.")
        to_run = candidate_tasks
    else:
        # chỉ chạy nếu task đó tồn tại trong train/dev (và test nếu có)
        wanted = [args.task]
        common = intersect_task_names(tasks_tr, tasks_dv, tasks_te)
        missing = set(wanted) - set(common)
        if missing:
            raise RuntimeError(f"Requested task(s) not available across splits: {missing}")
        to_run = wanted

    results = {}
    out_root = Path(art) / ds / "results"
    ensure_dir(out_root)

    for t in tqdm(to_run, desc="Tasks"):
        Xtr = tasks_tr[t]; Xdv = tasks_dv[t]; Xte = tasks_te[t] if tasks_te is not None else None
        res = train_and_eval_single_task(
            Xtr, y_tr, Xdv, y_dv, Xte, y_te,
            base_learners, meta_learner, cv_folds,
            threshold_metric=thr_metric,
            do_optuna=(args.optuna_trials > 0),
            optuna_trials=args.optuna_trials,
            outdir=out_root / t,
            seed=seed
        )
        results[t] = res

        # Save per-task metrics
        with open(out_root / t / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    # Save a summary file
    with open(out_root / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved:", out_root / "all_results.json")


if __name__ == "__main__":
    main()
