
import os, json, yaml, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from src.models.heads import build_base, predict_proba
from src.models.stacking import StackingOOF
from src.utils.metrics import compute_all
from src.tuning.optuna_tuner import tune_base, save_params
from tqdm.auto import tqdm

def load_feats(artifacts_dir, dataset_name, split):
    base = Path(artifacts_dir)/dataset_name/split
    y = np.load(base/'y.npy')
    feats = {}
    # discover any X_*.npy
    for npy in base.glob("X_*.npy"):
        key = npy.stem.replace("X_","")
        feats[key] = np.load(npy)
    return y, feats

def build_tasks(dataset_name: str, feats: dict) -> dict:
    tasks = {}
    ds = dataset_name.lower()
    # text / caption (HF encoders)
    if 'text' in feats:
        tasks['text_only'] = feats['text']
    if 'caption' in feats:
        tasks['caption_only'] = feats['caption']
    if 'text' in feats and 'caption' in feats and ds in ['vcc','wcc']:
        tasks['text_caption'] = np.concatenate([feats['text'], feats['caption']], axis=1)

    # image only (prefer EffNet; else precomputed)
    if 'image_effnet' in feats:
        tasks['image_only'] = feats['image_effnet']
    elif 'image_precomp' in feats:
        tasks['image_only'] = feats['image_precomp']

    # text + image (two regimes)
    # (A) CLIP for VCC/WCC

    if 'text_clip' in feats and 'image_clip' in feats:
        tasks['text_image'] = np.concatenate([feats['text_clip'], feats['image_clip']], axis=1)

    # (B) CLDI: concat HF text + precomputed image
    if ds == 'cldi' and 'text' in feats and 'image_precomp' in feats:
        tasks['text_image'] = np.concatenate([feats['text'], feats['image_precomp']], axis=1)

    return tasks

def train_and_eval_single_task(Xtr, ytr, Xdv, ydv, Xte, yte, base_learners, meta_learner, cv_folds, threshold_metric, do_optuna=False, optuna_trials=20, outdir: Path | None = None):
    base_params = {}
    if do_optuna and optuna_trials > 0:
        # Tune on train set for each base learner separately
        for name in tqdm(base_learners, desc="Optuna tuning", leave=False):
            params, best = tune_base(name, Xtr, ytr, n_trials=optuna_trials, cv_folds=cv_folds)
            base_params[name] = params
            if outdir is not None:
                (outdir/"optuna").mkdir(parents=True, exist_ok=True)
                save_params(str(outdir/"optuna"/f"{name}.json"), params)

    stack = StackingOOF(base_learners, base_params=base_params, meta_learner=meta_learner, cv_folds=cv_folds)
    stack.fit(Xtr, ytr)

    def run_set(split_name, X, y):
        probs = stack.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        report = classification_report(y, preds, digits=4, output_dict=True)
        metrics = compute_all(y, probs)
        return report, metrics

    rep_tr, m_tr = run_set("train", Xtr, ytr)
    rep_dv, m_dv = run_set("dev", Xdv, ydv)
    rep_te, m_te = run_set("test", Xte, yte)

    return {"train": {"report": rep_tr, "metrics": m_tr},
            "dev": {"report": rep_dv, "metrics": m_dv},
            "test": {"report": rep_te, "metrics": m_te}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["text_only","image_only","text_image","text_caption","caption_only","tabular_only","all"], default="all")
    parser.add_argument("--optuna_trials", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ds = cfg['dataset']['name']
    art = cfg['dataset'].get('artifacts_dir','artifacts')
    base_learners = [x.strip() for x in cfg['models']['base_learners']]
    meta_learner = cfg['models']['stacking'].get('meta','logreg')
    cv_folds = int(cfg['runtime'].get('cv_folds',5))
    threshold_metric = cfg.get('models',{}).get('threshold_metric','f1')

    y_tr, feats_tr = load_feats(art, ds, 'train')
    y_dv, feats_dv = load_feats(art, ds, 'dev')
    y_te, feats_te = load_feats(art, ds, 'test')

    tasks_tr = build_tasks(ds, feats_tr)
    tasks_dv = build_tasks(ds, feats_dv)
    tasks_te = build_tasks(ds, feats_te)

    to_run = list(tasks_tr.keys()) if args.task == "all" else [args.task]

    results = {}
    out_root = Path(art)/ds/"results"
    out_root.mkdir(parents=True, exist_ok=True)

    for t in tqdm(to_run, desc="Tasks"):
        Xtr = tasks_tr[t]; Xdv = tasks_dv[t]; Xte = tasks_te[t]
        res = train_and_eval_single_task(Xtr, y_tr, Xdv, y_dv, Xte, y_te, base_learners, meta_learner, cv_folds, threshold_metric, do_optuna=args.optuna_trials>0, optuna_trials=args.optuna_trials, outdir=out_root.joinpath(t))
        results[t] = res

    json.dump(results, open(out_root/"all_results.json","w", encoding="utf-8"), indent=2)
    print("Saved:", out_root/"all_results.json")

if __name__ == "__main__":
    main()
