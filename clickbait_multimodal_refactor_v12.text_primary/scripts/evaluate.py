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
        "threshold": float(thr),
        "probs": {"train": prob_tr, "dev": prob_dv, "test": prob_te}
    }


# ---------- Voting helpers ----------
def _normalize_weights(names, weights_dict=None):
    """Trả về vector trọng số (cùng thứ tự với names). Nếu không chỉ định → đều nhau."""
    if not weights_dict:
        return np.ones(len(names), dtype=np.float32) / max(len(names), 1)
    w = np.array([float(weights_dict.get(n, 0.0)) for n in names], dtype=np.float32)
    if w.sum() <= 0:
        w = np.ones(len(names), dtype=np.float32)
    return w / w.sum()

def _soft_vote(prob_list, weights):
    P = np.vstack(prob_list)  # [M, N]
    return (weights[:, None] * P).sum(axis=0)  # [N]


def train_and_eval_single_task(
    Xtr, ytr, Xdv, ydv, Xte, yte, base_learners, meta_learner, cv_folds,
    threshold_metric="f1", do_optuna=False, optuna_trials=0, optuna_objective="auc",
    outdir: Path | None = None, seed: int = 42, optuna_cfg: dict | None = None,
    voting_cfg: dict | None = None
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

    # ===== 6 BASE LEARNERS (đánh giá riêng lẻ & lưu xác suất cho Voting) =====
    base_results, base_probs_dev, base_probs_test, base_probs_train = {}, {}, {}, {}
    for name in base_learners:
        print(f"[Base] Training {name} ...")
        mdl = build_base(name, base_params.get(name, {}))
        r = _eval_estimator(mdl, Xtr, ytr, Xdv, ydv, Xte, yte, threshold_metric)
        base_results[name] = {
            "dev": r["dev"]["metrics"] if r["dev"] else None,
            "test": r["test"]["metrics"] if r["test"] else None
        }
        base_probs_train[name] = r["probs"]["train"]
        base_probs_dev[name]   = r["probs"]["dev"]
        base_probs_test[name]  = r["probs"]["test"]

        # Lưu preds base (dev/test)
        if outdir is not None:
            base_dir = outdir / "preds" / f"base_{name}"
            ensure_dir(base_dir)
            pd.DataFrame({"prob": r["probs"]["dev"]}).to_csv(base_dir/"dev_prob.csv", index=False)
            if r["probs"]["test"] is not None:
                pd.DataFrame({"prob": r["probs"]["test"]}).to_csv(base_dir/"test_prob.csv", index=False)

    # ===== VOTING (soft / weighted) =====
    voting_cfg = voting_cfg or {}
    vote_enabled = bool(voting_cfg.get("enabled", True))
    vote_weights = voting_cfg.get("weights", None)  # dict: {learner: weight}
    voting_result = None

    if vote_enabled:
        # chỉ dùng các base có xác suất hợp lệ trên DEV
        names = [n for n in base_learners if n in base_probs_dev and base_probs_dev[n] is not None]
        if not names:
            print("[Voting] No base probabilities available; skipping.")
        else:
            w = _normalize_weights(names, vote_weights)
            dv_probs = _soft_vote([base_probs_dev[n] for n in names], w)
            te_probs = None
            if any(base_probs_test.get(n) is None for n in names):
                te_probs = None
            else:
                te_probs = _soft_vote([base_probs_test[n] for n in names], w)

            thr_v, _ = tune_threshold(ydv, dv_probs, metric=threshold_metric)
            voting_result = {
                "train": None,  # không cần cho voting
                "dev":   _pack_split(ydv, dv_probs, thr_v, threshold_metric),
                "test":  _pack_split(yte, te_probs, thr_v, threshold_metric),
                "threshold": float(thr_v),
                "weights": {n: float(wi) for n, wi in zip(names, w.tolist())}
            }

            # lưu dự đoán voting
            if outdir is not None:
                preds_dir, models_dir = outdir/"preds", outdir/"models"
                ensure_dir(preds_dir); ensure_dir(models_dir)
                pd.DataFrame({"prob": dv_probs,
                              "pred": (dv_probs>=thr_v).astype(int),
                              "label": ydv}).to_csv(preds_dir/"voting_dev.csv", index=False)
                if te_probs is not None:
                    d = {"prob": te_probs, "pred": (te_probs>=thr_v).astype(int)}
                    if yte is not None: d["label"] = yte
                    pd.DataFrame(d).to_csv(preds_dir/"voting_test.csv", index=False)
                (models_dir/"voting_weights.json").write_text(
                    json.dumps(voting_result["weights"], ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                (models_dir/"voting_best_threshold.txt").write_text(str(float(thr_v)), encoding="utf-8")

    # ===== STACKING (OOF + meta=XGB/LR/…) =====
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
        "voting": {
            "dev": voting_result["dev"]["metrics"] if (voting_result and voting_result["dev"]) else None,
            "test": voting_result["test"]["metrics"] if (voting_result and voting_result["test"]) else None,
            "threshold": voting_result["threshold"] if voting_result else None,
            "weights": voting_result["weights"] if voting_result else None
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

    # Voting config
    voting_cfg = (cfg.get('models', {}).get('ensemble', {}).get('voting', {}) or {})

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
            voting_cfg=voting_cfg,
        )
        results[t] = res
        with open(out_root/t/"metrics.json","w",encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    with open(out_root/"all_results.json","w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved:", out_root/"all_results.json")


if __name__ == "__main__":
    main()
