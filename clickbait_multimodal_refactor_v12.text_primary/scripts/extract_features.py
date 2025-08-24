# scripts/extract_features.py
import argparse, yaml, os, json, numpy as np, pandas as pd
from pathlib import Path
from src.data.datasets import DatasetAdapter, ColumnMap
from src.features.text_encoder import HFTextEncoder
from src.features.image_encoder import EfficientNetEncoder   # wrapper EffNet timm
from src.features.clip_encoder import CLIPTextEncoder, CLIPImageEncoder
from tqdm.auto import tqdm

# ---------- PATH RESOLVER cho ảnh ----------
def resolve_path(p: str, root: Path | None = None) -> str | None:
    """Thử nhiều ứng viên (., .., ../..) + root để tìm file ảnh thật sự tồn tại."""
    if not isinstance(p, str) or not p:
        return None
    q = Path(p)
    cands = []
    if q.is_absolute():
        cands.append(q)
    else:
        cands += [Path(p), Path("..")/p, Path("../..")/p]
        if root is not None:
            cands += [root/p, root/Path(p).name]
    for c in cands:
        if c.exists():
            return str(c)
    return None

# ---------- DatasetAdapter ----------
def build_adapter(cfg, split_csv):
    cols = cfg["columns"]
    column_map = ColumnMap(
        text=cols.get("text"),
        text_fields=cols.get("text_fields"),
        label=cols["label"],
        claim_image=cols.get("claim_image"),
        document_image=cols.get("document_image"),
        caption=cols.get("caption"),
    )
    return DatasetAdapter(split_csv, column_map, cfg)

def compose_text(df: pd.DataFrame, colmap: ColumnMap) -> pd.Series:
    if colmap.text and colmap.text in df.columns:
        return df[colmap.text].astype(str)
    if colmap.caption and colmap.caption in df.columns and (not colmap.text_fields):
        return df[colmap.caption].astype(str)
    if colmap.text_fields:
        xs = [df[c].astype(str) for c in colmap.text_fields if c in df.columns]
        if xs:
            return xs[0].fillna('')
    return pd.Series(['']*len(df))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', choices=['train','dev','test'], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    dataset_name = cfg['dataset']['name']
    artifacts_dir = Path(cfg['dataset']['artifacts_dir'])
    split_csv = cfg['dataset'][args.split]

    adapter = build_adapter(cfg, split_csv)
    df = adapter.df.copy()

    # labels
    y = adapter.get_labels()
    if y.dtype == object:
        y = y.astype(str).str.lower().map({'clickbait':1, 'non-clickbait':0, '0':0, '1':1}).fillna(y)
        try: y = y.astype(int)
        except Exception: pass

    out_dir = artifacts_dir/dataset_name/args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = {}

    # ---------- TEXT (HF) ----------
    txt_series = compose_text(df, adapter.colmap)
    if cfg['text_encoder'].get('enabled', True) and txt_series is not None:
        hf_name = cfg['text_encoder']['name']
        max_len = cfg['text_encoder'].get('max_length', 128)
        batch_size = cfg['text_encoder'].get('batch_size', 16)
        device = cfg['runtime'].get('device','auto')
        hf = HFTextEncoder(hf_name, max_length=max_len, batch_size=batch_size, device=device)
        feats['text'] = hf.encode(txt_series.fillna('').astype(str).tolist())

    # ---------- CAPTION (HF) ----------
    cap_col = adapter.colmap.caption
    if cap_col and cap_col in df.columns and cfg.get('caption_encoder', {}).get('enabled', False):
        cap_hf = HFTextEncoder(cfg['caption_encoder']['name'],
                               max_length=cfg['caption_encoder'].get('max_length', 64),
                               batch_size=cfg['caption_encoder'].get('batch_size', 16),
                               device=cfg['runtime'].get('device','auto'))
        feats['caption'] = cap_hf.encode(df[cap_col].fillna('').astype(str).tolist())

    # ---------- IMAGE (EfficientNet timm) ----------
    img_cfg = cfg.get('image_encoder', {})
    if img_cfg.get('enabled', False):
        claim_col = adapter.colmap.claim_image
        doc_col   = adapter.colmap.document_image
        img_root  = Path(img_cfg.get('root') or cfg['dataset'].get('image_root','.' ))
        image_size = img_cfg.get('image_size', 380)
        encoder = EfficientNetEncoder(model_name=img_cfg.get('name','tf_efficientnet_b4'),
                                      image_size=image_size,
                                      device=cfg['runtime'].get('device','auto'))
        raw_paths = []
        for col in [claim_col, doc_col, 'img_path']:
            if col and col in df.columns:
                raw_paths = df[col].astype(str).tolist()
                break
        # resolve từng path để tránh encode ảnh xám hàng loạt
        res_paths = [resolve_path(p, img_root) for p in raw_paths]
        # EfficientNetEncoder.encode_paths chấp nhận None/không tồn tại (fallback ảnh xám),
        # nhưng ở đây paths đã resolve → phần lớn sẽ là file thật.
        feats['image_effnet'] = encoder.encode_paths(res_paths, root=None, batch_size=16)

    # ---------- CLIP (text + image) ----------
    clip_cfg = cfg.get('clip', {})
    if clip_cfg.get('enabled', False):
        clip_name = clip_cfg['name'].split('/')[-1] if 'openclip_' in clip_cfg['name'] else clip_cfg['name']
        device = cfg['runtime'].get('device','auto')
        if clip_cfg.get('text_enabled', True):
            txt_for_clip = (df[cap_col].fillna('').astype(str).tolist()
                            if cap_col and cap_col in df.columns else
                            txt_series.fillna('').astype(str).tolist())
            feats['text_clip'] = CLIPTextEncoder(clip_name, device=device).encode(txt_for_clip)
        if clip_cfg.get('image_enabled', True):
            raw_paths = []
            for col in [adapter.colmap.claim_image, adapter.colmap.document_image, 'img_path']:
                if col and col in df.columns:
                    raw_paths = df[col].astype(str).tolist()
                    break
            if raw_paths:
                res_paths = [resolve_path(p, Path(clip_cfg.get('root','.'))) for p in raw_paths]
                feats['image_clip'] = CLIPImageEncoder(clip_name, device=device).encode(res_paths, batch_size=16)

    # ---------- SAVE ----------
    np.save(out_dir/"y.npy", y)
    for key, arr in feats.items():
        if arr is not None:
            np.save(out_dir/f"X_{key}.npy", arr)

    meta = dict(num_samples=len(df), label_name=adapter.colmap.label, ids=df.index.tolist())
    json.dump(meta, open(out_dir/"meta.json","w", encoding="utf-8"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
