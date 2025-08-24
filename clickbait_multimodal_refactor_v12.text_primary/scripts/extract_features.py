import argparse, yaml, os, json, numpy as np, pandas as pd
from pathlib import Path
from src.data.datasets import DatasetAdapter, ColumnMap
from src.features.text_encoder import HFTextEncoder
from src.features.image_encoder import EfficientNetEncoder, MultiImageEncoder  # ✅ thêm MultiImageEncoder
from src.features.clip_encoder import CLIPTextEncoder, CLIPImageEncoder
from tqdm.auto import tqdm

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
    # Đừng truyền cfg.get('precomputed') nữa — truyền full cfg để bắt 'image_encoder.root'/'clip.root'
    return DatasetAdapter(split_csv, column_map, cfg)


def compose_text(df: pd.DataFrame, colmap: ColumnMap) -> pd.Series:
    # Prefer a single 'text' column, else join multiple fields if provided
    parts = []
    if colmap.text is not None and colmap.text in df.columns:
        return df[colmap.text].astype(str)
    if colmap.caption is not None and colmap.caption in df.columns and (not colmap.text_fields):
        return df[colmap.caption].astype(str)
    if colmap.text_fields:
        xs = []
        for c in colmap.text_fields:
            if c in df.columns:
                xs.append(df[c].astype(str))
        if xs:
            return (xs[0].fillna(''))
        # fallback empty
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
    # normalize labels to {0,1}
    if y.dtype == object:
        y = y.astype(str).str.lower().map({'clickbait':1, 'non-clickbait':0, '0':0, '1':1}).fillna(y)
        try:
            y = y.astype(int)
        except Exception:
            pass


    out_dir = artifacts_dir/dataset_name/args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = {}

    # ---------------- TEXT ENCODING (HF) ----------------
    # Compose text from config: text or text_fields or caption
    txt_series = compose_text(df, adapter.colmap)
    if cfg['text_encoder'].get('enabled', True) and txt_series is not None:
        hf_name = cfg['text_encoder']['name']
        max_len = cfg['text_encoder'].get('max_length', 128)
        batch_size = cfg['text_encoder'].get('batch_size', 16)
        hf = HFTextEncoder(hf_name, max_length=max_len, batch_size=batch_size, device=cfg['runtime'].get('device','auto'))
        feats['text'] = hf.encode(txt_series.fillna('').astype(str).tolist())

    # ---------------- CAPTION (optional HF) -------------
    cap_col = adapter.colmap.caption
    if cap_col and cap_col in df.columns and cfg.get('caption_encoder', {}).get('enabled', False):
        cap_hf = HFTextEncoder(cfg['caption_encoder']['name'],
                               max_length=cfg['caption_encoder'].get('max_length', 64),
                               batch_size=cfg['caption_encoder'].get('batch_size', 16),
                               device=cfg['runtime'].get('device','auto'))
        feats['caption'] = cap_hf.encode(df[cap_col].fillna('').astype(str).tolist())

    # ---------------- IMAGE ENCODING --------------------
    # Case A: use raw images via EfficientNet
    if cfg.get('image_encoder', {}).get('enabled', False):
        claim_col = adapter.colmap.claim_image
        doc_col   = adapter.colmap.document_image
        img_root  = Path(cfg['image_encoder'].get('root', '.'))
        image_size = cfg['image_encoder'].get('image_size', 380)
        encoder = MultiImageEncoder(model_name=cfg['image_encoder'].get('name','tf_efficientnet_b4'),
                                    image_size=image_size, device=cfg['runtime'].get('device','auto'))
        paths = []
        if claim_col and claim_col in df.columns:
            paths = df[claim_col].astype(str).tolist()
        elif doc_col and doc_col in df.columns:
            paths = df[doc_col].astype(str).tolist()
        else:
            # fallback to a generic 'img_path' column if exists
            if 'img_path' in df.columns:
                paths = df['img_path'].astype(str).tolist()
        if paths:
            feats['image_effnet'] = encoder.encode(paths, root=str(img_root))

    # Case B: CLIP encoders (text/image) for VCC/WCC
    if cfg.get('clip', {}).get('enabled', False):
        clip_name = cfg['clip']['name']
        device = cfg['runtime'].get('device','auto')
        if cfg['clip'].get('text_enabled', True):
            # Prefer caption column; else fall back to text
            cap_col = adapter.colmap.caption
            if cap_col and cap_col in df.columns:
                txt_for_clip = df[cap_col].fillna('').astype(str).tolist()
            else:
                txt_for_clip = txt_series.fillna('').astype(str).tolist()
            feats['text_clip'] = CLIPTextEncoder(clip_name, device=device).encode(txt_for_clip)
        if cfg['clip'].get('image_enabled', True):
            paths = []
            # try standard path columns
            for col in [adapter.colmap.claim_image, adapter.colmap.document_image, 'img_path']:
                if col and col in df.columns:
                    paths = df[col].astype(str).tolist()
                    break
                if isinstance(col, str) and col == 'img_path' and 'img_path' in df.columns:
                    paths = df['img_path'].astype(str).tolist()
            if paths:
                feats['image_clip'] = CLIPImageEncoder(clip_name, device=device).encode(paths, root=cfg['clip'].get('root','.'))

    # Case C: Precomputed image features from CSV range (for CLDI)
    pre = cfg.get('precomputed', {})
    if pre.get('enabled', False):
        rng = pre.get('image_range')  # [start, end] inclusive
        if rng:
            start, end = rng
            cols = [str(i) for i in range(start, end+1) if str(i) in df.columns] + [i for i in range(start, end+1) if i in df.columns]
            # Build matrix in order
            matrices = []
            for i in range(start, end+1):
                cstr = str(i)
                if cstr in df.columns:
                    matrices.append(df[cstr].astype(np.float32).to_numpy())
                elif i in df.columns:
                    matrices.append(df[i].astype(np.float32).to_numpy())
            if matrices:
                Ximg = np.stack(matrices, axis=1)
                feats['image_precomp'] = Ximg.astype(np.float32)

    # ---------------- SAVE ------------------------------
    np.save(out_dir/"y.npy", y)
    for key, arr in feats.items():
        if arr is not None:
            np.save(out_dir/f"X_{key}.npy", arr)

    meta = dict(num_samples=len(df), label_name=adapter.colmap.label, ids=df.index.tolist())
    json.dump(meta, open(out_dir/"meta.json","w", encoding="utf-8"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
