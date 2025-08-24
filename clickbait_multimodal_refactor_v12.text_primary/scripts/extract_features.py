# scripts/extract_features.py
import argparse, yaml, os, json, numpy as np, pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from src.data.datasets import DatasetAdapter, ColumnMap
from src.features.text_encoder import HFTextEncoder
from src.features.image_encoder import MultiImageEncoder, EfficientNetEncoder
from src.features.clip_encoder import CLIPTextEncoder, CLIPImageEncoder


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
    # Truyền full cfg để Adapter biết image_root/clip.root nếu cần
    return DatasetAdapter(split_csv, column_map, cfg)


def compose_text(df: pd.DataFrame, colmap: ColumnMap) -> pd.Series:
    """Ưu tiên 1 cột 'text'; nếu không có thì ghép từ text_fields; nếu không có nữa thì dùng caption; fallback rỗng."""
    if colmap.text is not None and colmap.text in df.columns:
        return df[colmap.text].astype(str)

    if colmap.text_fields:
        xs = []
        for c in colmap.text_fields:
            if c in df.columns:
                xs.append(df[c].astype(str))
        if xs:
            # Ghép các field bằng khoảng trắng (có thể tùy chỉnh)
            return (" ".join([s.fillna('') for s in xs])).astype(str)

    if colmap.caption is not None and colmap.caption in df.columns:
        return df[colmap.caption].astype(str)

    return pd.Series([''] * len(df))


def _normalize_clip_name(name: str) -> str:
    """Chuẩn hóa tên model CLIP cho encoder của bạn."""
    if not isinstance(name, str):
        return name
    low = name.lower()
    if low.startswith("openclip_"):
        # ví dụ 'openclip_ViT-B-32' -> 'ViT-B-32'
        return name.split("openclip_", 1)[1]
    return name


def _join_root(paths, root: Path):
    abs_paths = []
    for p in paths:
        if isinstance(p, str) and p and not os.path.isabs(p):
            abs_paths.append(str(root / p))
        else:
            abs_paths.append(p if isinstance(p, str) else None)
    return abs_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    dataset_name = cfg['dataset']['name']
    artifacts_dir = Path(cfg['dataset']['artifacts_dir'])
    split_csv = cfg['dataset'][args.split]

    adapter = build_adapter(cfg, split_csv)
    df = adapter.df.copy()

    # ===== Labels =====
    y = adapter.get_labels()
    if y.dtype == object:
        y = y.astype(str).str.lower().map({'clickbait': 1, 'non-clickbait': 0, '0': 0, '1': 1}).fillna(y)
        try:
            y = y.astype(int)
        except Exception:
            pass

    out_dir = artifacts_dir / dataset_name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = {}

    # ===== TEXT (HF) =====
    txt_series = compose_text(df, adapter.colmap)
    if cfg.get('text_encoder', {}).get('enabled', True) and txt_series is not None:
        hf_name = cfg['text_encoder']['name']
        max_len = cfg['text_encoder'].get('max_length', 128)
        bs = cfg['text_encoder'].get('batch_size', 16)
        device = cfg['runtime'].get('device', 'auto')

        hf = HFTextEncoder(hf_name, max_length=max_len, device=device)  # KHÔNG truyền batch_size vào __init__
        texts = txt_series.fillna('').astype(str).tolist()
        vecs = []
        for i in tqdm(range(0, len(texts), bs), desc="HFText encode", leave=False):
            chunk = texts[i:i + bs]
            v = hf.encode(chunk)  # encode(list[str]) -> np.ndarray/torch.Tensor [B, D]
            try:
                v = v.cpu().numpy()
            except Exception:
                pass
            vecs.append(v.astype(np.float32))
        feats['text'] = np.concatenate(vecs, axis=0)

    # ===== CAPTION (HF optional) =====
    cap_col = adapter.colmap.caption
    if cap_col and cap_col in df.columns and cfg.get('caption_encoder', {}).get('enabled', False):
        cap_name = cfg['caption_encoder']['name']
        cap_max = cfg['caption_encoder'].get('max_length', 64)
        cap_bs = cfg['caption_encoder'].get('batch_size', 16)
        device = cfg['runtime'].get('device', 'auto')

        cap_hf = HFTextEncoder(cap_name, max_length=cap_max, device=device)
        caps = df[cap_col].fillna('').astype(str).tolist()
        cv = []
        for i in tqdm(range(0, len(caps), cap_bs), desc="Caption HF encode", leave=False):
            v = cap_hf.encode(caps[i:i + cap_bs])
            try:
                v = v.cpu().numpy()
            except Exception:
                pass
            cv.append(v.astype(np.float32))
        feats['caption'] = np.concatenate(cv, axis=0)

    # ===== IMAGE (raw images) =====
    if cfg.get('image_encoder', {}).get('enabled', False):
        claim_col = adapter.colmap.claim_image
        doc_col = adapter.colmap.document_image

        # Ưu tiên image_encoder.root, nếu không có dùng dataset.image_root
        img_root = Path(cfg['image_encoder'].get('root') or cfg['dataset'].get('image_root', '.'))
        enc_type = cfg['image_encoder'].get('type', 'timm')  # 'timm' | 'multi'
        image_size = cfg['image_encoder'].get('image_size', 380 if enc_type == 'timm' else 224)

        # Tìm cột chứa đường ảnh
        paths = []
        for col in [claim_col, doc_col, 'img_path']:
            if isinstance(col, str) and col in df.columns:
                paths = df[col].astype(str).tolist()
                break

        if paths:
            abs_paths = _join_root(paths, img_root)
            if enc_type == 'multi':
                names = cfg['image_encoder'].get('names', ["timm_efficientnet_b4", "openclip_ViT-B-32"])
                enc = MultiImageEncoder(names=names, image_size=image_size)
                feats['image_multi'] = enc.transform(abs_paths)  # np.float32 [N, D]
            else:
                enc = EfficientNetEncoder(
                    model_name=cfg['image_encoder'].get('name', 'tf_efficientnet_b4'),
                    image_size=image_size,
                    device=cfg['runtime'].get('device', 'auto')
                )
                feats['image_effnet'] = enc.encode_paths(abs_paths)

    # ===== CLIP (text/image) =====
    if cfg.get('clip', {}).get('enabled', False):
        clip_raw_name = cfg['clip']['name']
        clip_model_id = _normalize_clip_name(clip_raw_name)
        device = cfg['runtime'].get('device', 'auto')

        if cfg['clip'].get('text_enabled', True):
            # Ưu tiên caption cho CLIP text; fallback về text
            cap_col = adapter.colmap.caption
            if cap_col and cap_col in df.columns:
                txt_for_clip = df[cap_col].fillna('').astype(str).tolist()
            else:
                txt_for_clip = txt_series.fillna('').astype(str).tolist()
            feats['text_clip'] = CLIPTextEncoder(clip_model_id, device=device).encode(txt_for_clip)

        if cfg['clip'].get('image_enabled', True):
            paths = []
            for col in [adapter.colmap.claim_image, adapter.colmap.document_image, 'img_path']:
                if isinstance(col, str) and col in df.columns:
                    paths = df[col].astype(str).tolist()
                    break
            if paths:
                clip_root = Path(cfg['clip'].get('root') or cfg['dataset'].get('image_root', '.'))
                abs_paths = _join_root(paths, clip_root)
                feats['image_clip'] = CLIPImageEncoder(clip_model_id, device=device).encode(abs_paths)

    # ===== PRECOMPUTED (nếu dùng cho CLDI) =====
    pre = cfg.get('precomputed', {})
    if pre.get('enabled', False):
        rng = pre.get('image_range')  # [start, end] inclusive
        if rng:
            start, end = rng
            matrices = []
            for i in range(start, end + 1):
                cstr = str(i)
                if cstr in df.columns:
                    matrices.append(df[cstr].astype(np.float32).to_numpy())
                elif i in df.columns:
                    matrices.append(df[i].astype(np.float32).to_numpy())
            if matrices:
                Ximg = np.stack(matrices, axis=1)
                feats['image_precomp'] = Ximg.astype(np.float32)

    # ===== SAVE =====
    np.save(out_dir / "y.npy", y)
    for key, arr in feats.items():
        if arr is not None:
            print(f"[save] {key}: {getattr(arr, 'shape', None)} {getattr(arr, 'dtype', None)}")
            np.save(out_dir / f"X_{key}.npy", arr)

    meta = dict(num_samples=len(df), label_name=adapter.colmap.label, ids=df.index.tolist())
    json.dump(meta, open(out_dir / "meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
