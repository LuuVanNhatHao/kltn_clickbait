from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import re
import numpy as np
import os, json, ast

@dataclass
class ColumnMap:
    text: Optional[str] = None
    text_fields: Optional[List[str]] = None
    label: str = "label"
    claim_image: Optional[str] = None
    document_image: Optional[str] = None
    caption: Optional[str] = None

class DatasetAdapter:
    def __init__(self, csv_path, column_map: ColumnMap, cfg: Optional[Dict[str, Any]]):
        self.cm = column_map
        self.colmap = self.cm            # <— alias để tương thích script cũ
        self.cfg = cfg or {}
        self.df = pd.read_csv(csv_path)

        # --------- xác định image_root ---------
        img_root = None
        # Nếu cfg là full YAML, các khối nằm ở cấp 1
        if isinstance(self.cfg.get("image_encoder"), dict):
            img_root = self.cfg["image_encoder"].get("root")
        if not img_root and isinstance(self.cfg.get("clip"), dict):
            img_root = self.cfg["clip"].get("root")
        # Nếu vẫn None, suy luận theo cột phổ biến
        if not img_root:
            if "img_path" in self.df.columns:
                img_root = "data/media"
            elif "thumbnail_url" in self.df.columns:
                img_root = "data/images"
        self.image_root = img_root  # string hoặc None

        # --------- chuẩn hoá label -> 0/1 ---------
        self.y = self._normalize_labels(self.df[self.cm.label].values)

        # --------- chuẩn hoá cột ảnh (nếu có) ---------
        img_col = self.cm.claim_image or self.cm.document_image
        if img_col and img_col in self.df.columns:
            self.df[img_col] = self.df[img_col].apply(self._resolve_image_path)

    # ------------ helpers ------------
    def _first_from_listlike(self, x):
        # WCC: nhiều trường lưu dạng "['text']"
        if isinstance(x, list):
            return x[0] if len(x) else ""
        if isinstance(x, str) and (x.startswith('[') and x.endswith(']')):
            try:
                arr = ast.literal_eval(x)
                if isinstance(arr, list) and arr:
                    return arr[0]
            except Exception:
                pass
        return x if isinstance(x, str) else str(x)

    def _resolve_image_path(self, p):
        """
        Chuẩn hoá path ảnh:
        - Hỗ trợ str, list/tuple (lấy phần tử đầu), list-string "['...']".
        - Nếu relative và có image_root -> join.
        - NaN/None -> None.
        """
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return None

        # list/tuple
        if isinstance(p, (list, tuple)):
            p = p[0] if len(p) > 0 else None
            if p is None:
                return None

        # chuỗi
        if isinstance(p, str):
            s = p.strip()

            # parse list-string
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        s = str(val[0])
                except Exception:
                    s = re.sub(r"^[\[\(\s'\" ]+|[\]\)\s'\" ]+$", "", s)

            s = s.strip().strip("'").strip('"')

            if self.image_root and isinstance(self.image_root, str):
                root_norm = os.path.normpath(self.image_root)
                s_norm = os.path.normpath(s)
                if not s_norm.startswith(root_norm) and not os.path.isabs(s_norm):
                    s = os.path.join(self.image_root, s)

            return os.path.normpath(s)

        # kiểu khác -> cast str + join root
        s = str(p)
        if self.image_root and isinstance(self.image_root, str):
            return os.path.normpath(os.path.join(self.image_root, s))
        return os.path.normpath(s)

    def _normalize_labels(self, y):
        """
        Chuẩn hoá nhãn về {0,1}:
          - string: clickbait -> 1; non-clickbait/no-clickbait/... -> 0
          - số: 0/1; float -> ngưỡng 0.5
        """
        pos_set = {"clickbait", "cb", "bait", "yes", "true", "1"}
        neg_set = {
            "nonclickbait", "noclickbait", "no-clickbait", "non-clickbait",
            "nocb", "noncb", "nobait", "no", "false", "0", "no clickbait"
        }

        out = []
        for v in y:
            if isinstance(v, (int, np.integer)):
                out.append(1 if int(v) != 0 else 0); continue
            if isinstance(v, (float, np.floating)):
                out.append(1 if float(v) >= 0.5 else 0); continue
            if isinstance(v, str):
                s = v.strip().lower()
                s_clean = re.sub(r"[\s\-_]+", "", s)
                if s_clean in pos_set: out.append(1); continue
                if s_clean in neg_set: out.append(0); continue
                try:
                    f = float(s)
                    out.append(1 if f >= 0.5 else 0); continue
                except Exception:
                    pass
                raise ValueError(
                    f"Unrecognized label '{v}'. Hỗ trợ {sorted(list(pos_set | neg_set))} hoặc 0/1."
                )
            raise ValueError(f"Unsupported label type: {type(v)} with value={v}")
        return np.array(out, dtype=np.int64)

    # ------------ public API ------------
    def labels(self) -> np.ndarray:
        return self.y

    def get_labels(self) -> np.ndarray:  # alias cho script cũ
        return self.labels()

    def get_text(self) -> List[str]:
        """
        - Nếu có text_fields (CLDI): nối bằng ' [SEP] '
        - Nếu có text (VCC/WCC): trả về cột đó (tự bóc list-string nếu cần)
        """
        if self.cm.text_fields:
            parts = []
            for col in self.cm.text_fields:
                if col not in self.df.columns:
                    parts.append([""] * len(self.df)); continue
                series = self.df[col].fillna("").astype(str).tolist()
                series = [self._first_from_listlike(s) for s in series]
                parts.append(series)
            merged = [" [SEP] ".join([p[i] for p in parts]) for i in range(len(self.df))]
            return merged

        if self.cm.text and self.cm.text in self.df.columns:
            series = self.df[self.cm.text].fillna("").astype(str).tolist()
            return [self._first_from_listlike(s) for s in series]
        return []

    def get_caption(self) -> List[str]:
        if not self.cm.caption or self.cm.caption not in self.df.columns:
            return []
        series = self.df[self.cm.caption].fillna("").astype(str).tolist()
        return [self._first_from_listlike(s) for s in series]

    def get_image_paths(self) -> List[Optional[str]]:
        col = self.cm.claim_image or self.cm.document_image
        if not col or col not in self.df.columns:
            return []
        vals = self.df[col].tolist()
        out = []
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(None); continue
            p = self._resolve_image_path(v)
            out.append(p if p and len(str(p)) > 0 else None)
        return out

    def get_tabular(self, start: int, end: int) -> np.ndarray:
        want = list(range(start, end + 1))
        mapped = []
        for i in want:
            s = str(i)
            if s in self.df.columns: mapped.append(s); continue
            if i in self.df.columns: mapped.append(i); continue
        if not mapped:
            raise ValueError(f"No matching feature columns found in range [{start},{end}]")
        return self.df[mapped].to_numpy(dtype=np.float32)
