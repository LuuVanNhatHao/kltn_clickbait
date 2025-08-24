from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
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
    def __init__(self, csv_path, column_map, cfg):
        self.cm = column_map
        self.cfg = cfg or {}
        self.df = pd.read_csv(csv_path)
        # chuẩn hoá label -> 0/1 (đoạn bạn đã dán trước đó giữ nguyên)

        # ---- xác định image_root từ image_encoder hoặc clip ----
        img_root = None
        if isinstance(self.cfg.get("image_encoder"), dict):
            img_root = self.cfg["image_encoder"].get("root")
        if not img_root and isinstance(self.cfg.get("clip"), dict):
            img_root = self.cfg["clip"].get("root")
        self.image_root = img_root  # string hoặc None

        # ---- map label ----
        self.y = self._normalize_labels(self.df[self.cm.label].values)

        # ---- chuẩn hoá cột ảnh (nếu có) ----
        img_col = self.cm.claim_image
        if img_col and img_col in self.df.columns:
            self.df[img_col] = self.df[img_col].apply(self._resolve_image_path)


    def _first_from_listlike(self, x):
        # WCC sometimes stores lists as strings like "['text']"
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
        - Hỗ trợ: str, list/tuple (lấy phần tử đầu), chuỗi dạng "['...']" từ CSV.
        - Nếu relative và có image_root -> join vào.
        - Nếu NaN/None -> trả None.
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

            # Nếu string trông như list JSON/py: "['media/a.jpg']" -> parse
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        s = str(val[0])
                except Exception:
                    # fallback: cố bóc ký tự rác
                    s = re.sub(r"^[\[\(\s'\" ]+|[\]\)\s'\" ]+$", "", s)

            # bóc quote
            s = s.strip().strip("'").strip('"')

            # nếu có image_root và s chưa chứa root thì join
            if self.image_root and isinstance(self.image_root, str):
                # chuẩn hoá để so sánh
                root_norm = os.path.normpath(self.image_root)
                s_norm = os.path.normpath(s)
                if not s_norm.startswith(root_norm) and not os.path.isabs(s_norm):
                    s = os.path.join(self.image_root, s)

            return os.path.normpath(s)

        # loại còn lại -> cast str
        s = str(p)
        if self.image_root and isinstance(self.image_root, str):
            return os.path.normpath(os.path.join(self.image_root, s))
        return os.path.normpath(s)

    def _normalize_labels(self, y):
        """
        Chuẩn hoá nhãn về {0,1}:
          - Hỗ trợ chuỗi: clickbait / non-clickbait / no-clickbait (mọi kiểu viết, có/không gạch nối/khoảng trắng)
          - Hỗ trợ số: 0/1, "0"/"1", 0.0/1.0
          - Nếu gặp giá trị lạ -> raise lỗi gợi ý.
        """
        import re
        import numpy as np

        pos_set = {
            "clickbait", "cb", "bait", "yes", "true", "1"
        }
        neg_set = {
            "nonclickbait", "noclickbait", "no-clickbait", "non-clickbait", "nocb",
            "noncb", "nobait", "no", "false", "0"
        }

        out = []
        for v in y:
            # số nguyên
            if isinstance(v, (int, np.integer)):
                out.append(1 if int(v) != 0 else 0)
                continue
            # số thực
            if isinstance(v, (float, np.floating)):
                out.append(1 if float(v) >= 0.5 else 0)
                continue
            # chuỗi
            if isinstance(v, str):
                s = v.strip().lower()
                # bỏ khoảng trắng/gạch nối/gạch dưới để gom các biến thể: "non-clickbait" -> "nonclickbait"
                s_clean = re.sub(r"[\s\-_]+", "", s)

                if s_clean in pos_set:
                    out.append(1)
                    continue
                if s_clean in neg_set:
                    out.append(0)
                    continue
                # Thử parse số trong chuỗi
                try:
                    f = float(s)
                    out.append(1 if f >= 0.5 else 0)
                    continue
                except Exception:
                    pass

                raise ValueError(
                    f"Unrecognized label '{v}'. Expected one of {sorted(list(pos_set | neg_set))} or numeric 0/1."
                )

            # kiểu khác không hỗ trợ
            raise ValueError(f"Unsupported label type: {type(v)} with value={v}")

        return np.array(out, dtype=np.int64)

    def labels(self) -> np.ndarray:
        return self.y

    def get_text(self) -> List[str]:
        if not self.cm.text:
            return []
        return self.df[self.cm.text].fillna("").astype(str).tolist()

    def get_caption(self) -> List[str]:
        if not self.cm.caption:
            return []
        return self.df[self.cm.caption].fillna("").astype(str).tolist()

    def get_image_paths(self) -> List[str]:
        col = self.cm.claim_image or self.cm.document_image
        if not col:
            return []
        return self.df[col].fillna("").astype(str).tolist()

    def get_tabular(self, start: int, end: int) -> np.ndarray:
        cols = [str(i) for i in range(start, end+1)]
        # Some CLDI csvs may have integer column names; coerce
        mapped = []
        for c in cols:
            if c in self.df.columns:
                mapped.append(c)
            else:
                try:
                    ic = int(c)
                    if ic in self.df.columns:
                        mapped.append(ic)
                except Exception:
                    pass
        if not mapped:
            raise ValueError("No matching feature columns found for the given range")
        return self.df[mapped].to_numpy(dtype=np.float32)
