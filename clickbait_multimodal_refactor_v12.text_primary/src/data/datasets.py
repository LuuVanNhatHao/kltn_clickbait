from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os, re, ast

@dataclass
class ColumnMap:
    # Một trong hai: text hoặc text_fields
    text: Optional[str] = None
    text_fields: Optional[List[str]] = None
    # Nhãn
    label: str = "label"
    # Ảnh
    claim_image: Optional[str] = None
    document_image: Optional[str] = None
    # Caption sinh từ ảnh (nếu có)
    caption: Optional[str] = None

class DatasetAdapter:
    """
    Adapter hợp nhất cho 3 bộ: VCC, WCC, CLDI
    - VCC: text='title', caption='caption_vi', label='label', claim_image='thumbnail_url'
    - WCC: text='postText'(list-like), caption='caption_en', label='truthClass', claim_image='img_path'
    - CLDI: text_fields=['Captions','Hashtags'], label='Clickbait', (không có ảnh gốc; dùng cột 0..2047 nếu cần)
    """
    def __init__(self, csv_path: str, column_map: ColumnMap, cfg: Optional[Dict[str, Any]] = None):
        self.cm = column_map
        self.cfg = cfg or {}

        # Đọc CSV
        self.df = pd.read_csv(csv_path)

        # --- xác định image_root từ config ---
        self.image_root = None
        try:
            img_enc = self.cfg.get("image_encoder", None)
            if isinstance(img_enc, dict) and isinstance(img_enc.get("root", None), str):
                self.image_root = img_enc["root"]
            clip = self.cfg.get("clip", None)
            if not self.image_root and isinstance(clip, dict) and isinstance(clip.get("root", None), str):
                self.image_root = clip["root"]
        except Exception:
            # an toàn: để None
            self.image_root = None

        # --- chuẩn hoá nhãn -> 0/1 ---
        self.y = self._normalize_labels(self.df[self.cm.label].values)

        # --- chuẩn hoá cột ảnh nếu có ---
        img_col = self.cm.claim_image or self.cm.document_image
        if img_col and img_col in self.df.columns:
            self.df[img_col] = self.df[img_col].apply(self._resolve_image_path)

    # =======================
    # Helpers
    # =======================
    def _first_from_listlike(self, x):
        """
        WCC lưu nhiều trường dạng list-string: "['text a', 'text b']".
        Trả về phần tử đầu tiên (hoặc chuỗi rỗng nếu không có).
        """
        if isinstance(x, list):
            return x[0] if len(x) else ""
        if isinstance(x, str) and ((x.startswith("[") and x.endswith("]")) or (x.startswith("(") and x.endswith(")"))):
            try:
                arr = ast.literal_eval(x)
                if isinstance(arr, (list, tuple)) and len(arr) > 0:
                    return arr[0]
            except Exception:
                # fallback: bóc ký tự rác
                s = x.strip().strip("[]()").strip().strip("'").strip('"')
                return s
        return x if isinstance(x, str) else str(x)

    def _resolve_image_path(self, p):
        """
        Chuẩn hoá path ảnh cho VCC/WCC:
        - Hỗ trợ str, list/tuple, list-string.
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

        # parse list-string
        if isinstance(p, str) and ((p.strip().startswith("[") and p.strip().endswith("]")) or
                                   (p.strip().startswith("(") and p.strip().endswith(")"))):
            try:
                val = ast.literal_eval(p)
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    p = val[0]
            except Exception:
                p = p.strip().strip("[]()").strip().strip("'").strip('"')

        if not isinstance(p, str):
            p = str(p)

        s = p.strip().strip("'").strip('"')
        if self.image_root and isinstance(self.image_root, str):
            s_norm = os.path.normpath(s)
            root_norm = os.path.normpath(self.image_root)
            # nếu s không absolute và chưa chứa root -> join
            if not os.path.isabs(s_norm) and not s_norm.startswith(root_norm):
                s = os.path.join(self.image_root, s)
        return os.path.normpath(s)

    def _normalize_labels(self, y):
        """
        Chuẩn hoá nhãn {0,1}:
          - string: 'clickbait' -> 1; 'non-clickbait','no-clickbait','non clickbait'... -> 0
          - số: 0/1 hoặc float -> ngưỡng 0.5
        """
        pos_set = {
            "clickbait", "cb", "bait", "yes", "true", "1"
        }
        neg_set = {
            "nonclickbait", "noclickbait", "no-clickbait", "non-clickbait",
            "no clickbait", "nocb", "noncb", "nobait", "no", "false", "0"
        }

        out = []
        for v in y:
            # int
            if isinstance(v, (int, np.integer)):
                out.append(1 if int(v) != 0 else 0)
                continue
            # float
            if isinstance(v, (float, np.floating)):
                out.append(1 if float(v) >= 0.5 else 0)
                continue
            # str
            if isinstance(v, str):
                s = v.strip().lower()
                s_clean = re.sub(r"[\s\-_]+", "", s)  # gom mọi biến thể
                if s_clean in pos_set:
                    out.append(1);
                    continue
                if s_clean in neg_set:
                    out.append(0);
                    continue
                # cố gắng parse số
                try:
                    f = float(s)
                    out.append(1 if f >= 0.5 else 0)
                    continue
                except Exception:
                    pass
                raise ValueError(
                    f"Unrecognized label '{v}'. Hỗ trợ {sorted(list(pos_set | neg_set))} hoặc số 0/1."
                )
            # khác
            raise ValueError(f"Unsupported label type: {type(v)} with value={v}")
        return np.array(out, dtype=np.int64)

    # =======================
    # Public API
    # =======================
    def labels(self) -> np.ndarray:
        return self.y

    # alias để tương thích script cũ
    def get_labels(self) -> np.ndarray:
        return self.labels()

    def get_text(self) -> List[str]:
        """
        Trả về mảng chuỗi:
          - Nếu khai báo text_fields (CLDI): nối các trường bằng ' [SEP] '
          - Nếu khai báo text (VCC/WCC): trả về cột đó (đã tách list-string nếu cần)
        """
        # nhiều trường -> concat
        if self.cm.text_fields:
            parts = []
            for col in self.cm.text_fields:
                if col not in self.df.columns:
                    parts.append([""] * len(self.df))
                    continue
                series = self.df[col].fillna("").astype(str).tolist()
                # bóc phần tử đầu nếu là list-string
                series = [self._first_from_listlike(s) for s in series]
                parts.append(series)
            # zip rồi ghép với [SEP]
            merged = [" [SEP] ".join([p[i] for p in parts]) for i in range(len(self.df))]
            return merged

        # 1 trường
        if self.cm.text and self.cm.text in self.df.columns:
            series = self.df[self.cm.text].fillna("").astype(str).tolist()
            return [self._first_from_listlike(s) for s in series]
        return []

    def get_caption(self) -> List[str]:
        """
        Caption sinh từ ảnh (đã được cung cấp sẵn trong CSV: VCC: caption_vi, WCC: caption_en).
        CLDI không có caption sinh từ ảnh.
        """
        if not self.cm.caption or self.cm.caption not in self.df.columns:
            return []
        series = self.df[self.cm.caption].fillna("").astype(str).tolist()
        return [self._first_from_listlike(s) for s in series]

    def get_image_paths(self) -> List[Optional[str]]:
        """
        Trả về path ảnh đã chuẩn hoá; nếu ô rỗng -> None.
        """
        col = self.cm.claim_image or self.cm.document_image
        if not col or col not in self.df.columns:
            return []
        vals = self.df[col].tolist()
        out = []
        for v in vals:
            p = self._resolve_image_path(v) if isinstance(v, (str, list, tuple)) or not (
                        isinstance(v, float) and np.isnan(v)) else None
            out.append(p if p and len(str(p)) > 0 else None)
        return out

    def get_tabular(self, start: int, end: int) -> np.ndarray:
        """
        Lấy ma trận đặc trưng số theo dải cột tên 'start'..'end'.
        Hỗ trợ cả tên cột là số (int) hoặc chuỗi số.
        """
        # dựng danh sách cột mong muốn
        want = list(range(start, end + 1))
        mapped = []
        for i in want:
            # tên chuỗi
            s = str(i)
            if s in self.df.columns:
                mapped.append(s);
                continue
            # tên số nguyên thật
            if i in self.df.columns:
                mapped.append(i);
                continue
        if not mapped:
            raise ValueError(f"No matching feature columns found in range [{start}, {end}]")
        return self.df[mapped].to_numpy(dtype=np.float32)

