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
    def __init__(self, csv_path: str, column_map: ColumnMap, image_root: Optional[str]=None, dataset_name: Optional[str]=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.cm = column_map
        self.image_root = image_root
        self.dataset_name = (dataset_name or "").lower()

        # Normalize labels per dataset
        if self.cm.label not in self.df.columns:
            raise ValueError(f"Label column '{self.cm.label}' not found in {csv_path}")
        self.y = self._normalize_labels(self.df[self.cm.label].values)

        # Normalize text/caption fields that might be stored as JSON-like lists
        if self.cm.text and self.cm.text in self.df.columns:
            self.df[self.cm.text] = self.df[self.cm.text].apply(self._first_from_listlike)
        if self.cm.caption and self.cm.caption in self.df.columns:
            self.df[self.cm.caption] = self.df[self.cm.caption].apply(self._first_from_listlike)

        # Resolve image paths if present
        for img_col in [self.cm.claim_image, self.cm.document_image]:
            if img_col and img_col in self.df.columns:
                self.df[img_col] = self.df[img_col].apply(lambda p: self._resolve_image_path(p))

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

    def _resolve_image_path(self, p: Any) -> str:
        if pd.isna(p):
            return ""
        p = str(p)
        if self.image_root and not os.path.isabs(p):
            # Handle cases where column is "['media/xxx.jpg']"
            if p.startswith('[') and p.endswith(']'):
                try:
                    arr = ast.literal_eval(p)
                    if isinstance(arr, list) and arr:
                        p = arr[0]
                except Exception:
                    pass
            p = os.path.join(self.image_root, p) if not p.startswith(self.image_root) else p
        return p

    def _normalize_labels(self, y):
        # Convert to {non-clickbait:0, clickbait:1} for VCC/WCC; leave ints for CLDI
        if self.dataset_name in ["vcc", "wcc"]:
            y_norm = []
            for v in y:
                if isinstance(v, str):
                    vs = v.strip().lower()
                    if "clickbait" in vs and "non" not in vs:
                        y_norm.append(1)
                    elif "non" in vs or "not" in vs:
                        y_norm.append(0)
                    else:
                        # fallback: try cast
                        try:
                            y_norm.append(int(float(v)))
                        except Exception:
                            y_norm.append(0)
                else:
                    y_norm.append(int(v))
            return np.array(y_norm, dtype=np.int64)
        # CLDI already has 0/1 in Clickbait
        return np.array(y, dtype=np.int64)

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
