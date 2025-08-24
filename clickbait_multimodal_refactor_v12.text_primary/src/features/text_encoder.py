import numpy as np
from typing import List, Optional

def _maybe_vi_segment(text: str, method: Optional[str] = None) -> str:
    if not isinstance(text, str):
        return ""
    if not method:
        return text
    method = method.lower()
    try:
        if method == "underthesea":
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
    except Exception:
        pass
    return text

class HFTextEncoder:
    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 256, vi_segmenter: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.vi_segmenter = vi_segmenter
        self.dim = 768
        self._ok = False
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.dim = int(self.model.config.hidden_size)
            self._ok = True
        except Exception:
            # fall back to simple hashing vectorizer
            from sklearn.feature_extraction.text import HashingVectorizer
            self.hash = HashingVectorizer(n_features=2048, alternate_sign=False)
            self.dim = 2048

    def transform(self, texts: List[Optional[str]]) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        if self._ok:
            import torch
            outs = []
            for t in texts:
                t0 = _maybe_vi_segment(t, self.vi_segmenter)
                enc = self.tok(t0, truncation=True, max_length=self.max_length, return_tensors="pt")
                with torch.no_grad():
                    h = self.model(**enc).last_hidden_state  # [1, L, H]
                    cls = h[:,0,:]                            # [1, H]
                outs.append(cls.squeeze(0).numpy())
            return np.stack(outs, axis=0).astype(np.float32)
        else:
            return self.hash.transform(texts).toarray().astype(np.float32)
