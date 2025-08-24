# src/features/clip_encoder.py
from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _resolve_device(device: Optional[str]) -> str:
    import torch
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def _l2norm(x):
    import torch
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

class CLIPTextEncoder:
    def __init__(self, name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = "auto"):
        import open_clip, torch
        self.device = _resolve_device(device)
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            name, pretrained=(pretrained or "laion2b_s34b_b79k"), device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)
        with torch.no_grad():
            tok = self.tokenizer(["hi"]).to(self.device)
            emb = self.model.encode_text(tok)
            self.dim = int(emb.shape[-1])

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tok = self.tokenizer(list(texts[i:i+batch_size])).to(self.device)
                emb = self.model.encode_text(tok)
                emb = _l2norm(emb).float().cpu().numpy().astype(np.float32)
                outs.append(emb)
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.dim), dtype=np.float32)

class CLIPImageEncoder:
    def __init__(self, name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = "auto"):
        import open_clip, torch
        self.device = _resolve_device(device)
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            name, pretrained=(pretrained or "laion2b_s34b_b79k"), device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            from PIL import Image
            dummy = Image.new("RGB", (224,224), (128,128,128))
            x = self.preprocess_val(dummy).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(x)
            self.dim = int(emb.shape[-1])

    def _load(self, p: Optional[str]):
        from PIL import Image
        import os
        if not isinstance(p, str) or not os.path.exists(p):
            return Image.new("RGB", (224,224), (128,128,128))
        return Image.open(p).convert("RGB")

    def encode(self, paths: Sequence[Optional[str]], batch_size: int = 32) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                imgs = [self._load(p) for p in paths[i:i+batch_size]]
                x = torch.stack([self.preprocess_val(im) for im in imgs], 0).to(self.device)
                emb = self.model.encode_image(x)
                emb = _l2norm(emb).float().cpu().numpy().astype(np.float32)
                outs.append(emb)
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.dim), dtype=np.float32)
