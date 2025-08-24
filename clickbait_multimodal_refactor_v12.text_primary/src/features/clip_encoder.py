# src/features/clip_encoder.py
from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _resolve_device(device: Optional[str]) -> str:
    import torch
    if not device or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def _l2norm(x):
    import torch
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

class CLIPTextEncoder:
    def __init__(self, name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = "auto"):
        import torch, open_clip
        self.device = _resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=(pretrained or "laion2b_s34b_b79k"), device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)
        with torch.no_grad():
            tok = self.tokenizer(["hi"]).to(self.device)
            self.dim = int(self.model.encode_text(tok).shape[-1])

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tok = self.tokenizer(list(texts[i:i+batch_size])).to(self.device)
                emb = self.model.encode_text(tok)
                outs.append(_l2norm(emb).float().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, getattr(self, "dim", 512)), np.float32)

class CLIPImageEncoder:
    def __init__(self, name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = "auto"):
        import torch, open_clip
        self.device = _resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=(pretrained or "laion2b_s34b_b79k"), device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            dummy = Image.new("RGB", (224, 224), (128, 128, 128))
            emb = self.model.encode_image(self.preprocess(dummy).unsqueeze(0).to(self.device))
            self.dim = int(emb.shape[-1])

    def _load(self, p: Optional[str]) -> Image.Image:
        import os
        return Image.open(p).convert("RGB") if isinstance(p, str) and os.path.exists(p) \
               else Image.new("RGB", (224, 224), (128, 128, 128))

    def encode(self, image_paths: Sequence[Optional[str]], batch_size: int = 32) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                imgs = [self._load(p) for p in image_paths[i:i+batch_size]]
                x = torch.stack([self.preprocess(img) for img in imgs], dim=0).to(self.device)
                emb = self.model.encode_image(x)
                outs.append(_l2norm(emb).float().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, getattr(self, "dim", 512)), np.float32)
