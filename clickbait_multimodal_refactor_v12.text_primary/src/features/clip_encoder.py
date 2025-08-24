# src/features/clip_encoder.py
from typing import List, Optional
import os
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm.auto import tqdm

class _CLIPBase:
    def __init__(self, name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(name)
        self.model.eval()

    @staticmethod
    def _l2(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

class CLIPTextEncoder(_CLIPBase):
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        outs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text", leave=False):
            chunk = texts[i:i+batch_size]
            toks = self.tokenizer(chunk).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_text(toks)
                feat = self._l2(feat)
            outs.append(feat.float().cpu())
        return torch.cat(outs, dim=0).numpy()

    # alias cho code cũ nếu gọi tên này
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size)

class CLIPImageEncoder(_CLIPBase):
    def encode_paths(self, paths: List[str], root: Optional[str] = None, batch_size: int = 16) -> np.ndarray:
        imgs = []
        for p in paths:
            fp = os.path.join(root, p) if root and not os.path.isabs(p) else p
            img = Image.open(fp).convert("RGB")
            imgs.append(self.preprocess(img))
        outs = []
        for i in tqdm(range(0, len(imgs), batch_size), desc="CLIP image", leave=False):
            batch = torch.stack(imgs[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(batch)
                feat = self._l2(feat)
            outs.append(feat.float().cpu())
        return torch.cat(outs, dim=0).numpy()

    # alias cho code cũ nếu gọi tên này
    def encode_images(self, paths: List[str], root: Optional[str] = None, batch_size: int = 16) -> np.ndarray:
        return self.encode_paths(paths, root=root, batch_size=batch_size)
