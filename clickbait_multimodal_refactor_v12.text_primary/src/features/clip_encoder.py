# src/features/clip_encoder.py
from __future__ import annotations
import os
from typing import List, Optional, Sequence
import numpy as np
from PIL import Image, ImageFile

# Tránh lỗi "Truncated File"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resolve_device(device: Optional[str]) -> str:
    import torch
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _l2norm(x):
    import torch
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


# -------------------------
# CLIP Text Encoder (OpenCLIP)
# -------------------------
class CLIPTextEncoder:
    """
    OpenCLIP text tower.
    - name: model id, ví dụ "ViT-B-32" (đối với open_clip).
    - pretrained: mặc định để open_clip tự chọn checkpoint phổ biến ("laion2b_s34b_b79k").
    - device: "auto" | "cpu" | "cuda" | "cuda:0" ...
    API:
        encode(texts: list[str], batch_size=32) -> np.ndarray [N, D]
    """
    def __init__(self, name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = "auto"):
        import torch, open_clip
        self.device = _resolve_device(device)
        # create_model_and_transforms trả về (model, preprocess_train, preprocess_val)
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            name, pretrained=(pretrained or "laion2b_s34b_b79k"), device=self.device
        )
        self.model.eval()
        # tokenizer cho OpenCLIP
        self.tokenizer = open_clip.get_tokenizer(name)
        # xác định dim
        with torch.no_grad():
            # Hack nhỏ: embed dummy để lấy dim
            tok = self.tokenizer(["hi"]).to(self.device)
            emb = self.model.encode_text(tok)
            self.dim = int(emb.shape[-1])

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                tok = self.tokenizer(list(chunk)).to(self.device)
                emb = self.model.encode_text(tok)        # [B, D]
                emb = _l2norm(emb).float().cpu().numpy() # L2-norm + về CPU
                outs.append(emb.astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, getattr(self, "dim", 512)), dtype=np.float32)


# -------------------------
# CLIP Image Encoder (OpenCLIP)
# -------------------------
class CLIPImageEncoder:
    """
    OpenCLIP image tower.
    - name: model id, ví dụ "ViT-B-32"
    - device: "auto" | "cpu" | "cuda" | "cuda:0"
    API:
        encode(image_paths: list[str], batch_size=32) -> np.ndarray [N, D]
    """
    def __init__(self, name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = "auto"):
        import torch, open_clip
        self.device = _resolve_device(device)
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            name, pretrained=(pretrained or "laion2b_s34b_b79k"), device=self.device
        )
        self.model.eval()
        # xác định dim
        with torch.no_grad():
            # tạo ảnh giả 224x224 (preprocess sẽ resize/crop lại phù hợp)
            dummy = Image.new("RGB", (224, 224), (128, 128, 128))
            x = self.preprocess_val(dummy).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(x)
            self.dim = int(emb.shape[-1])

    def _load_image(self, path_or_none: Optional[str]) -> Image.Image:
        if not isinstance(path_or_none, str) or not os.path.exists(path_or_none):
            return Image.new("RGB", (224, 224), (128, 128, 128))
        return Image.open(path_or_none).convert("RGB")

    def encode(self, image_paths: Sequence[Optional[str]], batch_size: int = 32) -> np.ndarray:
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                chunk = image_paths[i:i + batch_size]
                batch_imgs = [self._load_image(p) for p in chunk]
                x = torch.stack([self.preprocess_val(img) for img in batch_imgs], dim=0).to(self.device)  # [B,3,H,W]
                emb = self.model.encode_image(x)           # [B, D]
                emb = _l2norm(emb).float().cpu().numpy()   # L2-norm + về CPU
                outs.append(emb.astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, getattr(self, "dim", 512)), dtype=np.float32)
