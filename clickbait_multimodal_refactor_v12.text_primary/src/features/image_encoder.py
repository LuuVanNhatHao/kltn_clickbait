# src/features/image_encoder.py
import os
from typing import List, Optional
import numpy as np
from PIL import Image, ImageFile

# Tránh lỗi "Truncated File"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_image(path_or_none: Optional[str]) -> Image.Image:
    """Load ảnh (RGB). Nếu đường dẫn không hợp lệ, trả về ảnh xám 224x224."""
    if not isinstance(path_or_none, str) or not os.path.exists(path_or_none):
        return Image.new("RGB", (224, 224), (128, 128, 128))
    return Image.open(path_or_none).convert("RGB")


# -------------------------
# Fallback encoder: ResNet18
# -------------------------
class _TorchvisionResNet18:
    """
    Fallback encoder dùng torchvision ResNet18 (pretrained).
    Trả về vector 512-d, đã qua Identity FC.
    """
    def __init__(self, image_size: int = 224):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device).eval()
        # Transform chuẩn theo weights (bao gồm resize & normalize)
        self.transform = weights.transforms()
        self.dim = 512
        self.image_size = image_size

    def encode(self, pil_img: Image.Image) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = self.transform(pil_img).unsqueeze(0).to(self.device)  # [1,3,H,W]
            v = self.model(x)  # [1,512]
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
            return v.squeeze(0).float().cpu().numpy()


# -------------------------------------
# EfficientNet B4 encoder (timm backbone)
# -------------------------------------
class _TimmEffNetB4:
    """
    Encoder ảnh dùng timm EfficientNet (mặc định 'tf_efficientnet_b4').
    Nếu timm không có sẵn -> fallback ResNet18.
    """
    def __init__(self, image_size: int = 380, model_name: str = "tf_efficientnet_b4"):
        self.image_size = image_size
        self.dim = 1792  # default cho EfficientNet-B4
        self._ok = False

        try:
            import torch
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # num_classes=0 -> trả về embedding (global pooled)
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.model = self.model.to(self.device).eval()

            # Lấy transform chuẩn theo backbone
            cfg = resolve_data_config({}, model=self.model)
            # Ghi đè kích thước nếu người dùng yêu cầu
            if self.image_size:
                cfg["input_size"] = (3, self.image_size, self.image_size)
            self.transform = create_transform(**cfg)

            # Cập nhật dim nếu model có thuộc tính num_features
            if hasattr(self.model, "num_features"):
                self.dim = int(self.model.num_features)

            self._ok = True
        except Exception:
            # Fallback nếu thiếu timm/torch hoặc lỗi khởi tạo
            self.fallback = _TorchvisionResNet18(image_size=image_size)
            self.dim = self.fallback.dim

    def encode(self, pil_img: Image.Image) -> np.ndarray:
        if self._ok:
            import torch
            with torch.no_grad():
                x = self.transform(pil_img).unsqueeze(0).to(self.device)  # [1,3,H,W]
                v = self.model(x)  # [1,D]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
                return v.squeeze(0).float().cpu().numpy()
        else:
            return self.fallback.encode(pil_img)


# ----------------------------
# OpenCLIP ViT-B/32 image side
# ----------------------------
class _OpenCLIP_B32:
    """
    Image encoder dùng OpenCLIP (ViT-B-32). Nếu không có open-clip-torch, fallback ResNet18.
    """
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.dim = 512  # embedding size của CLIP ViT-B/32 (image tower)
        self._ok = False
        try:
            import torch
            import open_clip

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # create_model_and_transforms trả về (model, preprocess, tokenizer)
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.model = self.model.to(self.device).eval()
            self._ok = True
        except Exception:
            self.fallback = _TorchvisionResNet18(image_size=image_size)
            self.dim = self.fallback.dim

    def encode(self, pil_img: Image.Image) -> np.ndarray:
        if self._ok:
            import torch
            with torch.no_grad():
                x = self.preprocess(pil_img).unsqueeze(0).to(self.device)  # [1,3,H,W]
                v = self.model.encode_image(x)  # [1,512]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
                return v.squeeze(0).float().cpu().numpy()
        else:
            return self.fallback.encode(pil_img)


# --------------------------------------
# Multi-image encoder (concat các backbone)
# --------------------------------------
class MultiImageEncoder:
    """
    Kết hợp nhiều encoder ảnh và concat đặc trưng: tên hợp lệ:
      - "timm_efficientnet_b4"
      - "openclip_ViT-B-32"
      - tên khác -> fallback ResNet18
    """
    def __init__(self, names, image_size: int = 224):
        names = names if isinstance(names, list) else [names]
        self.encoders = []
        self.image_size = image_size
        for n in names:
            if n == "timm_efficientnet_b4":
                self.encoders.append(_TimmEffNetB4(image_size=image_size, model_name="tf_efficientnet_b4"))
            elif n == "openclip_ViT-B-32":
                self.encoders.append(_OpenCLIP_B32(image_size=image_size))
            else:
                self.encoders.append(_TorchvisionResNet18(image_size=image_size))
        self.dim = sum(e.dim for e in self.encoders)

    def transform(self, image_paths: List[Optional[str]]) -> np.ndarray:
        feats = []
        for p in image_paths:
            img = _load_image(p)
            vecs = [enc.encode(img) for enc in self.encoders]
            feats.append(np.concatenate(vecs, axis=0))
        return np.stack(feats, axis=0).astype(np.float32)


# ----------------------------------------------------
# Wrapper để khớp với import: EfficientNetEncoder class
# ----------------------------------------------------
class EfficientNetEncoder:
    """
    Wrapper tên 'EfficientNetEncoder' để scripts/extract_features.py import được.
    - Bọc _TimmEffNetB4 (mặc định tf_efficientnet_b4, image_size=380).
    - API:
        encode_paths(paths, root=None, batch_size=16) -> np.ndarray [N, D]
        encode_images(...) alias của encode_paths
    """
    def __init__(self, model_name: str = "tf_efficientnet_b4", image_size: int = 380, device: Optional[str] = None):
        # _TimmEffNetB4 tự xử lý device nội bộ; tham số device để tương thích chữ ký
        self.backbone = _TimmEffNetB4(image_size=image_size, model_name=model_name)
        self.dim = self.backbone.dim
        self.image_size = image_size

    def encode_paths(self, paths: List[Optional[str]], root: Optional[str] = None, batch_size: int = 16) -> np.ndarray:
        from tqdm.auto import tqdm
        feats = []
        for p in tqdm(paths, desc="EfficientNet (timm) encode", leave=False):
            fp = os.path.join(root, p) if root and isinstance(p, str) and not os.path.isabs(p) else p
            img = _load_image(fp)
            v = self.backbone.encode(img)
            feats.append(v)
        return np.stack(feats, axis=0).astype(np.float32)

    # alias cho tên cũ
    def encode_images(self, paths: List[Optional[str]], root: Optional[str] = None, batch_size: int = 16) -> np.ndarray:
        return self.encode_paths(paths, root=root, batch_size=batch_size)
