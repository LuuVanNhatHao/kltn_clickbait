import os, numpy as np
from PIL import Image
from typing import List, Optional

def _load_image(path_or_none, image_size=224):
    if not isinstance(path_or_none, str) or not os.path.exists(path_or_none):
        img = Image.new("RGB", (image_size, image_size), (128,128,128))
    else:
        img = Image.open(path_or_none).convert("RGB").resize((image_size, image_size))
    return img

class _TorchvisionResNet18:
    def __init__(self, image_size=224):
        import torchvision.models as models
        import torch.nn as nn
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.dim = 512
        self.image_size = image_size
    def encode(self, pil_img):
        import torchvision.transforms as T, torch
        tr = T.Compose([T.Resize((self.image_size, self.image_size)), T.ToTensor()])
        x = tr(pil_img).unsqueeze(0)
        with torch.no_grad():
            v = self.model(x).squeeze(0).numpy()
        return v.astype(np.float32)

class _TimmEffNetB4:
    def __init__(self, image_size=224):
        try:
            import timm, torch.nn as nn
            self.model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=0)
            self.dim = self.model.num_features if hasattr(self.model, "num_features") else 1792
            self._ok = True
        except Exception:
            self._ok = False
            self.fallback = _TorchvisionResNet18(image_size=image_size)
            self.dim = self.fallback.dim
        self.image_size = image_size
    def encode(self, pil_img):
        if self._ok:
            import torchvision.transforms as T, torch
            tr = T.Compose([T.Resize((self.image_size, self.image_size)), T.ToTensor()])
            x = tr(pil_img).unsqueeze(0)
            with torch.no_grad():
                v = self.model(x).squeeze(0).numpy()
            return v.astype(np.float32)
        else:
            return self.fallback.encode(pil_img)

class _OpenCLIP_B32:
    def __init__(self, image_size=224):
        try:
            import open_clip, torch
            self.model, _, self.tok = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            self.model.eval()
            self._ok = True
            self.dim = 512
        except Exception:
            self._ok = False
            self.fallback = _TorchvisionResNet18(image_size=image_size)
            self.dim = self.fallback.dim
        self.image_size = image_size
    def encode(self, pil_img):
        if self._ok:
            import torch
            with torch.no_grad():
                x = self.tok(pil_img).unsqueeze(0)
                v = self.model.encode_image(x)
            return v.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            return self.fallback.encode(pil_img)

class MultiImageEncoder:
    def __init__(self, names, image_size=224):
        names = names if isinstance(names, list) else [names]
        self.encoders = []
        self.image_size = image_size
        for n in names:
            if n == "timm_efficientnet_b4":
                self.encoders.append(_TimmEffNetB4(image_size=image_size))
            elif n == "openclip_ViT-B-32":
                self.encoders.append(_OpenCLIP_B32(image_size=image_size))
            else:
                self.encoders.append(_TorchvisionResNet18(image_size=image_size))
        self.dim = sum(e.dim for e in self.encoders)
    def transform(self, image_paths: List[Optional[str]]) -> np.ndarray:
        feats = []
        for p in image_paths:
            img = _load_image(p, self.image_size)
            vecs = [enc.encode(img) for enc in self.encoders]
            feats.append(np.concatenate(vecs, axis=0))
        return np.stack(feats, axis=0).astype(np.float32)
