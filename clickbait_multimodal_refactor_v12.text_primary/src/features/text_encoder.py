import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

# -----------------------------
# Helpers
# -----------------------------
def _auto_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------
# HFTextEncoder
# -----------------------------
@dataclass
class HFTextEncoderConfig:
    name: str
    max_length: int = 128
    batch_size: int = 16
    device: str = "auto"
    use_pooler: bool = True  # nếu model có pooler_output; nếu không thì mean-pool last hidden

class HFTextEncoder:
    """
    Encoder văn bản dùng HuggingFace Transformers.
    encode(texts: List[str]) -> np.ndarray [N, D]
    """
    def __init__(self, name: str, max_length: int = 128, batch_size: int = 16,
                 device: str = "auto", use_pooler: bool = True):
        from transformers import AutoTokenizer, AutoModel
        self.name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = _auto_device(device)
        self.use_pooler = use_pooler

        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval().to(self.device)

        # suy luận chiều D
        with torch.no_grad():
            toks = self.tokenizer(["dummy"], padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt").to(self.device)
            out = self.model(**toks)
            if self.use_pooler and hasattr(out, "pooler_output") and out.pooler_output is not None:
                dim = out.pooler_output.shape[-1]
            else:
                dim = out.last_hidden_state.shape[-1]
        self.dim = int(dim)

    def _batch_iter(self, arr: List[str], bs: int):
        for i in range(0, len(arr), bs):
            yield arr[i:i+bs]

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = []
        with torch.no_grad():
            for chunk in self._batch_iter(texts, self.batch_size):
                toks = self.tokenizer(
                    chunk, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                out = self.model(**toks)
                if self.use_pooler and hasattr(out, "pooler_output") and out.pooler_output is not None:
                    vec = out.pooler_output
                else:
                    # mean-pool theo attention mask
                    last = out.last_hidden_state   # [B, T, H]
                    mask = toks["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                    last = last * mask
                    vec = last.sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-6))
                embs.append(vec.detach().cpu())
        return torch.cat(embs, dim=0).numpy().astype(np.float32)

# -----------------------------
# CLIPTextEncoder (OpenCLIP)
# -----------------------------
class CLIPTextEncoder:
    """
    Text encoder dùng OpenCLIP (ViT-B/32...). Chủ yếu phục vụ nhánh text của CLIP.
    encode(texts: List[str]) -> np.ndarray [N, D]
    """
    def __init__(self, name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                 batch_size: int = 16, device: str = "auto"):
        try:
            import open_clip
        except Exception as e:
            raise ImportError("open_clip chưa được cài. Hãy thêm 'open-clip-torch' vào requirements.txt") from e

        self.batch_size = batch_size
        self.device = _auto_device(device)
        self.model, _, self.tokenizer = open_clip.create_model_and_transforms(
            name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        # suy luận D
        self.dim = int(self.model.text_projection.shape[-1]) if hasattr(self.model, "text_projection") else 512

    def _batch_iter(self, arr: List[str], bs: int):
        for i in range(0, len(arr), bs):
            yield arr[i:i+bs]

    def encode(self, texts: List[str]) -> np.ndarray:
        import open_clip
        embs = []
        with torch.no_grad():
            for chunk in self._batch_iter(texts, self.batch_size):
                toks = open_clip.tokenize(chunk).to(self.device)
                vec = self.model.encode_text(toks)
                vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
                embs.append(vec.detach().cpu())
        return torch.cat(embs, dim=0).numpy().astype(np.float32)
