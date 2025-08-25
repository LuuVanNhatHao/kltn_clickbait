from typing import List, Optional

class Captioner:
    def __init__(self, name="none", hf_model="Salesforce/blip2-flan-t5-xl-coco", max_new_tokens=30):
        self.name = name
        self.hf_model = hf_model
        self.max_new_tokens = max_new_tokens
        if self.name == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            self.processor = Blip2Processor.from_pretrained(hf_model)
            self.model = Blip2ForConditionalGeneration.from_pretrained(hf_model)
            self.torch = torch
        elif self.name == "none":
            pass
        else:
            raise ValueError(f"Unknown captioner: {self.name}")

    def generate(self, pil_images: List):
        if self.name == "none":
            return [None for _ in pil_images]
        import torch
        outs = []
        for img in pil_images:
            inputs = self.processor(img, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            outs.append(self.processor.decode(out[0], skip_special_tokens=True))
        return outs
