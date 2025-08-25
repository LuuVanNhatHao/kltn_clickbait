import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    MBartForConditionalGeneration, MBart50Tokenizer
)
# === CONFIG ===
CSV_PATH   = "data/vcc.csv"
IMG_FOLDER = "data/images"
OUT_CSV    = "data/vcc_with_caption.csv"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD MODELS ===

# BLIP-2 (FLAN-T5-XL)
print("🔄 Loading BLIP-2 FLAN-T5-XL...")
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl-coco")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl-coco").to(DEVICE).eval()

# mBART-50 for EN → VI translation
print("🔄 Loading mBART-50...")
mbart_name = "facebook/mbart-large-50-many-to-many-mmt"
mbart_tokenizer = MBart50Tokenizer.from_pretrained(mbart_name)
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_name).to(DEVICE).eval()


# === FUNCTIONS ===

def generate_caption(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
        output = blip_model.generate(**inputs, max_new_tokens=30)
        return blip_processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[❌] Failed on {img_path}: {e}")
        return ""

def translate_en_vi(texts, batch_size=8):
    translated = []
    mbart_tokenizer.src_lang = "en_XX"
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = mbart_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        outputs = mbart_model.generate(
            **inputs,
            forced_bos_token_id=mbart_tokenizer.lang_code_to_id["vi_VN"],
            max_new_tokens=100
        )
        translated += mbart_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translated


# === MAIN PIPELINE ===

df = pd.read_csv(CSV_PATH)
captions_en = []

print("🖼️ Generating English captions for VCC...")
for img_file in tqdm(df['thumbnail_url']):
    full_path = img_file.strip()  # Không cần ghép thêm
    captions_en.append(generate_caption(full_path))
df['caption_en'] = captions_en

print("🌐 Translating to Vietnamese with mBART-50...")
df['caption_vi'] = translate_en_vi(captions_en)

# Save
df.to_csv(OUT_CSV, index=False)
print(f"✅ Done! VCC caption saved to {OUT_CSV}")
