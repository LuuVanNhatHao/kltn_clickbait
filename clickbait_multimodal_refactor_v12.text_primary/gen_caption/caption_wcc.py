import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# === CONFIG ===
CSV_PATH = "data/wcc_with_images.csv"
IMG_FOLDER = "data"
OUT_CSV = "data/wcc_caption.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD BLIP-2 ===
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl-coco")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl-coco").to(DEVICE).eval()


# === FUNCTION: Generate caption for one image ===
def generate_caption(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(DEVICE)
        output = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return ""


# === READ CSV ===
df = pd.read_csv(CSV_PATH)

# === Generate captions ===
captions = []

for media in tqdm(df['postMedia']):
    if pd.isna(media):
        captions.append("")
        continue

    # Xử lý để lấy được danh sách tên file ảnh đúng
    cleaned = str(media).replace("[", "").replace("]", "").replace("'", "").replace('"', '')
    media_list = [img.strip() for img in cleaned.split(',') if img.strip()]

    sample_captions = []
    for img_name in media_list:
        img_path = os.path.join(IMG_FOLDER, img_name)
        if os.path.isfile(img_path):
            caption = generate_caption(img_path)
            if caption:
                sample_captions.append(caption)
        else:
            print(f"⚠️ Not found: {img_path}")

    captions.append(" ".join(sample_captions))

# === Save to CSV ===
df['caption_en'] = captions
df.to_csv(OUT_CSV, index=False)
print(f"✅ Done! Saved to {OUT_CSV}")
