import os, argparse, numpy as np, joblib
from flask import Flask, render_template, request, jsonify
from src.utils.io import load_yaml
from src.features.text_encoder import TextEncoder
from src.features.image_encoder import ImageEncoder

app = Flask(__name__)
CFG = None
MODEL = None
TEXT_ENC = None
IMG_ENC = None
BEST_THR = 0.5

def load_everything(cfg_path):
    global CFG, MODEL, TEXT_ENC, IMG_ENC, BEST_THR
    CFG = load_yaml(cfg_path)
    name = CFG["dataset_name"]
    paths = CFG["paths"]
    # Load model
    MODEL = joblib.load(os.path.join(paths["artifacts_dir"], "models", name, "stacking.joblib"))
    # Load threshold
    import json
    thr_path = os.path.join(paths["artifacts_dir"], "metrics", name + "_dev_metrics.json")
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            BEST_THR = json.load(f).get("best_thr", 0.5)
    # Encoders for online inference
    if "text" in CFG["modalities"]:
        TEXT_ENC = TextEncoder(**CFG["encoders"]["text"]).fit(["warmup sample"])
    if "image" in CFG["modalities"]:
        IMG_ENC = ImageEncoder(**CFG["encoders"]["image"])

@app.route("/")
def index():
    return render_template("index.html")

def concat_feats(a,b):
    if a is None: return b
    if b is None: return a
    return np.concatenate([a,b], axis=1)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.json if request.is_json else request.form
    text = data.get("text", "")
    image_path = data.get("image_path", None)

    X_text = TEXT_ENC.transform([text]) if TEXT_ENC else None
    X_img = IMG_ENC.transform([image_path]) if IMG_ENC and image_path else None
    X = concat_feats(X_text, X_img)
    prob = float(MODEL.predict_proba(X)[0])
    pred = int(prob >= BEST_THR)
    return jsonify({"prob": prob, "pred": pred, "threshold": BEST_THR})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/cldi.yaml)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    load_everything(args.config)
    app.run(host=args.host, port=args.port, debug=True)
