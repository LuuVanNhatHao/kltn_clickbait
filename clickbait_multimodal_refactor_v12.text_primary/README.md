
# Clickbait Multimodal – Refactor (v8)

This drop adds your requested **Optuna hyperparameter tuning**, firm separation between **EfficientNet‑B4** (for `image_only`) and **CLIP (OpenCLIP ViT‑B/32)** (for `text_image`), and **tqdm** progress bars throughout training. We also keep stacked ensembling and the full **classification report**.

## Modalities
- **text_only** — caption/title encoded by HF (VCC→PhoBERT‑large, WCC→RoBERTa‑large).
- **image_only** — image features from **EfficientNet‑B4** (timm).
- **caption_only** — caption with same HF encoder as text.
- **text_caption** — concat of (text ⊕ caption) [HF].
- **text_image** — concat of (**CLIP text** ⊕ **CLIP image**). Caption is the preferred text input; falls back to `text` if caption missing.
- **tabular_only** — for CLDI (unchanged).

> “Concat” means we **concatenate** feature vectors (text is primary, image supports). No mixing/averaging.

## What changed
- `src/tuning/optuna_tuner.py` — Optuna search spaces for each base learner (LR, LinearSVM, RF, GBDT, ExtraTrees, MLP), with CV and F1 as objective.
- `src/models/heads.py` — every base learner now accepts tuned params (e.g., LR `C`, RF `n_estimators`, MLP `hidden_layer_sizes`, …).
- `src/models/stacking.py` — adds **tqdm** progress bars for base‑learner folds and supports a LightGBM meta‑learner (optional, if installed).
- `scripts/extract_features.py` —
  - **EfficientNet‑B4** used only for `image_only` → saved as `X_image_effnet.npy`.
  - **OpenCLIP ViT‑B/32** used only for `text_image` → saves `X_text_clip.npy` and `X_image_clip.npy` and we concat at evaluation time.
  - Caption encodings saved as `X_caption.npy`.
- `scripts/evaluate.py` — new `--optuna_trials N` flag, uses tuned params per base learner, prints **classification_report** and summary metrics; **tqdm** on tasks/tuning.
- `configs/vcc.yaml`, `configs/wcc.yaml` — clarified columns (`caption_vi` vs `caption_en`) and encoder roles.

## How to run

### 1) Feature extraction
```bash
# VCC (PhoBERT + EffNet‑B4 + OpenCLIP)
python scripts/extract_features.py --config configs/vcc.yaml --split train
python scripts/extract_features.py --config configs/vcc.yaml --split dev
python scripts/extract_features.py --config configs/vcc.yaml --split test

# WCC (RoBERTa + EffNet‑B4 + OpenCLIP)
python scripts/extract_features.py --config configs/wcc.yaml --split train
python scripts/extract_features.py --config configs/wcc.yaml --split dev
python scripts/extract_features.py --config configs/wcc.yaml --split test
```

### 2) Train + tune + evaluate
```bash
# Run all tasks with Optuna tuning (e.g., 30 trials) and stacked ensemble
python scripts/evaluate.py --config configs/vcc.yaml --task all --optuna_trials 30
python scripts/evaluate.py --config configs/wcc.yaml --task all --optuna_trials 30

# Or just a single modality
python scripts/evaluate.py --config configs/vcc.yaml --task text_image --optuna_trials 40
python scripts/evaluate.py --config configs/vcc.yaml --task image_only --optuna_trials 20
```

- Results are saved to `artifacts/<dataset>/results/all_results.json`.
- Best hyperparameters per base learner are saved under `artifacts/<dataset>/results/<task>/optuna/*.json`.

## Notes
- **tqdm** bars show base‑learner progress across CV folds and Optuna progress.
- **CLIP for text_image**: we use the **caption** column (`caption_vi` for VCC, `caption_en` for WCC) as the text source inside CLIP’s text encoder; if caption is missing, we fall back to `text`.
- **EfficientNet‑B4** is not used in `text_image` to avoid mixing encoder spaces as requested.
- Everything still works without Optuna (`--optuna_trials 0`).

---

If you want me to pin exact hyperparameter ranges differently (e.g., deeper trees, larger MLPs) or to bias the meta‑learner toward LightGBM, say the word and I’ll wire it in.



## Multimodal Fusion Policy
- **Text is PRIMARY** signal; **Caption** (generated from image) and **Image** are SUPPORT signals.
- Fusion is simple **concatenation** keeping the order `[text, support]`:
  - `text_caption`: `[X_text, X_caption]`
  - `text_image`   (VCC/WCC CLIP): `[X_text_clip, X_image_clip]`
  - `text_image`   (CLDI precomputed): `[X_text, X_image_precomputed]`
- CLDI has no `text_caption` because captions are not generated-from-image; dataset only provides tabular + optional text fields and precomputed image features.
