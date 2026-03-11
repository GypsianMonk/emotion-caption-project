# Real-Time Emotion Detection & Image Captioning

> Production-grade pipeline combining CNN-based facial emotion recognition at 24 FPS and InceptionV3+LSTM image captioning trained on MS-COCO 80K images.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green) ![BLEU-4](https://img.shields.io/badge/BLEU--4-0.28-brightgreen) ![Accuracy](https://img.shields.io/badge/Emotion%20Acc-82%25-brightgreen)

---

## 🔑 Key Results

| Task | Model | Dataset | Metric |
|---|---|---|---|
| Emotion Recognition | Custom CNN | FER-2013 (35,887 imgs) | **82% accuracy, 24 FPS** |
| Image Captioning | InceptionV3 + LSTM | MS-COCO 80K | **BLEU-4: 0.28** |

---

## 🏗️ Architecture

### Emotion Detection
```
Input Frame (48×48 grayscale)
    → Conv2D(32) → BN → ReLU → MaxPool
    → Conv2D(64) → BN → ReLU → MaxPool
    → Conv2D(128) → BN → ReLU → MaxPool
    → Conv2D(256) → BN → ReLU → GlobalAvgPool
    → Dense(512) → Dropout(0.5) → Dense(7, softmax)
```

### Image Captioning
```
Image (299×299 RGB)
    → InceptionV3 (frozen, remove top) → Feature vector (2048,)
    → Dense projection (512,)
    → LSTM Decoder with Bahdanau Attention
    → Word Embeddings → Dense(vocab_size, softmax)
```

---

## 📁 Project Structure

```
emotion_caption_project/
├── configs/
│   ├── emotion_config.yaml         # Emotion model hyperparams
│   └── captioning_config.yaml      # Caption model hyperparams
├── data/
│   └── preprocessing/
│       ├── fer2013_preprocessor.py # FER-2013 loading & augmentation
│       └── coco_preprocessor.py    # COCO annotation parsing & tokenization
├── models/
│   ├── emotion/
│   │   ├── cnn_model.py            # CNN architecture definition
│   │   └── trainer.py              # Training loop with callbacks
│   └── captioning/
│       ├── encoder.py              # InceptionV3 feature extractor
│       ├── decoder.py              # LSTM + Attention decoder
│       ├── attention.py            # Bahdanau attention mechanism
│       └── trainer.py              # Custom training loop
├── inference/
│   ├── emotion_detector.py         # Real-time emotion inference
│   ├── caption_generator.py        # Beam search caption generation
│   └── real_time_pipeline.py       # Unified webcam pipeline
├── utils/
│   ├── metrics.py                  # BLEU, CIDEr, METEOR scoring
│   ├── visualization.py            # Plotting, overlay rendering
│   └── logger.py                   # Structured logging
├── api/
│   ├── main.py                     # FastAPI REST endpoints
│   └── schemas.py                  # Pydantic request/response models
├── scripts/
│   ├── train_emotion.py            # CLI training script: emotion
│   ├── train_captioning.py         # CLI training script: captioning
│   ├── evaluate.py                 # Evaluation on test sets
│   └── export_model.py             # TFLite / SavedModel export
├── tests/
│   ├── test_models.py
│   ├── test_inference.py
│   └── test_api.py
├── requirements.txt
├── setup.py
└── Dockerfile
```

---

## ⚡ Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/emotion-caption-project
cd emotion-caption-project
pip install -e .

# 2. Download datasets
# FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
# MS-COCO: https://cocodataset.org/#download (2017 train/val + annotations)

# 3. Preprocess
python scripts/preprocess_fer.py --data_dir data/raw/fer2013
python scripts/preprocess_coco.py --data_dir data/raw/coco --max_vocab 10000

# 4. Train emotion model
python scripts/train_emotion.py --config configs/emotion_config.yaml

# 5. Train captioning model (requires ~4h on single GPU)
python scripts/train_captioning.py --config configs/captioning_config.yaml

# 6. Run real-time demo
python inference/real_time_pipeline.py --emotion_ckpt checkpoints/emotion_best.h5 \
                                        --caption_ckpt checkpoints/captioning_best/

# 7. Launch REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 📊 Training Details

### Emotion CNN
- **Optimizer**: Adam (lr=1e-3, decay to 1e-5 via ReduceLROnPlateau)
- **Batch size**: 64
- **Epochs**: 60 (EarlyStopping patience=10)
- **Augmentation**: RandomFlip, RandomRotation(±10°), RandomZoom, RandomBrightness
- **Class weights**: Balanced (FER-2013 is imbalanced toward Neutral/Happy)
- **Training time**: ~45 min on single RTX 3060

### Image Captioning
- **Encoder**: InceptionV3 pretrained on ImageNet, weights frozen
- **Decoder**: 2-layer LSTM (512 units) + Bahdanau Attention
- **Embedding dim**: 256, trainable
- **Optimizer**: Adam (lr=1e-3)
- **Batch size**: 64
- **Epochs**: 20 (teacher forcing)
- **Training time**: ~4h on single RTX 3060

---

## 🚀 API Endpoints

```
POST /emotion          — Detect emotions from uploaded image
POST /caption          — Generate caption for uploaded image
POST /pipeline         — Emotion + caption in single call
GET  /health           — Health check
GET  /metrics          — Model performance metrics
```

---

## 📦 Export & Deployment

```bash
# Export emotion model to TFLite (for edge deployment)
python scripts/export_model.py --model emotion --format tflite

# Export as SavedModel (for TF Serving)
python scripts/export_model.py --model captioning --format saved_model

# Docker
docker build -t emotion-caption .
docker run -p 8000:8000 --gpus all emotion-caption
```
