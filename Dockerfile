FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy project
COPY . .

# Install package in editable mode
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables (override at runtime)
ENV EMOTION_CKPT="checkpoints/emotion/best_model.h5"
ENV ENCODER_CKPT=""
ENV DECODER_CKPT=""
ENV VOCAB_PATH=""
ENV FACE_METHOD="haar"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
