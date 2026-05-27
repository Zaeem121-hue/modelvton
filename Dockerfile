# ==========================================
# Hybrid Virtual Try-On — RunPod Docker Image
# Base: PyTorch 2.1.0 + CUDA 12.1 + cuDNN 8
# ==========================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# ==========================================
# System dependencies for OpenCV / image ops
# ==========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Install Python dependencies
# torch/torchvision already provided by base image
# ==========================================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi>=0.104.0 \
        uvicorn>=0.24.0 \
        pydantic>=2.0 \
        pydantic[email] \
        python-multipart \
        Pillow>=10.0 \
        numpy>=1.24 \
        opencv-python>=4.8 \
        diffusers>=0.24.0 \
        transformers>=4.36.0 \
        accelerate>=0.25.0 \
        huggingface_hub>=0.20.0 \
        rembg>=0.0.9

# ==========================================
# Copy application code
# ==========================================
COPY . .

# ==========================================
# Create model cache directory
# (Mount a RunPod network volume here for persistence)
# ==========================================
RUN mkdir -p /workspace/model_cache

# ==========================================
# Environment configuration
# ==========================================
ENV DEVICE=cuda
ENV DTYPE=fp16
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/model_cache
ENV HF_HUB_CACHE=/workspace/model_cache

# ==========================================
# Expose API port
# ==========================================
EXPOSE 8000

# ==========================================
# Health check
# ==========================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ==========================================
# Start FastAPI server with uvicorn
# ==========================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
