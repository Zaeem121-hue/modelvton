import os

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

OOTD_MODEL_PATH = os.getenv("OOTD_MODEL_PATH", os.path.join(MODEL_CACHE_DIR, "ootdiffusion"))
OMNITRY_MODEL_PATH = os.getenv("OMNITRY_MODEL_PATH", os.path.join(MODEL_CACHE_DIR, "omnitry"))

DEVICE = os.getenv("DEVICE", "cuda")
DTYPE = os.getenv("DTYPE", "fp16")
