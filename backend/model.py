import torch
from transformers import AutoImageProcessor, AutoModel
from config import MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded once at FastAPI startup via lifespan
processor: AutoImageProcessor = None
model: AutoModel = None


def load_model():
    global processor, model
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("Model ready.")