import io
import time
import base64

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import model as M
from config import (
    DEVICE,
    IMAGE_SIZE,
    PATCH_SIZE,
    NUM_SPECIAL_TOKENS,
    TOP_K_PERCENT,
    HEATMAP_ALPHA,
    FAISS_INDEX_PATH,
    THRESHOLD_PATH,
)

# ── FAISS index + threshold — loaded once at startup ──────────────────────
index: faiss.Index = None
threshold: float = None


def load_artifacts():
    global index, threshold

    import json

    print("Loading FAISS index ...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"FAISS index loaded — {index.ntotal} vectors.")

    with open(THRESHOLD_PATH) as f:
        data = json.load(f)

    threshold = data.get("threshold")
    print(f"Threshold (p95): {threshold:.6f}")


# ── Patch embedding extraction (mirrors notebook exactly) ─────────────────

@torch.inference_mode()
def extract_patch_embeddings(image: Image.Image):
    """
    Takes a PIL RGB image, returns:
        patch_embeddings : (N, 384) float32 ndarray  — L2-normalised
        grid_size        : int  (e.g. 28 for 448px)
    """
    image = TF.center_crop(image, IMAGE_SIZE)
    image = image.convert("L").convert("RGB")          # grayscale trick from notebook
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    inputs = M.processor(
        images=image,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(DEVICE)

    outputs = M.model(**inputs, interpolate_pos_encoding=True)
    tokens = outputs.last_hidden_state[0]

    patch_tokens = tokens[NUM_SPECIAL_TOKENS:]
    patch_embeddings = patch_tokens.detach().cpu().numpy().astype("float32")

    # L2 normalise (makes FAISS IndexFlatIP behave like cosine similarity)
    norms = np.linalg.norm(patch_embeddings, axis=1, keepdims=True) + 1e-8
    patch_embeddings = patch_embeddings / norms

    grid_size = IMAGE_SIZE // PATCH_SIZE
    expected = grid_size * grid_size
    actual = patch_embeddings.shape[0]
    if actual != expected:
        raise ValueError(
            f"Patch count mismatch: expected {expected}, got {actual}. "
            "Check IMAGE_SIZE / PATCH_SIZE / positional interpolation."
        )

    return patch_embeddings, grid_size


# ── Anomaly scoring ────────────────────────────────────────────────────────

def compute_anomaly_score(patch_embeddings: np.ndarray, grid_size: int):
    """
    Returns:
        score_map_norm : (grid_size, grid_size) float  — [0,1] for visualisation
        image_score    : float  — raw score used against threshold
        raw_score_map  : (grid_size, grid_size) float  — unnormalised
    """
    similarities, _ = index.search(patch_embeddings, k=1)
    similarities = similarities[:, 0]

    patch_scores = 1.0 - similarities          # anomaly = 1 − cosine similarity

    raw_score_map = patch_scores.reshape(grid_size, grid_size)

    score_map_norm = (raw_score_map - raw_score_map.min()) / (
        raw_score_map.max() - raw_score_map.min() + 1e-8
    )

    k = max(1, int(TOP_K_PERCENT * len(patch_scores)))
    image_score = float(np.mean(np.sort(patch_scores)[-k:]))

    return score_map_norm, image_score, raw_score_map


# ── Heatmap overlay ────────────────────────────────────────────────────────

def generate_overlay(original_pil: Image.Image, score_map_norm: np.ndarray) -> np.ndarray:
    """
    Blends the JET heatmap onto the original image.
    Returns an RGB uint8 ndarray.
    """
    original_rgb = np.array(original_pil.convert("RGB"))
    h, w = original_rgb.shape[:2]

    heatmap = cv2.resize(score_map_norm, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_rgb, 1 - HEATMAP_ALPHA, colored, HEATMAP_ALPHA, 0)
    return overlay


def ndarray_to_b64_png(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Main entry point called by the route ─────────────────────────────────

def run_inspection(image_bytes: bytes):
    """
    Full pipeline: raw bytes → InspectionResult fields.
    """
    start = time.time()

    original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    patch_embeddings, grid_size = extract_patch_embeddings(original_pil)
    score_map_norm, image_score, _ = compute_anomaly_score(patch_embeddings, grid_size)
    overlay = generate_overlay(original_pil, score_map_norm)
    overlay_b64 = ndarray_to_b64_png(overlay)

    latency_ms = round((time.time() - start) * 1000, 2)
    verdict = "DEFECTIVE" if image_score > threshold else "GOOD"

    return {
        "anomaly_score": image_score,
        "threshold": threshold,
        "verdict": verdict,
        "latency_ms": latency_ms,
        "heatmap_overlay_b64": overlay_b64,
    }