from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

# ── Image / patch settings ─────────────
IMAGE_SIZE = 448
PATCH_SIZE = 16
NUM_SPECIAL_TOKENS = 5       # 1 CLS token + 4 register tokens
TOP_K_PERCENT = 0.01         # top 1% most anomalous patches → image score

# ── Artifacts (downloaded from Colab) ─────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
THRESHOLD_PATH   = ARTIFACTS_DIR / "threshold.json"

# ── Inference ──────────────────────────────────────────────────────────────
HEATMAP_ALPHA = 0.55         # overlay blend strength