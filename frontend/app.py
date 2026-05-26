import base64
import io
import os
from dotenv import load_dotenv
import requests
import streamlit as st
from PIL import Image

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

st.set_page_config(
    page_title="DinoInspect",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0d0d;
    color: #d8d8d8;
  }

  #MainMenu, footer, header { visibility: hidden; }

  .block-container {
    padding-top: 32px;
    padding-bottom: 32px;
    max-width: 1200px;
  }

  .app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 20px;
    border-bottom: 1px solid #242424;
    margin-bottom: 24px;
  }
  .app-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #f0f0f0;
    margin: 0;
  }
  .app-header .subtitle {
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #505050;
  }

  .card {
    background: #111111;
    border: 1px solid #242424;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .label-sm {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #505050;
    margin-bottom: 8px;
  }

  .verdict-good     { border-left: 3px solid #27ae60; }
  .verdict-defective{ border-left: 3px solid #c0392b; }
  .verdict-icon {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 0.06em;
    line-height: 1;
  }
  .verdict-good      .verdict-icon { color: #27ae60; }
  .verdict-defective .verdict-icon { color: #c0392b; }
  .verdict-sub {
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #505050;
    margin-top: 4px;
  }

  .bar-track {
    position: relative;
    height: 6px;
    background: #242424;
    border-radius: 3px;
    overflow: visible;
    margin: 10px 0 6px;
  }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-fill.good       { background: #27ae60; }
  .bar-fill.defective  { background: #c0392b; }
  .bar-tick {
    position: absolute;
    top: -5px;
    width: 2px;
    height: 16px;
    background: #d4a017;
    border-radius: 1px;
  }
  .bar-nums {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #3a3a3a;
  }
  .bar-nums .tick-label { color: #d4a017; }

  .metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
  }
  .metric-cell {
    background: #0d0d0d;
    border: 1px solid #242424;
    border-radius: 6px;
    padding: 12px 14px;
  }
  .metric-cell .m-label {
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #505050;
    margin-bottom: 4px;
  }
  .metric-cell .m-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px;
    color: #f0f0f0;
  }

  [data-testid="stFileUploader"] {
    background: #111111 !important;
    border: 1px dashed #242424 !important;
    border-radius: 6px !important;
    padding: 16px !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #d4a017 !important;
  }

  /* constrain images to never blow up */
  [data-testid="stImage"] img {
    max-height: 340px !important;
    width: auto !important;
    object-fit: contain;
    border-radius: 6px;
    border: 1px solid #242424;
  }

  [data-testid="caption"] {
    font-size: 10px !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #505050 !important;
    text-align: left !important;
    margin-top: 6px !important;
  }

  .empty-state {
    height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 1px dashed #242424;
    border-radius: 6px;
    color: #3a3a3a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.16em;
    gap: 10px;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>🔩 SCREW INSPECTOR</h1>
  <span class="subtitle">DINOv3 &middot; Anomaly Detection &middot; Industrial QC</span>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def backend_alive():
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


if not backend_alive():
    st.error("Cannot reach backend at `localhost:8000`. Run `uvicorn main:app --reload` first.")
    st.stop()

# ── Upload (full width, above columns) ────────────────────────────────────
st.markdown('<div class="label-sm">Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    label_visibility="collapsed",
)

# ── Two columns only appear after upload ──────────────────────────────────
if not uploaded:
    st.markdown('<div class="empty-state"><div>🔩</div><div>AWAITING INPUT</div></div>', unsafe_allow_html=True)

else:
    with st.spinner("Analysing ..."):
        uploaded.seek(0)
        resp = requests.post(
            f"{BACKEND_URL}/inspect",
            files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
            timeout=60,
        )

    if resp.status_code != 200:
        st.error(f"Backend error {resp.status_code}: {resp.text}")
        st.stop()

    data        = resp.json()
    verdict     = data["verdict"]
    score       = data["anomaly_score"]
    threshold   = data["threshold"]
    latency_ms  = data["latency_ms"]
    overlay_b64 = data["heatmap_overlay_b64"]

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="label-sm">Input</div>', unsafe_allow_html=True)
        uploaded.seek(0)
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=False)

    with right:
        st.markdown('<div class="label-sm">Heatmap</div>', unsafe_allow_html=True)
        overlay_bytes = base64.b64decode(overlay_b64)
        overlay_img   = Image.open(io.BytesIO(overlay_bytes))
        st.image(overlay_img, use_container_width=False)

    # ── Results (full width, below images) ────────────────────────────────
    v_class  = "verdict-good" if verdict == "GOOD" else "verdict-defective"
    v_icon   = "✓  GOOD" if verdict == "GOOD" else "✗  DEFECTIVE"
    v_sub    = "No anomalies detected" if verdict == "GOOD" else "Anomalies detected"

    ceiling   = max(score, threshold) * 1.35
    fill_pct  = round(min(score / ceiling * 100, 100), 2)
    tick_pct  = round(min(threshold / ceiling * 100, 100), 2)
    bar_class = "good" if verdict == "GOOD" else "defective"

    r1, r2 = st.columns([1, 2], gap="large")

    with r1:
        st.markdown(f"""
        <div class="card {v_class}">
          <div class="verdict-icon">{v_icon}</div>
          <div class="verdict-sub">{v_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="card">
          <div class="label-sm">Anomaly Score vs Threshold</div>
          <div class="bar-track">
            <div class="bar-fill {bar_class}" style="width:{fill_pct}%"></div>
            <div class="bar-tick" style="left:{tick_pct}%"></div>
          </div>
          <div class="bar-nums">
            <span>0</span>
            <span class="tick-label">&#9650; {threshold:.4f}</span>
            <span>{ceiling:.4f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-cell">
        <div class="m-label">Anomaly Score</div>
        <div class="m-value">{score:.6f}</div>
      </div>
      <div class="metric-cell">
        <div class="m-label">Threshold (p92)</div>
        <div class="m-value">{threshold:.6f}</div>
      </div>
      <div class="metric-cell">
        <div class="m-label">Margin</div>
        <div class="m-value">{abs(score - threshold):.6f}</div>
      </div>
      <div class="metric-cell">
        <div class="m-label">Latency</div>
        <div class="m-value">{latency_ms} ms</div>
      </div>
    </div>
    """, unsafe_allow_html=True)