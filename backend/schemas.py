from pydantic import BaseModel


class InspectionResult(BaseModel):
    anomaly_score: float
    threshold: float
    verdict: str          # "GOOD" | "DEFECTIVE"
    latency_ms: float
    heatmap_overlay_b64: str   # base64-encoded PNG sent to Streamlit