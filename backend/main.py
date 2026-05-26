from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

import model as M
import inference as I
from schemas import InspectionResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything once at startup — model + FAISS index + threshold
    M.load_model()
    I.load_artifacts()
    yield
    # Nothing to clean up


app = FastAPI(
    title="DINOv3 Screw Inspector",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect", response_model=InspectionResult)
async def inspect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        result = I.run_inspection(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result