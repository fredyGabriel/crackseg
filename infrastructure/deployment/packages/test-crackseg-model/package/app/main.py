"""CrackSeg Deployment Application.

This is the main entry point for the CrackSeg deployment package.
Provides REST API endpoints for model inference and health monitoring.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="CrackSeg API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    image_path: str
    confidence_threshold: float = 0.5


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: dict[str, Any]
    confidence: float
    processing_time_ms: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "crackseg"}


@app.get("/model/info")
async def model_info():
    """Get model information."""
    model_path = Path("app/model.pth")
    if model_path.exists():
        return {
            "model_path": str(model_path),
            "model_size_mb": model_path.stat().st_size / 1024 / 1024,
            "framework": "pytorch",
        }
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Perform crack segmentation prediction."""
    try:
        start_time = time.time()

        # Load model (simplified for demo)
        model_path = Path("app/model.pth")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Simulate prediction
        prediction = {
            "crack_detected": True,
            "confidence": 0.85,
            "segmentation_mask": "base64_encoded_mask",
        }

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=prediction,
            confidence=prediction["confidence"],
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8501))
    uvicorn.run(app, host="0.0.0.0", port=port)
