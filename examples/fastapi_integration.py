"""
Example: FastAPI integration with Prediction Guard.

This shows how to integrate Prediction Guard into a FastAPI inference endpoint.
"""

# Requires: pip install fastapi uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import random

# Add parent to path for running without install
import sys
sys.path.insert(0, "..")

from prediction_guard.types import GuardConfig
from prediction_guard.middleware import PredictionInterceptor


# =============================================================================
# Configuration
# =============================================================================

config = GuardConfig(
    log_directory="./api_logs",
    baseline_directory="./api_baselines",
    incident_directory="./api_incidents",
    current_model_version="v2.0",
)

interceptor = PredictionInterceptor(config)


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="ML Inference API with Prediction Guard")


class PredictRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    region: Optional[str] = "default"


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    request_id: str


def mock_ml_model(text: str) -> dict:
    """Your actual ML model would go here."""
    # Simulate prediction
    confidence = random.uniform(0.6, 0.99)
    prediction = "positive" if random.random() > 0.4 else "negative"
    embedding = [random.random() for _ in range(16)]
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": [confidence, 1 - confidence],
        "embedding": embedding,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run ML prediction with Prediction Guard monitoring.
    """
    # Build context for logging
    request_context = {
        "region": request.region,
        "user_id": request.user_id,
    }
    
    # Use interceptor to log telemetry
    with interceptor.intercept(request.text, request_context) as ctx:
        # Run actual model
        result = mock_ml_model(request.text)
        
        # Set result for logging
        ctx.set_result(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            embedding=result["embedding"],
        )
    
    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        request_id=ctx.request_id,
    )


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown():
    """Flush logs on shutdown."""
    interceptor.close()


# =============================================================================
# Run with: uvicorn fastapi_integration:app --reload
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
