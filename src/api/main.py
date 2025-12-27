import logging
import time
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.schemas import HeartFeatures, PredictionResponse
from src.models.predict_model import HeartDiseaseModelService

logger = logging.getLogger("heart_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    docs_url="/",  # Swagger UI at root
    redoc_url=None,  # optional: disable ReDoc
    openapi_url="/openapi.json",
)

model_service = HeartDiseaseModelService()

# Simple in-memory metrics
metrics: Dict[str, int] = {
    "total_requests": 0,
    "total_predictions": 0,
    "positive_predictions": 0,
    "negative_predictions": 0,
}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} completed_in={duration:.4f}s "
        f"status_code={response.status_code}"
    )
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HeartFeatures):
    metrics["total_requests"] += 1

    # Predict single
    result = model_service.predict_single(features.model_dump())

    metrics["total_predictions"] += 1
    if result["prediction"] == 1:
        metrics["positive_predictions"] += 1
    else:
        metrics["negative_predictions"] += 1

    return PredictionResponse(**result)


@app.get("/metrics")
async def get_metrics():
    # Simple metrics endpoint; could be adapted to Prometheus format later
    return JSONResponse(content=metrics)
