from typing import Dict, Union
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


TF_SERVING_URL = os.getenv(
    "TF_SERVING_URL",
    "http://localhost:8501/v1/models/user_engagement:predict",
)

MODEL_STATUS_URL = os.getenv(
    "TF_SERVING_STATUS_URL",
    "http://localhost:8501/v1/models/user_engagement",
)


app = FastAPI(
    title="User Engagement Prediction API",
    version="1.0.0",
)


NUMERIC_FEATURES = [
    "hour_of_day",
    "day_index",
    "same_category",
    "same_topic",
    "category_match_strength",
    "topic_match_strength",
    "event_category_confidence",
    "promoted_category_confidence",
    "event_topic_confidence",
    "promoted_topic_confidence",
    "event_category_missing",
    "promoted_category_missing",
    "event_topic_missing",
    "promoted_topic_missing",
    "user_prior_displays",

    "ad_prior_impressions",
    "ad_prior_clicks",
    "ad_historical_ctr",
    "ad_has_history",

    "campaign_prior_impressions",
    "campaign_prior_clicks",
    "campaign_historical_ctr",
    "campaign_has_history",

    "advertiser_prior_impressions",
    "advertiser_prior_clicks",
    "advertiser_historical_ctr",
    "advertiser_has_history",

    "promoted_document_prior_impressions",
    "promoted_document_prior_clicks",
    "promoted_document_historical_ctr",
    "promoted_document_has_history",
]


CATEGORICAL_FEATURES = [
    "platform_category",
    "country",
    "region",
    "event_category_id",
    "promoted_category_id",
    "event_topic_id",
    "promoted_topic_id",
]


class PredictionRequest(BaseModel):
    features: Dict[str, Union[float, int, str]]


@app.get("/")
def root():
    return {
        "service": "User Engagement Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    try:
        response = requests.get(
            MODEL_STATUS_URL,
            timeout=2,
        )

        response.raise_for_status()

        model_status = response.json()

        return {
            "api": "healthy",
            "model_server": "healthy",
            "model_status": model_status,
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"TensorFlow Serving unavailable: {exc}",
        )


@app.post("/predict")
def predict(request: PredictionRequest):
    features = request.features

    required_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing = [
        feature
        for feature in required_features
        if feature not in features
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing_features": missing,
            },
        )

    tf_inputs = {}

    try:
        for feature in NUMERIC_FEATURES:
            tf_inputs[feature] = [
                [float(features[feature])]
            ]

        for feature in CATEGORICAL_FEATURES:
            tf_inputs[feature] = [
                [str(features[feature])]
            ]

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature value: {exc}",
        )

    payload = {
        "inputs": tf_inputs
    }

    try:
        response = requests.post(
            TF_SERVING_URL,
            json=payload,
            timeout=5,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model server error: {exc}",
        )

    result = response.json()

    try:
        probability = float(
            result["outputs"][0][0]
        )

    except (KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Invalid response from TensorFlow Serving",
        )

    return {
        "engagement_probability": probability,
        "prediction": int(
            probability >= 0.5
        ),
        "model": "user_engagement",
    }