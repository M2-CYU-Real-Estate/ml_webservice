"""All services dependencies are defined here
"""
from functools import lru_cache

from fastapi import Depends

from ..config.dependencies import get_api_version, get_regression_model_path
from .price_prediction.PricePredictionService import PricePredictionService
from .health import HealthService

@lru_cache()
def get_health_service(api_version: str = Depends(get_api_version)) -> HealthService:
    return HealthService(api_version)

@lru_cache
def get_price_prediction_service(
    regression_model_path: str = Depends(get_regression_model_path)
) -> PricePredictionService:
    return PricePredictionService(regression_model_path)
