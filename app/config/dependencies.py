
from functools import lru_cache
from fastapi import Depends

from loguru import logger
from .properties import Properties

@lru_cache()
def get_properties() -> Properties:
    logger.info("Load settings from environment files")
    return Properties()

# Some shortcut functions for retrieving only a specific property
def get_api_version(properties: Properties = Depends(get_properties)) -> str:
    return properties.api_version

def get_regression_model_path(properties: Properties = Depends(get_properties)) -> str:
    return properties.regression_model_path
