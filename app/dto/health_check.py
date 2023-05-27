from datetime import datetime
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    api_version: str = "1.0"
    current_datetime: datetime