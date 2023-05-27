
from pydantic import BaseModel


class PricePredictionRequest(BaseModel):
    house_area: int
    ground_area: int
    rooms: int
    # Latitude and longitude in str format in order to avoid float imprecision
    latitude: str
    longitude: str
    
class PricePredictionResponse(BaseModel):
    predicted_value: int