
from pathlib import Path

from loguru import logger

from app.dto.price_prediction import PricePredictionRequest, PricePredictionResponse


class PricePredictionService:
    
    def __init__(self, model_path: str):
        # The pathlib API is so better than os.path
        self.model_path = Path(model_path)
        if not self.model_path.exists() or not self.model_path.is_file():
            raise ValueError("The provided path does not exists or is not a file : "
                             f"{model_path}")
            
        # TODO: load the model !
        pass
        
    def predict_price(request: PricePredictionRequest) -> int:
        # TODO perform the regression
        logger.warning("Method not implemented for now")
        return 1
