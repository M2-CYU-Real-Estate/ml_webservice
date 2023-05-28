
from pathlib import Path

from loguru import logger

from app.dto.price_prediction import PricePredictionRequest, PricePredictionResponse

import pickle 

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PricePredictionService:
    
    def __init__(self, model_path: str):
        # The pathlib API is so better than os.path
        self.model_path = Path(model_path)
        if not self.model_path.exists() or not self.model_path.is_file():
            raise ValueError("The provided path does not exists or is not a file : "
                             f"{model_path}")
            
        #load the predicting model
        try:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            raise ValueError("Le fichier du modèle n'a pas été trouvé : {}".format(model_path))
        except Exception as e:
            raise ValueError("Une erreur s'est produite lors du chargement du modèle : {}".format(str(e)))              
        
    def predict_price(self, request: PricePredictionRequest) -> int:
        # perform the prediction
        data_request = [request.house_area, request.rooms, request.ground_area, float(request.latitude), float(request.longitude)]
        return int(self.model.predict([data_request])[0])
