from datetime import datetime
from fastapi import FastAPI, Depends, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.dto.health_check import HealthCheckResponse

from app.dto.price_prediction import PricePredictionRequest, PricePredictionResponse

from .services.dependencies import get_health_service, get_price_prediction_service
from .services.health import HealthService
from .services.price_prediction.PricePredictionService import PricePredictionService

# ==== SETUP APPLICATION ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== EXCEPTION HANDLER ====
@app.exception_handler(Exception)
async def default_exception_handler(request: Request, exception: RuntimeError):
    print(exception)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "path": request.url.path, 
            "message": str(exception)
        }
    )

# ==== ROUTES ====

# ---- Health Check ----
@app.get("/", response_model=HealthCheckResponse)
async def health_check(health_service: HealthService = Depends(get_health_service)):
    # NOTE : this is for the example, this is too complicated for a simple health check
    return HealthCheckResponse(
        api_version=health_service.get_api_version(),
        current_datetime=health_service.get_current_time()
    )


# ---- Price prediction task ----
@app.post("/predict-price", response_model=PricePredictionResponse)
async def predict_price(
    request: PricePredictionRequest, 
    service: PricePredictionService = Depends(get_price_prediction_service)
):
    # TODO : complete the function (always return 1 for now)
    return PricePredictionResponse(
        predicted_value=service.predict_price(request)
    )


# ---- Suggestion task ----

# TODO: create route for regression task (do not forget the "async" keyword !)