from fastapi import FastAPI
from pydantic import BaseModel

from src.prediction.predict import (
    predict_revenue,
    predict_demand
)

from src.training.train_revenue_model import train_revenue_model
from src.training.train_demand_model import train_demand_model

app = FastAPI()


class PredictionRequest(BaseModel):
    product_id: str

    unit_price: float
    quantity_sold: float

    month: int
    day: int
    day_of_week: int
    is_weekend: int

    rolling_avg_qty: float
    previous_qty: float

    # REQUIRED (missing before)
    lag_1: float
    lag_2: float
    rolling_avg_7: float
    rolling_avg_30: float


@app.post("/predict")
def predict(data: PredictionRequest):

    predicted_demand = predict_demand(
        product_id=data.product_id,
        unit_price=data.unit_price,
        month=data.month,
        day=data.day,
        day_of_week=data.day_of_week,
        is_weekend=data.is_weekend,
        rolling_avg_qty=data.rolling_avg_qty,
        previous_qty=data.previous_qty,
        lag_1=data.lag_1,
        lag_2=data.lag_2,
        rolling_avg_7=data.rolling_avg_7,
        rolling_avg_30=data.rolling_avg_30
    )

    predicted_revenue = predict_revenue(
        unit_price=data.unit_price,
        quantity_sold=data.quantity_sold,
        month=data.month,
        day=data.day,
        day_of_week=data.day_of_week,
        is_weekend=data.is_weekend,
        rolling_avg_qty=data.rolling_avg_qty,
        previous_qty=data.previous_qty,
        lag_1=data.lag_1,
        lag_2=data.lag_2,
        rolling_avg_7=data.rolling_avg_7,
        rolling_avg_30=data.rolling_avg_30
    )

    return {
        "predicted_demand": predicted_demand,
        "predicted_revenue": predicted_revenue
    }


@app.post("/retrain")
def retrain_models():
    train_revenue_model()
    train_demand_model()

    return {"message": "Models retrained successfully"}