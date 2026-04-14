from fastapi import FastAPI
from pydantic import BaseModel
from src.prediction.predict import predict_revenue, predict_demand

app = FastAPI()


class PredictionRequest(BaseModel):
    product_id: str
    unit_price: float
    quantity_sold: int
    month: int
    day: int
    day_of_week: int
    is_weekend: int
    rolling_avg_qty: float
    previous_qty: float


@app.get("/")
def home():
    return {"message": "Hardware AI Prediction API running"}


@app.post("/predict")
def predict(data: PredictionRequest):
    revenue = predict_revenue(
        unit_price=data.unit_price,
        quantity_sold=data.quantity_sold,
        month=data.month,
        day=data.day,
        day_of_week=data.day_of_week,
        is_weekend=data.is_weekend,
        rolling_avg_qty=data.rolling_avg_qty,
        previous_qty=data.previous_qty
    )

    demand = predict_demand(
        product_id=data.product_id,
        unit_price=data.unit_price,
        month=data.month,
        day=data.day,
        day_of_week=data.day_of_week,
        is_weekend=data.is_weekend,
        rolling_avg_qty=data.rolling_avg_qty,
        previous_qty=data.previous_qty
    )

    return {
        "predicted_revenue": revenue,
        "predicted_demand": demand
    }