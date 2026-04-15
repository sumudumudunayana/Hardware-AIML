import joblib
import pandas as pd

# LOAD MODELS
revenue_model = joblib.load("models/revenue_forecast_model.pkl")
demand_model = joblib.load("models/demand_forecast_model.pkl")
encoder = joblib.load("models/product_label_encoder.pkl")


def predict_revenue(
    unit_price,
    quantity_sold,
    month,
    day,
    day_of_week,
    is_weekend,
    rolling_avg_qty,
    previous_qty
):
    features = pd.DataFrame([{
        "unit_price": unit_price,
        "quantity_sold": quantity_sold,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "rolling_avg_qty": rolling_avg_qty,
        "previous_qty": previous_qty
    }])

    prediction = revenue_model.predict(features)[0]

    return round(prediction, 2)


def predict_demand(
    product_id,
    unit_price,
    month,
    day,
    day_of_week,
    is_weekend,
    rolling_avg_qty,
    previous_qty
):
    encoded_product = encoder.transform([product_id])[0]

    features = pd.DataFrame([{
        "product_encoded": encoded_product,
        "unit_price": unit_price,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "rolling_avg_qty": rolling_avg_qty,
        "previous_qty": previous_qty
    }])

    prediction = demand_model.predict(features)[0]

    return round(prediction, 2)


if __name__ == "__main__":
    revenue = predict_revenue(
        unit_price=650,
        quantity_sold=10,
        month=5,
        day=1,
        day_of_week=3,
        is_weekend=0,
        rolling_avg_qty=8,
        previous_qty=9
    )

    demand = predict_demand(
        product_id="E001",
        unit_price=650,
        month=5,
        day=1,
        day_of_week=3,
        is_weekend=0,
        rolling_avg_qty=8,
        previous_qty=9
    )

    print(f"Predicted Revenue: {revenue}")
    print(f"Predicted Demand: {demand}")