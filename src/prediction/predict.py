import joblib
import pandas as pd

revenue_model = joblib.load("models/revenue_forecast_model.pkl")
demand_model = joblib.load("models/demand_forecast_model.pkl")
encoder = joblib.load("models/product_label_encoder.pkl")


def encode_product(product_id):
    if product_id not in encoder.classes_:
        raise ValueError(f"Unknown product_id: {product_id}")
    return encoder.transform([product_id])[0]


def predict_demand(product_id, unit_price, month, day, day_of_week,
                   is_weekend, rolling_avg_qty, previous_qty,
                   lag_1, lag_2, rolling_avg_7, rolling_avg_30):

    encoded_product = encode_product(product_id)

    features = pd.DataFrame([{
        "product_encoded": encoded_product,
        "unit_price": unit_price,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "rolling_avg_qty": rolling_avg_qty,
        "previous_qty": previous_qty,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "rolling_avg_7": rolling_avg_7,
        "rolling_avg_30": rolling_avg_30
    }])

    return round(float(demand_model.predict(features)[0]), 2)
    # return max(0, int(round(demand_model.predict(features)[0])))


def predict_revenue(unit_price, quantity_sold, month, day,
                    day_of_week, is_weekend,
                    rolling_avg_qty, previous_qty,
                    lag_1, lag_2, rolling_avg_7, rolling_avg_30):

    features = pd.DataFrame([{
        "unit_price": unit_price,
        "quantity_sold": quantity_sold,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "rolling_avg_qty": rolling_avg_qty,
        "previous_qty": previous_qty,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "rolling_avg_7": rolling_avg_7,
        "rolling_avg_30": rolling_avg_30
    }])

    return round(float(revenue_model.predict(features)[0]), 2)
    # return max(0, int(round(revenue_model.predict(features)[0])))