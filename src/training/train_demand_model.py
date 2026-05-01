import pandas as pd
import os
import joblib
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

MONGO_URI = "mongodb+srv://root:12345@cluster0.qwlmca2.mongodb.net/hardware_db"
DB_NAME = "hardware_db"

MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/demand_forecast_model.pkl"
ENCODER_FILE = f"{MODEL_DIR}/product_label_encoder.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_data_from_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    sale_items = list(db.saleitems.find())

    data = []

    for sale in sale_items:
        item = db.items.find_one({"_id": sale.get("itemId")})
        if not item:
            continue

        created_at = sale.get("createdAt")
        if not created_at:
            continue

        data.append({
            "product_id": str(item["_id"]),
            "date": pd.to_datetime(created_at),
            "quantity_sold": sale.get("quantity", 0),
            "unit_price": sale.get("unitPrice", 0),
        })

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("No data found in MongoDB")

    # 🔥 DAILY GROUPING
    df["date_only"] = df["date"].dt.date

    df = df.groupby(["product_id", "date_only"]).agg({
        "quantity_sold": "sum",
        "unit_price": "mean"
    }).reset_index()

    df["month"] = pd.to_datetime(df["date_only"]).dt.month
    df["day"] = pd.to_datetime(df["date_only"]).dt.day
    df["day_of_week"] = pd.to_datetime(df["date_only"]).dt.weekday
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    df = df.sort_values(by=["product_id", "date_only"])

    # FEATURES
    df["rolling_avg_qty"] = (
        df.groupby("product_id")["quantity_sold"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )

    df["previous_qty"] = df.groupby("product_id")["quantity_sold"].shift(1).fillna(0)

    df["lag_1"] = df["previous_qty"]
    df["lag_2"] = df.groupby("product_id")["quantity_sold"].shift(2).fillna(0)

    df["rolling_avg_7"] = df["rolling_avg_qty"]
    df["rolling_avg_30"] = df["rolling_avg_qty"]

    return df


def train_demand_model():
    df = fetch_data_from_mongodb()

    print(f"\nMongoDB data loaded: {len(df)} rows")

    encoder = LabelEncoder()
    df["product_encoded"] = encoder.fit_transform(df["product_id"])

    feature_columns = [
        "product_encoded",
        "unit_price",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "rolling_avg_qty",
        "previous_qty",
        "lag_1",
        "lag_2",
        "rolling_avg_7",
        "rolling_avg_30"
    ]

    X = df[feature_columns]
    y = df["quantity_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n📊 Demand Model Performance")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R² Score: {r2_score(y_test, predictions):.4f}")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print("\n✅ Demand model saved successfully")


if __name__ == "__main__":
    train_demand_model()