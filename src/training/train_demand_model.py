import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# FILE PATHS
INPUT_FILE = "data/processed/cleaned_sales_dataset.csv"
MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/demand_forecast_model.pkl"
ENCODER_FILE = f"{MODEL_DIR}/product_label_encoder.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


def train_demand_model():
    # Load processed dataset
    df = pd.read_csv(INPUT_FILE)

    print("Processed dataset loaded successfully")
    print(f"Rows: {len(df)}")

    # ENCODE PRODUCT IDS
    encoder = LabelEncoder()
    df["product_encoded"] = encoder.fit_transform(df["product_id"])

    # FEATURE SELECTION
    feature_columns = [
        "product_encoded",
        "unit_price",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "rolling_avg_qty",
        "previous_qty"
    ]

    X = df[feature_columns]

    # Target = future quantity demand
    y = df["quantity_sold"]

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # TRAIN MODEL
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # EVALUATE MODEL
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nDemand model trained successfully")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    # SAVE MODEL + ENCODER
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Encoder saved to: {ENCODER_FILE}")


if __name__ == "__main__":
    train_demand_model()