import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# FILE PATHS
INPUT_FILE = "data/processed/cleaned_sales_dataset.csv"
MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/revenue_forecast_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


def train_revenue_model():
    # Load processed dataset
    df = pd.read_csv(INPUT_FILE)

    print("Processed dataset loaded successfully")
    print(f"Rows: {len(df)}")

    # FEATURE SELECTION
    feature_columns = [
        "unit_price",
        "quantity_sold",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "rolling_avg_qty",
        "previous_qty"
    ]

    X = df[feature_columns]
    y = df["revenue"]

    # TRAIN TEST SPLIT
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =============================
    # TRAIN MODEL
    # =============================
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =============================
    # EVALUATION
    # =============================
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel trained successfully")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    # =============================
    # SAVE MODEL
    # =============================
    joblib.dump(model, MODEL_FILE)

    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_revenue_model()