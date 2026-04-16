import pandas as pd
import os

# FILE PATHS
INPUT_FILE = "data/generated/hardware_sales_dataset_12months.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = f"{OUTPUT_DIR}/cleaned_sales_dataset.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_data():
    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    print("Original dataset loaded successfully")
    print(f"Rows: {len(df)}")

    # DATE PROCESSING
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week

    # FEATURE ENGINEERING
    df["is_weekend"] = df["day_of_week"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    # Revenue check
    df["calculated_revenue"] = (
        df["unit_price"] * df["quantity_sold"]
    )

    # Moving average demand per product
    df["rolling_avg_qty"] = (
        df.groupby("product_id")["quantity_sold"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Previous day sales (lag feature)
    df["previous_qty"] = (
        df.groupby("product_id")["quantity_sold"]
        .shift(1)
        .fillna(0)
    )

    # SAVE CLEANED DATA
    df.to_csv(OUTPUT_FILE, index=False)

    print("Preprocessed dataset saved successfully")
    print(f"Saved to: {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    preprocess_data()