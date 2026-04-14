import pandas as pd
import random
import os
from datetime import datetime, timedelta

# CREATE OUTPUT FOLDER
OUTPUT_DIR = "data/generated"
OUTPUT_FILE = f"{OUTPUT_DIR}/hardware_sales_dataset_12months.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# PRODUCT MASTER LIST
products = [
    {
        "id": "E001",
        "name": "13A Plug base (ACL)",
        "category": "Electrical",
        "price": 650,
        "sales_type": "fast",
    },
    {
        "id": "E002",
        "name": "1 GANG Switch (KEVILTON)",
        "category": "Electrical",
        "price": 320,
        "sales_type": "fast",
    },
    {
        "id": "E003",
        "name": "LED 9W pin (GREEN)",
        "category": "Electrical",
        "price": 575,
        "sales_type": "fast",
    },
    {
        "id": "E004",
        "name": "MCB enclosure 18 way",
        "category": "Electrical",
        "price": 2800,
        "sales_type": "slow",
    },
    {
        "id": "P001",
        "name": "PVC pipe 1/2 (ARPICO)",
        "category": "Plumbing",
        "price": 330,
        "sales_type": "medium",
    },
    {
        "id": "P002",
        "name": "Valve socket 1/2",
        "category": "Plumbing",
        "price": 40,
        "sales_type": "fast",
    },
    {
        "id": "P003",
        "name": "Elbow socket 1/2",
        "category": "Plumbing",
        "price": 40,
        "sales_type": "fast",
    },
    {
        "id": "O001",
        "name": "Caltex 20W50",
        "category": "Oil",
        "price": 1200,
        "sales_type": "medium",
    },
]

# SALES GENERATION LOGIC
def generate_quantity(sales_type, month):
    seasonal_multiplier = 1.0

    # Seasonal demand increase
    if month in [10, 11, 12, 1]:
        seasonal_multiplier = 1.25

    if sales_type == "fast":
        qty = random.randint(5, 25)
    elif sales_type == "medium":
        qty = random.randint(1, 8)
    else:
        qty = random.randint(0, 2)

    return max(0, round(qty * seasonal_multiplier))


# GENERATE 12-MONTH SALES DATA
def generate_dataset():
    rows = []

    start_date = datetime(2025, 1, 1)
    num_days = 365

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)

        for product in products:
            qty = generate_quantity(
                product["sales_type"],
                current_date.month
            )

            # Random no-sale days
            if random.random() < 0.15:
                qty = 0

            if qty > 0:
                revenue = qty * product["price"]

                rows.append({
                    "date": current_date.date(),
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "category": product["category"],
                    "unit_price": product["price"],
                    "quantity_sold": qty,
                    "revenue": revenue,
                    "month": current_date.month,
                    "day_of_week": current_date.weekday()
                })

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset generated successfully")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print("\nSample data:")
    print(df.head())


# RUN SCRIPT
if __name__ == "__main__":
    generate_dataset()