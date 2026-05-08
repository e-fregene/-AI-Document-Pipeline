import json
import pandas as pd

# Task 1: Load
with open("src/data/sample.json", "r") as act_file:
    json_activity = json.load(act_file)



#Task 2: Order IDs and totals
order_totals = pd.json_normalize(json_activity, record_path=["orders"])[["order_id", "total"]]
print("Order IDs and Totals:\n", order_totals)



# Task 3: Normalize into flat DataFrame
json_df = pd.json_normalize(
    json_activity,
    record_path=["orders", "items"],
    meta=[
        "customer_id",
        ["orders", "order_id"],
        ["orders", "total"]
    ]
).rename(columns={
    "orders.order_id": "order_id",
    "orders.total": "total"
})[["customer_id", "order_id", "product", "quantity", "total"]]

print(json_df)


# Task 4: Save
json_df.to_json("src/data/processed_orders.json", orient="records", indent=2)
print("Saved to src/data/processed_orders.json")