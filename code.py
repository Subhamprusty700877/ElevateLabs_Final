# LTV_Prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# ----------------------------
# STEP 1: Load dataset
# ----------------------------
data = pd.read_csv("transactions.csv")

print("✅ Data loaded successfully")
print(data.head())

# ----------------------------
# STEP 2: Feature Engineering
# ----------------------------
# Rename columns for consistency
data = data.rename(columns={"order_date": "transaction_date", "order_value": "transaction_value"})

# Convert date column to datetime
data["transaction_date"] = pd.to_datetime(data["transaction_date"])

# Aggregate features per customer
features = data.groupby("customer_id").agg(
    frequency=("transaction_value", "count"),
    recency=("transaction_date", lambda x: (data["transaction_date"].max() - x.max()).days),
    avg_order_value=("transaction_value", "mean"),
    total_value=("transaction_value", "sum")
).reset_index()

print("✅ Features created successfully")
print(features.head())

# ----------------------------
# STEP 3: Prepare data for training
# ----------------------------
X = features[["frequency", "recency", "avg_order_value", "total_value"]]
y = features["total_value"]  # Using total spend as proxy for LTV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# STEP 4: Train model
# ----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# Save trained model
joblib.dump(model, "ltv_model.joblib")
print("💾 Model saved as ltv_model.joblib")

# ----------------------------
# STEP 5: Make predictions
# ----------------------------
features["predicted_ltv"] = model.predict(X)

# Segment customers
bins = [-1, 1500, 3000, float("inf")]
labels = ["Low", "Medium", "High"]
features["segment"] = pd.cut(features["predicted_ltv"], bins=bins, labels=labels)

# ----------------------------
# STEP 6: Save results
# ----------------------------
features.to_csv("ltv_predictions.csv", index=False)
print("💾 Predictions saved as ltv_predictions.csv")

# ----------------------------
# STEP 7: Summary report
# ----------------------------
print("\n📊 Customer Segmentation Summary:")
print(features.groupby("segment")["predicted_ltv"].agg(["count", "mean", "sum"]))
