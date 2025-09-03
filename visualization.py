import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load saved model
model = joblib.load("ltv_model.joblib")

# Load predictions
preds = pd.read_csv("ltv_predictions.csv")

print(preds.head())
import pandas as pd
import matplotlib.pyplot as plt

preds = pd.read_csv("ltv_predictions.csv")

# Segment distribution
preds["segment"].value_counts().plot(kind="bar", figsize=(6,4), color=["#66c2a5","#fc8d62","#8da0cb"])
plt.title("Customer Segments Distribution")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.savefig("segment_distribution.png")
plt.show()
