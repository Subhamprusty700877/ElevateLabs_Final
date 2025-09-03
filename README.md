# ElevateLabs_Final

:

🛍️ Customer Lifetime Value (CLV) Prediction
📌 Project Overview
This project predicts the Customer Lifetime Value (CLV) based on transaction history.
By analyzing frequency, recency, average order value, and total purchase value, the model estimates long-term customer contribution, enabling targeted marketing and retention strategies.
⚙️ Process / Workflow
Data Collection
Used a transactional dataset (transactions.csv) with customer purchase history (order date, order value).
Data Preprocessing
Converted dates into datetime format.
Cleaned and structured data by customer IDs.
Feature Engineering (RFM-style)
Frequency → Number of purchases per customer.
Recency → Days since last purchase.
Average Order Value (AOV) → Mean order amount.
Total Value → Total spend by each customer.
Model Training
Algorithms: Random Forest Regressor / XGBoost.
Evaluation Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
Saved trained model as ltv_model.joblib.
Prediction & Segmentation
Predicted LTV for all customers → saved as ltv_predictions.csv.
Segmented into Low, Medium, High based on predicted values.
Visualization & Insights
Created plots for segment distribution, LTV ranges, and revenue contribution.
Identified high-value customers (~top 10%) contributing majority of revenue.
📊 Deliverables
✅ ltv_model.joblib → Trained model file
✅ ltv_predictions.csv → Predicted LTV per customer with segment
✅ visualization.py → Script for charts (segment distribution, LTV comparison)
✅ Plots (.png) showing customer insights
