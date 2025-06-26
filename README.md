# 🧾 Dept 18 Seasonal Sales Forecasting

Predict weekly sales for Walmart’s seasonal department (Dept 18) using a Random Forest model with a Streamlit user interface.

---

## 🚀 Model Comparison

We evaluated various forecasting approaches using RMSE on Dept 18’s test data:

| Category               | Best Model        | RMSE     | Notes                                                              |
|------------------------|-------------------|----------|--------------------------------------------------------------------|
| Exponential Smoothing  | Holt Damped       | 5,106.67 | Captured slowing trend effectively.                                |
| ARIMA Family           | SARIMA            | 6,035.26 | Outperformed ARIMA & SARIMAX by modeling seasonality.              |
| Machine Learning       | Random Forest     | **1,678.54** | Best performance—captured weekly patterns via feature engineering. |
| Deep Learning          | LSTM              | 7,700.08 | Strongest among DL—handled sequences well.                         |
| Automated Forecasting  | Prophet           | 14,367.80 | Fast but struggled with irregular seasonal spikes.                 |

---

## 🧠 Why Random Forest Won

Used lags (1–4 weeks), moving averages (4, 8, 12 weeks), holiday indicators, month/quarter/week‑of‑year, and trend index.  
These engineered features transform raw time series into structured inputs, enabling Random Forest to model complex, non‑linear patterns—especially critical for capturing Dept 18's seasonal spikes, holiday surges, and trend shifts.

---

## 🚀 Running & Deployment

### Run the Streamlit app locally:
```bash
streamlit run app.py
