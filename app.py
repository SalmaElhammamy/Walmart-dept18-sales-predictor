import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from feature_engineering import create_simple_features


# 1. Load artifacts
model = joblib.load("rf_all_features_model.pkl")
all_features = joblib.load("model_features_all.pkl")

# 2. Load historical data
df = pd.read_csv("project_dataset.csv", parse_dates=["Date"])
history = df[df["Dept"] == 18].set_index("Date")

# 3. Forecast logic
def recursive_forecast(model, hist_df, n_weeks):
    df2 = hist_df.copy()
    preds = []
    for _ in range(n_weeks):
        feat_row = create_simple_features(df2).dropna().iloc[-1:][all_features]
        yhat = model.predict(feat_row)[0]
        next_idx = df2.index[-1] + pd.Timedelta(weeks=1)
        df2.loc[next_idx] = [
            yhat,
            df2['IsHoliday'][-1],
            df2['Dept'][-1],
            df2['Temperature'][-1],
            df2['Fuel_Price'][-1],
            df2['CPI'][-1],
            df2['Unemployment'][-1]
        ]
        preds.append({"date": next_idx, "predicted_sales": yhat})
    return pd.DataFrame(preds).set_index("date")

# 4. Streamlit UI
st.title("Dept 18 Weekly Sales Forecasting")
n_weeks = st.sidebar.number_input("Weeks to forecast", 1, 12, value=4)
if st.button("Forecast"):
    forecast_df = recursive_forecast(model, history, n_weeks)
    combined = pd.concat([
        history["Weekly_Sales"].rename("Historical"),
        forecast_df["predicted_sales"].rename("Forecast")
    ], axis=1)
    st.line_chart(combined)
    st.table(forecast_df.style.format({"predicted_sales": "${:,.2f}"}))
