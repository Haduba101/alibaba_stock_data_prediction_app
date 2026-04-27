
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# API URL (your deployed Render app)
API_URL = "https://alibaba-stock-data-prediction-app-xif8.onrender.com/predict"

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Prediction App")
st.write("Predict the next closing price using your trained ML model")

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("Enter Stock Data")

col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price", value=150.0)
    high_price = st.number_input("High Price", value=155.0)

with col2:
    low_price = st.number_input("Low Price", value=149.0)
    volume = st.number_input("Volume", value=1000000)

# Date input
date = st.date_input("Select Date", datetime.today())

# ---------------------------
# Feature Engineering (basic)
# ---------------------------
year = date.year
month = date.month
day = date.day
day_of_week = date.weekday()
day_of_year = date.timetuple().tm_yday
week_of_year = date.isocalendar()[1]

# Placeholder values for advanced features
# (You can improve this later with real historical data)
input_data = {
    "Open": open_price,
    "High": high_price,
    "Low": low_price,
    "Volume": volume,

    # Lag features (dummy for now)
    "Close_Lag_1": open_price,
    "Close_Lag_2": open_price,
    "Close_Lag_3": open_price,
    "Close_Lag_4": open_price,
    "Close_Lag_5": open_price,

    "Volume_Lag_1": volume,
    "Volume_Lag_2": volume,
    "Volume_Lag_3": volume,
    "Volume_Lag_4": volume,
    "Volume_Lag_5": volume,

    # Moving averages (dummy)
    "Close_MA_7": open_price,
    "Close_Std_7": 0,
    "Volume_MA_7": volume,

    "Close_MA_30": open_price,
    "Close_Std_30": 0,
    "Volume_MA_30": volume,

    # Date features
    "Year": year,
    "Month": month,
    "Day": day,
    "DayOfWeek": day_of_week,
    "DayOfYear": day_of_year,
    "WeekOfYear": week_of_year
}

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Closing Price"):
    try:
        with st.spinner("Making prediction..."):
            response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()

            if "predicted_close_price" in result:
                prediction = result["predicted_close_price"]

                st.success(f"💰 Predicted Close Price: ${prediction:.2f}")

                # Show input data
                st.subheader("Input Summary")
                st.write(pd.DataFrame([input_data]))

            else:
                st.error(result.get("error", "Unknown error"))

        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Request failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built with Streamlit + Flask API 🚀")

