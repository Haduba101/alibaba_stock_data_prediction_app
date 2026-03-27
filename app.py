
# import the necessary libraries

import os

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the features that were used for training (must match the order)
features_list = [
    'Open', 'High', 'Low', 'Volume',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
    'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_4', 'Volume_Lag_5',
    'Close_MA_7', 'Close_Std_7', 'Volume_MA_7',
    'Close_MA_30', 'Close_Std_30', 'Volume_MA_30',
    'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'WeekOfYear'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all required features are present, fill missing with 0 or a reasonable default
        for feature in features_list:
            if feature not in input_df.columns:
                input_df[feature] = 0 # Or a more sophisticated default/imputation

        # Reorder columns to match the training order
        input_df = input_df[features_list]

        # Scale the input features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = svm_model.predict(input_scaled)

        return jsonify({'predicted_close_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "Welcome to the Stock Price Prediction API! Send POST requests to /predict with stock data."

    # Run Flask app
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)