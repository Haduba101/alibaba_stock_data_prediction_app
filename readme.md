**Alibaba Stock Data Prediction App**

This app will have an endpoint that accepts new stock data,
preprocesses it using the saved scaler, 
makes predictions with the saved SVM model, 
and returns the predicted close price.

# Project structure

ALIBABA/
|-- .venv
    |-- Scripts\activate
|-- Alibaba Stock Data.csv
|-- Alibaba_Stock_Data(1).ipynb
|-- app.py
|-- Procfile
|-- readme.md
|-- requirements.txt
|-- scaler.pkl
|-- svm_model.pkl