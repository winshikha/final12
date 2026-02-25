import streamlit as st
import joblib
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load trained model
model = joblib.load("my_model.pkl")

st.title("Stock Price Prediction App")

st.write("Enter the stock details below:")

# Input fields
open_price = st.number_input("OPEN")
high_price = st.number_input("HIGH")
low_price = st.number_input("LOW")
prev_close = st.number_input("PREV. CLOSE")
ltp = st.number_input("LTP")
close_price = st.number_input("CLOSE")

# Predict button
if st.button("Predict"):
    data = [[open_price, high_price, low_price, prev_close, ltp, close_price]]
    
    prediction = model.predict(data)[0]
    
    st.success(f"Predicted Value: {prediction}")