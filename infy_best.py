import streamlit as st
import joblib
import warnings

warnings.filterwarnings("ignore")
model = joblib.load("my_model.pkl")

st.set_page_config(page_title="Stock Predictor", layout="centered")

st.title("📈 Stock Price Predictor")

with st.form("prediction_form"):
    open_price = st.number_input("OPEN", format="%.2f")
    high_price = st.number_input("HIGH", format="%.2f")
    low_price = st.number_input("LOW", format="%.2f")
    prev_close = st.number_input("PREV. CLOSE", format="%.2f")
    ltp = st.number_input("LTP", format="%.2f")
    close_price = st.number_input("CLOSE", format="%.2f")
    
    submit = st.form_submit_button("Predict")

if submit:
    data = [[open_price, high_price, low_price, prev_close, ltp, close_price]]
    prediction = model.predict(data)[0]
    st.success(f"Predicted Value: {prediction}")