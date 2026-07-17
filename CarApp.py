# Q1.
import streamlit as st
import pandas as pd
import joblib 
# Q2.
model = joblib.load('LR_ford_car.pkl')
scaler = joblib.load('scaler.pkl')
encoded_columns = joblib.load('columns.pkl')
# Q3.
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)
# Q4.
st.title("🚗 Ford Car Price Predictor")
st.write("Enter the car details below to predict its selling price.")
st.divider()

# Q5.
st.subheader("📊 Numerical Inputs")
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2018, step=1)
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=40000, step=1000)
    tax = st.number_input("Road Tax", min_value=0, max_value=2000, value=150, step=10)
with col2:
    mpg = st.number_input("MPG", min_value=5.0, max_value=100.0, value=45.0, step=0.1)
    engine_size = st.number_input("Engine Size", min_value=0.5, max_value=6.0, value=1.5, step=0.1)


# Q6.
st.subheader("⚙️ Categorical Inputs")
col3, col4 = st.columns(2)
with col3:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
with col4:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])


# Q7.
st.subheader("🏷️ Model Info")
car_model = st.text_input("Car Model Name", placeholder="e.g. Ford Fiesta")
predict_btn = st.button("Predict Price")