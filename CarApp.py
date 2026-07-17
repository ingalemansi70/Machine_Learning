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