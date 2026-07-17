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