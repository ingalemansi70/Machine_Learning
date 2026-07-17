# Q1.
import streamlit as st
import pandas as pd
import joblib 
# Q2.
model = joblib.load('LR_ford_car.pkl')
scaler = joblib.load('scaler.pkl')
encoded_columns = joblib.load('columns.pkl')
