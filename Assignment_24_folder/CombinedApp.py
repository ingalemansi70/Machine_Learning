import streamlit as st
import joblib
import numpy as np
import os

folder = r"C:\Users\user\OneDrive\Desktop\Machine Learning\Assignment_24_folder"

class_model = joblib.load(os.path.join(folder, "best_heart_model.pkl"))
reg_model = joblib.load(os.path.join(folder, "best_ford_model.pkl"))
scaler_c = joblib.load(os.path.join(folder, "scaler_heart.pkl"))
scaler_r = joblib.load(os.path.join(folder, "scaler_ford.pkl"))
class_cols = joblib.load(os.path.join(folder, "heart_columns.pkl"))
reg_cols = joblib.load(os.path.join(folder, "ford_columns.pkl"))

st.title("Session 24 ML Prediction App")
problem = st.selectbox("Select Problem Type", ["Classification", "Regression"])


heart_options = {
    "Sex": ["F", "M"], 
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Down", "Flat", "Up"]
}

if problem == "Classification":
    st.subheader("Heart Disease Prediction")
    inputs = []
    
    cols = st.columns(3)
    for i, col in enumerate(class_cols):
        with cols[i % 3]:
            if col in heart_options:
                val = st.selectbox(col, heart_options[col])
                inputs.append(heart_options[col].index(val))
            else:
                val = st.number_input(col, value=0.0, format="%.2f")
                inputs.append(val)
    
    if st.button("Predict Heart Disease"):
        arr = np.array(inputs).reshape(1, -1)
        arr_s = scaler_c.transform(arr)
        pred = class_model.predict(arr_s)
        prob = class_model.predict_proba(arr_s)[0][1]
        
        if pred[0] == 1:
            st.error(f"Result: Heart Disease - YES | Probability: {prob:.2f}")
        else:
            st.success(f"Result: Heart Disease - NO | Probability: {1-prob:.2f}")

else:
    st.subheader("Car Price Prediction")
    cols = st.columns(3)
    
    with cols[0]:
        year = st.number_input("year", 2010, 2024, 2019)
        mileage = st.number_input("mileage", 0, 200000, 40000)
        tax = st.number_input("tax", 0, 500, 145)
    with cols[1]:
        mpg = st.number_input("mpg", 0.0, 100.0, 50.0)
        engineSize = st.number_input("engineSize", 0.0, 5.0, 1.5)
        transmission = st.selectbox("transmission", ["Manual", "Automatic", "Semi-Auto"])
    with cols[2]:
        fuel = st.selectbox("fuelType", ["Petrol", "Diesel", "Electric"])
    
    transmission_Manual = 1 if transmission=="Manual" else 0
    transmission_Semi_Auto = 1 if transmission=="Semi-Auto" else 0
    fuelType_Electric = 1 if fuel=="Electric" else 0
    
    input_dict = {
        "year": year,
        "mileage": mileage, 
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize,
        "transmission_Manual": transmission_Manual,
        "transmission_Semi-Auto": transmission_Semi_Auto,
        "fuelType_Electric": fuelType_Electric
    }
    
    inputs = [input_dict.get(col, 0) for col in reg_cols]
    
    if st.button("Predict Price"):
        arr = np.array(inputs).reshape(1, -1)
        arr_s = scaler_r.transform(arr)
        pred = reg_model.predict(arr_s)
        st.success(f"Predicted Car Price: $ {pred[0]:,.2f}")