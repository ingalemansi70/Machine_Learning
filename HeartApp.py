import streamlit as st
import pandas as pd
import joblib

# Q8. LOAD SAVED OBJECTS
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")

# Number input
age = st.number_input("Age", 18, 100, 50)
chol = st.number_input("Cholesterol", 100, 600, 200)
restingBP = st.number_input("Resting BP",100,160,110)
maxHR = st.number_input("Maximum HR",60,200,120)
oldpeak= st.number_input("Oldpeak",-2,6,1)
# Categorical input
sex = st.selectbox("Sex", ["M", "F"])
chestPainType = st.selectbox("Chest pain type",["ATA", "NAP", "ASY", "TA"]) 
exerciseAngina = st.selectbox("Exercise Angina",["Y","N"])
stSlope = st.selectbox("ST Slope",["Flat","Up","Down"])

if st.button("Predict"):
    # 1. Make dataframe from inputs
    sample_data = {"Age": age,"Sex": sex, "Cholesterol": chol,"RestingBP":restingBP,"MaxHR":maxHR,"Oldpeak":oldpeak, "ExerciseAngina":exerciseAngina,"ST_Slope":stSlope,"ChestPainType":chestPainType}
    input_df = pd.DataFrame([sample_data])
    
    # 2. PREPROCESS 
    numerical = ["Age", "RestingBP", "Cholesterol", "MaxHR","Oldpeak"]
    categorical = ["Sex", "ChestPainType", "ExerciseAngina","ST_Slope"]
    
    input_df[numerical] = scaler.transform(input_df[numerical])
    input_df = pd.get_dummies(input_df, columns=categorical, drop_first=True)
    input_df = input_df.reindex(columns=columns, fill_value=0) # Q8 step
    
    # 3. PREDICT
    pred = model.predict(input_df)
    st.write("Result:", "Heart Disease" if pred[0]==1 else "No Heart Disease")