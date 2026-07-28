import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ML Assignment Models", layout="wide")
st.title("🤖 ML Assignment: All Models in 1 App")

tab1, tab2 = st.tabs(["💰 Salary Prediction - Linear Regression", "❤️ Heart Disease - 3 Classifiers"])

@st.cache_resource
def load_all():
    # Salary
    salary_model = joblib.load("linear_salary_model.joblib")
    salary_scaler = joblib.load("linear_scaler.joblib")
    salary_columns = joblib.load("linear_columns.joblib")

    # Heart
    heart_models = {
        "Logistic Regression": joblib.load("heart_logistic_model.joblib"),
        "KNN": joblib.load("heart_knn_model.joblib"),
        "Gaussian Naive Bayes": joblib.load("heart_naive_model.joblib")
    }
    heart_scaler = joblib.load("heart_scaler.joblib")
    heart_columns = joblib.load("heart_columns.joblib")

    return salary_model, salary_scaler, salary_columns, heart_models, heart_scaler, heart_columns

salary_model, salary_scaler, salary_columns, heart_models, heart_scaler, heart_columns = load_all()

# TAB 1: SALARY 
with tab1:
    st.header("💰 AI/ML Salary Prediction")
    col1, col2 = st.columns(2)

    with col1:
        years_exp = st.number_input("Years of Experience", 0, 30, 3)
        job_category = st.selectbox("Job Category", ["Data Scientist", "ML Engineer", "Data Analyst", "Data Engineer"])

    with col2:
        education = st.selectbox("Education", ["Bachelor's", "Master's", "PhD", "Bootcamp/Self-taught"])
        skills = st.selectbox("Required Skills", ["Python", "TensorFlow", "PyTorch", "SQL", "AWS"])

    is_senior = st.checkbox("Is Senior Role?")

    if st.button("Predict Salary", key="salary_btn"):
        input_dict = {
            "years_of_experience": years_exp,
            "job_category": job_category,
            "education_required": education,
            "required_skills": skills,
            "is_senior": int(is_senior)
        }
        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df, columns=["job_category","education_required","required_skills"], drop_first=False)
        input_df[["years_of_experience"]] = salary_scaler.transform(input_df[["years_of_experience"]])
        input_df = input_df.reindex(columns=salary_columns, fill_value=0) 

        pred = salary_model.predict(input_df)[0]
        st.success(f"### Predicted Annual Salary: ${pred:,.2f} USD")

# TAB 2: HEART DISEASE 
with tab2:
    st.header("❤️ Heart Disease Prediction")
    st.write("Choose Logistic, KNN, or Gaussian Naive Bayes")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting BP", 80, 200, 120)
    with col2:
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    with col3:
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, 0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    model_choice = st.selectbox("Choose Model", list(heart_models.keys()))

    if st.button("Predict Heart Disease", key="heart_btn"):
        # Build input dict matching training columns
        input_dict = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([np.zeros(len(heart_columns))], columns=heart_columns)
        for col in input_df.columns:
            if col in input_dict:
                input_df[col] = input_dict[col]

        # Scale numerical
        num_cols = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]
        input_df[num_cols] = heart_scaler.transform(input_df[num_cols])

        model = heart_models[model_choice]
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader(f"Result using {model_choice}")
        if prediction == 1:
            st.error(f"⚠️ Heart Disease Detected | Risk: {proba:.2f}%" if proba else "⚠️ Heart Disease Detected")
        else:
            st.success(f"✅ No Heart Disease | Risk: {(1-proba):.2f}%" if proba else "✅ No Heart Disease")

st.write("---")
st.caption("Note: For educational purposes only. Put all 7.joblib files in the same folder as this app.")