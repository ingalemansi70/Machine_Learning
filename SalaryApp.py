# Q1.
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Q2. Load your files
model = joblib.load("salary_multioutput_model.joblib")
scaler = joblib.load("salary_scaler.joblib")
encoded_columns = joblib.load("salary_model_columns.joblib")

# Q3.
st.set_page_config(
    page_title="Salary Predictor for AI/ML Jobs",
    layout="centered"
)

# Q4.
st.title("🤖 AI/ML Job Salary Predictor")
st.write("Enter the job details below to predict salary in USD.")
st.divider()

# Q5. Numerical Inputs
st.subheader("📊 Inputs")
col1, col2 = st.columns(2)
with col1:
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=15, value=2, step=1)
with col2:
    is_senior = st.selectbox("Senior Role?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# Q6. Categorical Inputs
st.subheader("⚙️ Job Details")
col3, col4 = st.columns(2)
with col3:
    job_category = st.selectbox("Job Category", [
        'AI Engineering', 'Data Engineering', 'Product', 'Security', 'Architecture',
        'ML Operations', 'Business', 'Robotics', 'Data Science', 'Governance',
        'Infrastructure', 'Research'
    ])

with col4:
    education_required = st.selectbox("Education Required", [
        "Master's", "Bachelor's", "Associate's", "Bootcamp/Self-taught", "PhD"
    ])

# For skills: take top 15 most common ones so dropdown isn't 1500 long
st.subheader("🛠️ Skills")
required_skills = st.text_input(
    "Required Skills",
    placeholder="e.g. Python|SQL|Machine Learning|Cloud",
    help="Enter skills separated by |. Must match training format"
)

predict_btn = st.button("Predict Salary")

# Q7. Prediction Logic
if predict_btn:
    try:
        if required_skills.strip() == "":
            st.warning("Please enter at least 1 skill separated by |")
        else:
            # 1. Create input dict with same column names as training
            input_data = {
                'years_of_experience': years_of_experience,
                'is_senior': is_senior,
                'job_category': job_category,
                'education_required': education_required,
                'required_skills': required_skills
            }

            input_df = pd.DataFrame([input_data])

            # 2. One Hot Encode same way as training
            input_df_encoded = pd.get_dummies(input_df)

            # 3. Align columns with training columns. Fill missing with 0
            input_df_encoded = input_df_encoded.reindex(columns=encoded_columns, fill_value=0)

            # 4. Scale numerical columns
            num_cols = ['years_of_experience']
            existing_num_cols = [col for col in num_cols if col in input_df_encoded.columns]
            input_df_encoded[existing_num_cols] = scaler.transform(input_df_encoded[existing_num_cols])

            # 5. Predict
            prediction = model.predict(input_df_encoded)[0] # returns [annual, min, max]

            predicted_annual = prediction[0]
            predicted_min = prediction[1]
            predicted_max = prediction[2]

            st.divider()
            st.subheader("💰 Predicted Salary Range")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Annual Salary", f"${predicted_annual:,.0f}")
            with col_b:
                st.metric("Min Salary", f"${predicted_min:,.0f}")
            with col_c:
                st.metric("Max Salary", f"${predicted_max:,.0f}")

            st.success(f"Prediction for: {job_category} | {education_required} | {years_of_experience} yrs exp")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.exception(e)