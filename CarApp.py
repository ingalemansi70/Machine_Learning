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
# Q8.
# Q9.
# Q10.
if predict_btn:
    try:
        if car_model.strip() == "":
            st.warning("Please enter a Car Model Name")
        else:
            input_data = {
                'year': year,
                'mileage': mileage,
                'tax': tax,
                'mpg': mpg,
                'engineSize': engine_size,
                'transmission': transmission,
                'fuelType': fuel_type,
                'model': car_model
            }
            input_df = pd.DataFrame([input_data])
            input_df_encoded = pd.get_dummies(input_df)
            input_df_encoded = input_df_encoded.reindex(columns=encoded_columns, fill_value=0)

            num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
            existing_num_cols = [col for col in num_cols if col in input_df_encoded.columns]
            input_df_encoded[existing_num_cols] = scaler.transform(input_df_encoded[existing_num_cols])

            prediction = model.predict(input_df_encoded)
            predicted_price = prediction[0]

            st.divider()
            st.metric(label="Estimated Selling Price", value=f"{predicted_price:,.2f}Pounds.")
            st.info(f"Prediction for: {car_model} | Year: {year}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
