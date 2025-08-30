import streamlit as st
import numpy as np
import pickle

# Load model & scaler
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page Config
st.set_page_config(page_title="❤️ Heart Disease Predictor", page_icon="❤️", layout="wide")

# Subtle Red Background
page_bg = """
<style>
.stApp {
    background-color: #374151; /* Neutral Gray */
    color: white;
}
.stButton button {
    background-color: #4B5563; /* Slightly lighter gray for buttons */
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.5rem 1.5rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)



# Title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to check the risk of heart disease.")

# Input form (split in 2 columns for better alignment)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=70, max_value=200, step=1)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, step=1)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, step=1)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox("Slope of Peak Exercise ST", ["Up", "Flat", "Down"])

# Predict button
if st.button("🔮 Predict"):
    # Create input dict same as training columns
    input_dict = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ASY": 1 if chest_pain == "ASY" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "RestingECG_LVH": 1 if rest_ecg == "LVH" else 0,
        "RestingECG_ST": 1 if rest_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }

    # Arrange input features in same order as training
    input_values = []
    for col in scaler.feature_names_in_:
        input_values.append(input_dict.get(col, 0))

    input_data = np.array([input_values])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    result = "🚨 High Risk of Heart Disease" if prediction[0] == 1 else "✅ Low Risk (Healthy)"
    st.markdown(f"<h2 style='color:#fbbf24;'>{result}</h2>", unsafe_allow_html=True)
