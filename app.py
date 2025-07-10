import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and required metadata
with open("model/final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("data/one_hot_columns.pkl", "rb") as f:
    one_hot_columns = pickle.load(f)

# App title
st.title("💼 Gender Pay Gap Predictor")

st.write("Fill in the details below to estimate expected compensation based on the trained Post-Double LASSO model.")

# Input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    coding_exp = st.slider("Years of Coding Experience", 0, 50, 5)
    edu_level = st.selectbox("Education Level", ["Bachelor’s degree", "Master’s degree", "PhD", "Other"])
    role_exp = st.slider("Years of Role Experience", 0, 50, 5)
    job_title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "ML Engineer", "Data Analyst", "Other"])
    country = st.selectbox("Country", ["United States", "India", "United Kingdom", "Other"])

    submitted = st.form_submit_button("Predict Salary")

# Run prediction
if submitted:
    input_data = {
        "Years_of_Coding_Experience": coding_exp,
        "Years_of_Role_Experience": role_exp,
        "Gender": 1 if gender == "Male" else 0,
        "Education_Level_" + edu_level: 1,
        "Job_Title_" + job_title: 1,
        "Country_" + country: 1
    }

    input_df = pd.DataFrame([input_data])

    # Ensure all required columns exist
    for col in one_hot_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Match training column order
    input_df = input_df[one_hot_columns]

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Compensation: USD {prediction:,.2f}")
