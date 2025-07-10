import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and metadata
with open("model/final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("data/one_hot_columns.pkl", "rb") as f:
    one_hot_columns = pickle.load(f)

# App title
st.title("💼 Gender Pay Gap Predictor")
st.write("Use the Post-Double LASSO model to estimate compensation based on key factors.")

# Input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    job_title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "ML Engineer", "Data Analyst", "Other"])
    edu_level = st.selectbox("Education Level", ["Bachelor’s degree", "Master’s degree", "PhD", "Other"])
    emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance", "Other"])
    coding_exp = st.slider("Years of Coding Experience", 0, 50, 5)
    role_exp = st.slider("Years in Current Role", 0, 50, 5)
    submitted = st.form_submit_button("Predict Salary")

# On submit, prepare input and predict
if submitted:
    user_input = pd.DataFrame([{
        "Gender": gender,
        "Job_Title": job_title,
        "Education_Level": edu_level,
        "Employment_Type": emp_type,
        "Years_of_Coding_Experience": coding_exp,
        "Years_of_Role_Experience": role_exp
    }])

    # Encode input
    input_encoded = pd.get_dummies(user_input)

    # Match training columns
    for col in one_hot_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[one_hot_columns]

    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Compensation: USD {prediction:,.2f}")

# Show top 10 features
st.write("Top 10 features selected by Post-Double LASSO:")
st.write(selected_features[:10])
