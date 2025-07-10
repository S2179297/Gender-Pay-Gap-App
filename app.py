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
    job_title = st.selectbox("Job Title", ["Chief Officer", "Other"])
    ml_exp = st.selectbox("ML Experience", [
        "< 1 year", "1-2 years", "3-4 years", "5-7 years",
        "8-10 years", "10-15 years", "15-20 years", "20+ years"
    ])
    role_exp = st.selectbox("Role Experience", [
        "< 1 year", "1-2 years", "3-4 years", "5-10 years",
        "10-15 years", "15-20 years", "20-25 years", "25-30 years", "30+ years"
    ])
    industry = st.selectbox("Industry", [
        "Accounting/Finance", "Online Business/Internet-based Sales", "Other"
    ])

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    # Manual one-hot encoding
    data = {
        f"Job_Title_{job_title}": 1,
        f"ML_Experience_{ml_exp}": 1,
        f"Role_Experience_{role_exp}": 1,
        f"Industry_{industry}": 1,
        "Gender": 1 if gender == "Male" else 0
    }

    input_df = pd.DataFrame([data])

    # Ensure all required columns are present
    for col in one_hot_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[one_hot_columns]
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Compensation: USD {prediction:,.2f}")

# Show top features used
st.write("Top 10 features used by the model:")
st.write(selected_features)
