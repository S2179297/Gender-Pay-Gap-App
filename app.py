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

# Full dropdown values extracted from dataset
job_titles = ['Business Analyst', 'Chief Officer', 'Consultant', 'DBA/Database Engineer', 'Data Analyst', 'Data Engineer', 'Data Journalist', 'Data Scientist', 'Developer Advocate', 'Manager', 'Marketing Analyst', 'Other', 'Principal Investigator', 'Product/Project Manager', 'Research Assistant', 'Research Scientist', 'Salesperson', 'Software Engineer', 'Statistician', 'Student']
industries = ['Academics/Education', 'Accounting/Finance', 'Broadcasting/Communications', 'Computers/Technology', 'Energy/Mining', 'Government/Public Service', 'Hospitality/Entertainment/Sports', 'I am a student', 'Insurance/Risk Assessment', 'Manufacturing/Fabrication', 'Marketing/CRM', 'Medical/Pharmaceutical', 'Military/Security/Defense', 'Non-profit/Service', 'Online Business/Internet-based Sales', 'Online Service/Internet-based Services', 'Other', 'Retail/Sales', 'Shipping/Transportation']
ml_exps = ['1-2 years', '10-15 years', '2-3 years', '20+ years', '3-4 years', '4-5 years', '5-10 years', '< 1 year', 'I have never studied machine learning and I do not plan to', 'Other']
role_exps = ['1-2 years', '10-15 years', '15-20 years', '2-3 years', '20-25 years', '25-30 years', '3-4 years', '30+ years', '4-5 years', '5-10 years', '< 1 year', 'Other']

# App layout
st.title("💼 Gender Pay Gap Predictor")
st.write("Use the Post-Double LASSO model to estimate compensation based on key factors.")

with st.form("input_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    job_title = st.selectbox("Job Title", job_titles)
    ml_exp = st.selectbox("ML Experience", ml_exps)
    role_exp = st.selectbox("Role Experience", role_exps)
    industry = st.selectbox("Industry", industries)
    submitted = st.form_submit_button("Predict Salary")

if submitted:
    # Encode user inputs
    data = {
        f"Job_Title_{job_title}": 1,
        f"ML_Experience_{ml_exp}": 1,
        f"Role_Experience_{role_exp}": 1,
        f"Industry_{industry}": 1,
        "Gender": 1 if gender == "Male" else 0
    }

    # Create DataFrame with all expected columns
    input_df = pd.DataFrame([data])
    for col in one_hot_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[one_hot_columns]

    # Predict and display
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Compensation: USD {prediction:,.2f}")

# Show features
st.write("Top 10 features used by the model:")
st.write(selected_features)
