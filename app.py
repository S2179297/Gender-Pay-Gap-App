
import streamlit as st
import pandas as pd
import pickle

# Load model and metadata
model = pickle.load(open("model/final_model.pkl", "rb"))
selected_features = pickle.load(open("model/selected_features.pkl", "rb"))
one_hot_columns = pickle.load(open("data/one_hot_columns.pkl", "rb"))

st.set_page_config(page_title="Gender Pay Gap Predictor", layout="centered")
st.title("💼 Gender Pay Gap Predictor")
st.markdown("Use the Post-Double LASSO model to estimate compensation based on key factors.")

# User input
st.sidebar.header("Input Features")
def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    job = st.sidebar.selectbox("Job Title", ["Data Scientist", "Software Engineer", "Student", "Manager"])
    education = st.sidebar.selectbox("Education Level", ["Bachelor’s", "Master’s", "Doctoral", "Other"])
    coding_exp = st.sidebar.slider("Years of Coding Experience", 0, 30, 3)
    role_exp = st.sidebar.slider("Years in Current Role", 0, 30, 3)
    employment = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Other"])

    return pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Job Title': [job],
        'Education Level': [education],
        'Years of Coding Experience': [coding_exp],
        'Role Experience': [role_exp],
        'Employment Type': [employment]
    })

df = user_input()

# Encode features
input_encoded = pd.get_dummies(df).reindex(columns=one_hot_columns, fill_value=0)

# Predict
if st.button("Predict Salary"):
    st.success(f"💰 Estimated Compensation: USD 88,000 (dummy output)")
    st.write("Top features selected by Post-Double LASSO:")
    st.write(selected_features)
