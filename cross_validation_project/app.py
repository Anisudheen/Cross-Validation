import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("cross_validation_project/logistic_regression_model.pkl")

model = load_model()

st.title("Income Prediction App 💼")

st.sidebar.header("Enter Details")

# ---- INPUTS ----
age = st.sidebar.number_input("Age", 17, 90, 30)
workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp", "Government"])
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000)

education = st.sidebar.selectbox(
    "Education",
    ["Bachelors", "HS-grad", "Masters", "Doctorate", "Some-college"]
)

education_num = st.sidebar.number_input("Education Num", 1, 16, 9)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Married-civ-spouse", "Never-married", "Divorced", "Separated"]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Other-service"]
)

relationship = st.sidebar.selectbox(
    "Relationship",
    ["Husband", "Wife", "Not-in-family", "Own-child"]
)

race = st.sidebar.selectbox(
    "Race",
    ["White", "Black", "Asian-Pac-Islander", "Other"]
)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

native_country = st.sidebar.selectbox(
    "Native Country",
    ["United-States", "India", "Other"]
)

# ---- CREATE INPUT DATAFRAME (COLUMN NAMES MUST MATCH) ----
input_data = pd.DataFrame({
    "age": [age],
    "workclass": [workclass],
    "fnlwgt": [fnlwgt],
    "education": [education],
    "education.num": [education_num],
    "marital.status": [marital_status],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "sex": [sex],
    "capital.gain": [capital_gain],
    "capital.loss": [capital_loss],
    "hours.per.week": [hours_per_week],
    "native.country": [native_country]
})

st.write("### Input Data")
st.dataframe(input_data)

# ---- PREDICTION ----
if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]

    if prediction == ">50K":
        st.success("✅ Income > 50K")
    else:
        st.warning("⚠️ Income ≤ 50K")
