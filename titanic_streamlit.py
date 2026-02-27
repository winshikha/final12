import streamlit as st
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Load Model
# -------------------------------
#p="E:\\robotronix_p\\ML\\deploy_model\\final_titanic_ensemble.pkl"
#model = joblib.load(p)
model = joblib.load("final_titanic_ensemble.pkl")

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter Passenger Details Below:")

# -------------------------------
# User Inputs
# -------------------------------

pclass = st.selectbox("Passenger Class", [1, 2, 3])

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)

sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=10, value=0)

parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10, value=0)

fare = st.number_input("Fare", min_value=0.0, value=50.0)

sex = st.radio("Sex", ["male", "female"])

# -------------------------------
# Feature Engineering
# -------------------------------

family_size = sibsp + parch + 1
sex = 0 if sex == "male" else 1

# Create DataFrame
input_data = [[pclass, age, fare, sex, family_size]]
columns = ["Pclass", "Age", "Fare", "Sex", "FamilySize"]

df1 = pd.DataFrame(input_data, columns=columns)

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict"):
    prediction = model.predict(df1)[0]

    if prediction == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")