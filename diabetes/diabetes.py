import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle as pkl


# Set up the home page layout
st.set_page_config(page_title="Diabetes Prediction", page_icon="::", layout="wide")
st.title("Diabetes Prediction")

# Load the heart disease dataset
diabetes_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "diabetes.csv"))


dt = pkl.load(open(os.path.join(os.path.dirname(__file__), "model_diabetes.pkl"),'rb'))

# Define a function to make predictions
def make_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age ):
    # Create a new DataFrame with the input data
    input_data = pd.DataFrame({
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "BMI": [BMI],
        "Age": [Age]
    })

    # Make a prediction using the trained model
    prediction = dt.predict(input_data)
    #probability = lr.predict_proba(input_data)
    probability = 1

    return prediction, probability


# Define the user interface
st.sidebar.subheader("Patient Information")
st.sidebar.write('''The information of the features in the dataset are mentioned below:
1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. Age: Age (years)''')
Pregnancies = st.sidebar.slider("Pregnancies", 0, 14, 2)

Glucose = st.sidebar.slider("Glucose Level", 40, 300, 100)

BloodPressure = st.sidebar.slider("Blood Pressure", 20, 140, 80)

SkinThickness = st.sidebar.slider("Skin Thickness", 5, 100, 30)

Insulin = st.sidebar.slider("Insulin", 5, 1000, 180)

BMI = st.sidebar.slider("BMI", 10, 100, 32)

Age = st.sidebar.slider("Age", 15, 90, 35)

submit_button = st.sidebar.button("Predict")

# Make a prediction when the user clicks the "Predict" button
if submit_button:
    prediction, probability = make_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age )

    # Show the prediction and the probability of having a stroke
    st.subheader("Prediction")
    if prediction == 0:
        st.write("The patient is **not** likely to have Diabetes.")
    else:
        st.write("The patient is **likely** to have Diabetes.")

    #Some information about Diabetes
    
    st.subheader("About Diabetes")
    st.write('''Diabetes is a chronic health condition characterized by elevated levels of 
    sugar in the blood. There are two types of diabetes: type 1 diabetes and type 2 diabetes. 
    Both types have similar symptoms, causes, and prevention methods.''')

    st.subheader("Diabetes Causes")
    st.write('''The causes of diabetes are complex and multifactorial. Type 1 diabetes is 
    caused by an autoimmune reaction that destroys the insulin-producing cells in the pancreas. 
    Type 2 diabetes is caused by a combination of genetic and lifestyle factors, such as 
    obesity, physical inactivity, and unhealthy eating habits. Other risk factors for developing 
    diabetes include age, family history, high blood pressure, and high cholesterol levels.''')
    
    st.subheader("Symptoms")
    st.write('''The common symptoms of diabetes include frequent urination, excessive thirst, 
    increased hunger, unexplained weight loss, blurred vision, fatigue, slow healing wounds, 
    and numbness or tingling in the hands or feet. People with type 1 diabetes may also 
    experience symptoms such as irritability, mood swings, and increased appetite.''')

    st.subheader("Prevention")
    st.write('''There are several ways to prevent or delay the onset of diabetes. One of the 
    most effective methods is to maintain a healthy lifestyle by exercising regularly, eating 
    a balanced diet rich in fiber, fruits, and vegetables, and avoiding sugary and processed 
    foods. Maintaining a healthy weight, quitting smoking, and reducing alcohol consumption 
    can also help prevent diabetes. For people at high risk of developing diabetes, regular 
    medical check-ups and screening tests can help identify the condition at an early stage, 
    when it is easier to manage.''')

    # Show some statistics about the dataset
    st.subheader("Dataset Statistics")
    st.write("Number of rows : {}".format(diabetes_df.shape[0]))
    st.write("Number of columns : {}".format(diabetes_df.shape[1]))



    # Show a correlation matrix
    st.subheader("Correlation Matrix")
    st.write("This matrix shows how each feature is correlated with each other feature.")
    st.write(diabetes_df.drop(["Outcome"], axis = 1).corr())