import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl


# Set up the home page layout
st.set_page_config(page_title="Stroke Prediction", page_icon="::", layout="wide")
st.title("Stroke Prediction")

# Load the heart disease dataset
stroke_df = pd.read_csv("Desktop/stroke/healthcare-dataset-stroke-data.csv")



lr = pkl.load(open('Desktop/stroke/stroke_lr.pkl','rb'))

# Define a function to make predictions
def make_prediction(Age, Hypertension, HeartDisease, AvgGlucoseLevel, BMI, Sex, Residency, Smoker ):
    # Create a new DataFrame with the input data
    input_data = pd.DataFrame({
        "age": [Age],
        "hypertension": [Hypertension],
        "heart_disease": [HeartDisease],
        "avg_glucose_level": [AvgGlucoseLevel],
        "bmi": [BMI],
        "Sex": [Sex],
        "Residency": [Residency],
        "Smoker": [Smoker]
    })

    # Make a prediction using the trained model
    prediction = lr.predict(input_data)
    #probability = lr.predict_proba(input_data)
    probability = 1

    return prediction, probability


# Define the user interface
st.sidebar.subheader("Patient Information")
Age = st.sidebar.slider("Age", 1, 100, 28)
Hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
if Hypertension == "Yes":
    Hypertension = 1
else:
    Hypertension = 0
HeartDisease = st.sidebar.selectbox("Heart Disease", ["Yes", "No"])
if HeartDisease == "Yes":
    HeartDisease = 1
else:
    HeartDisease = 0
AvgGlucoseLevel = st.sidebar.slider("AVG Glucose Level", 40, 300, 100)
BMI = st.sidebar.slider("BMI", 10, 100, 28)
Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
if Sex == "Male":
    Sex = 1
else:
    Sex = 0
Residency = st.sidebar.selectbox("Residency", ["Urban", "Rural"])
if Residency == "Urban":
    Residency = 1
else:
    Residency = 0
Smoker = st.sidebar.selectbox("Smoker", ['never smoked', 'Unknown', 'formerly smoked', 'smokes'])
if Smoker == 'never smoked':
    Smoker = 2
elif Smoker == 'formerly smoked':
    Smoker = 1
elif Smoker == 'smokes':
    Smoker = 3
else:
    Smoker = 0



submit_button = st.sidebar.button("Predict")

# Make a prediction when the user clicks the "Predict" button
if submit_button:
    prediction, probability = make_prediction(Age, Hypertension, HeartDisease, AvgGlucoseLevel, BMI, Sex, Residency, Smoker )

    # Show the prediction and the probability of having a stroke
    st.subheader("Prediction")
    if prediction == 0:
        st.write("The patient is **not** likely to have a Stroke.")
    else:
        st.write("The patient is **likely** to have a Stroke.")
    # st.write("Probability: {:.2f}%".format(probability * 100))

    #Some information about Stroke
    
    st.subheader("Stroke Causes")
    st.write('''A stroke occurs when the blood supply to the brain is interrupted or reduced, 
    which can cause brain cells to die. The most common causes of strokes are:

    Ischemic stroke: 
    This type of stroke is caused by a blood clot that blocks a blood vessel in 
    the brain, preventing blood and oxygen from reaching the brain tissue.

    Hemorrhagic stroke: 
    This type of stroke is caused by a ruptured blood vessel in the brain, which
    leads to bleeding in the brain tissue and can damage brain cells.

    Transient ischemic attack (TIA): 
    Also known as a mini-stroke, TIA is caused by a temporary disruption of blood 
    flow to the brain, often caused by a blood clot. TIAs typically last only a 
    few minutes and do not cause permanent brain damage.

Other factors that can increase the risk of stroke include:

    - High blood pressure
    - High cholesterol
    - Smoking
    - Diabetes
    - Obesity
    - Physical inactivity
    - Excessive alcohol consumption
    - Atrial fibrillation (a heart rhythm disorder)
    - Family history of stroke or heart disease
    - Age (the risk of stroke increases with age)''')
    
    st.subheader("Symptoms")
    st.write('''Signs and symptoms of stroke include:

    - Trouble speaking and understanding what others are saying. You may experience confusion, 
      slur words or have difficulty understanding speech.
    - Paralysis or numbness of the face, arm or leg. You may develop sudden numbness, weakness
      or paralysis in the face, arm or leg. This often affects just one side of the body. Try 
      to raise both your arms over your head at the same time. If one arm begins to fall, you 
      may be having a stroke. Also, one side of your mouth may droop when you try to smile.
    - Problems seeing in one or both eyes. You may suddenly have blurred or blackened vision 
      in one or both eyes, or you may see double.
    - Headache. A sudden, severe headache, which may be accompanied by vomiting, dizziness or 
      altered consciousness, may indicate that you're having a stroke.
    - Trouble walking. You may stumble or lose your balance. You may also have sudden dizziness
      or a loss of coordination.

''')

    st.subheader("Prevention")
    st.write('''Many stroke prevention strategies are the same as strategies to prevent heart disease. In general, healthy lifestyle recommendations include:

    - Controlling high blood pressure (hypertension). This is one of the most important things you can do to reduce your stroke risk. 
    - Lowering the amount of cholesterol and saturated fat in your diet. 
    - Quitting tobacco use. Smoking raises the risk of stroke for smokers and nonsmokers exposed to secondhand smoke. 
    - Managing diabetes. Diet, exercise and losing weight can help you keep your blood sugar in a healthy range. 
    - Maintaining a healthy weight. Being overweight contributes to other stroke risk factors, such as high blood pressure, cardiovascular disease and diabetes.
    - Eating a diet rich in fruits and vegetables. 
    - Exercising regularly. Aerobic exercise reduces the risk of stroke in many ways. 
    - Drinking alcohol in moderation, if at all. 
      Heavy alcohol consumption increases the risk of high blood pressure, ischemic strokes and hemorrhagic strokes. 
    - Treating obstructive sleep apnea (OSA). 
    - Avoiding illegal drugs. Certain street drugs, such as cocaine and methamphetamine, are established risk factors for a TIA or a stroke.
''')

    # Show some statistics about the dataset
    st.subheader("Dataset Statistics")
    st.write("Number of rows : {}".format(stroke_df.shape[0]))
    st.write("Number of columns : {}".format(stroke_df.shape[1]))



    # Show a correlation matrix
    st.subheader("Correlation Matrix")
    st.write("This matrix shows how each feature is correlated with each other feature.")
    st.write(stroke_df.corr())