import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import pickle as pkl
# import tensorflow as tf

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Set up the home page layout
st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:", layout="wide")
st.title("Heart Disease Prediction")

# Load the heart disease dataset
heart_df = pd.read_csv("Desktop/heart/heart_2020_cleaned.csv")

# Define a function to preprocess the data
# def preprocess_data(df):
#     X = df.drop("target", axis=1)
#     y = df["target"]
#     return X, y

# def clean_data_label(df):
#     obj_list = df.select_dtypes(include='object').columns
#     le = LabelEncoder()
#     for obj in obj_list:
#         df[obj] = le.fit_transform(df[obj].astype(str))
#     col_list = df.columns
#     x = df.values #returns a numpy array
#     # min_max_scaler = preprocessing.MinMaxScaler()
#     # x_scaled = min_max_scaler.fit_transform(x)
#     x_scaled = x
#     df = pd.DataFrame(data = x_scaled, columns = col_list)
#     return df

# df = clean_data_label(heart_df)

# def under_sample(df, x = 1):
#     disease = df[df['HeartDisease'] == 1]
#     no_disease = df[df['HeartDisease'] == 0]
#     no_disease = no_disease.sample(n = x * len(disease), random_state=101)
#     df = pd.concat([disease,no_disease],axis=0)
#     return df
# df = under_sample(df)

# X_under = np.array(df.drop('HeartDisease', axis = 1))
# Y_under = np.array(df.HeartDisease)

# X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size = 0.1, random_state=42)

# model = Sequential()
# model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# lr = 0.001
# adam = tf.keras.optimizers.Adam(learning_rate = lr)

# model.compile(loss = 'binary_crossentropy', optimizer= adam, metrics = ['accuracy','Precision','Recall'])

# model.fit(X, y, epochs = 50, batch_size = 32)

# Preprocess the data

# from sklearn.ensemble import RandomForestClassifier
# rf=RandomForestClassifier()

# rf.fit(X_train,y_train)


rf=pkl.load(open('Desktop/heart/modelrf.pkl','rb'))

# Define a function to make predictions
def make_prediction(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer ):
    # Create a new DataFrame with the input data
    input_data = pd.DataFrame({
        "BMI": [BMI],
        "Smoking": [Smoking],
        "AlcoholDrinking": [AlcoholDrinking],
        "Stroke": [Stroke],
        "PhysicalHealth": [PhysicalHealth],
        "MentalHealth": [MentalHealth],
        "DiffWalking": [DiffWalking],
        "Sex": [Sex],
        "AgeCategory": [AgeCategory],
        "Race": [Race],
        "Diabetic": [Diabetic],
        "PhysicalActivity": [PhysicalActivity],
        "GenHealth": [GenHealth],
        "SleepTime": [SleepTime],
        "Asthma": [Asthma],
        "KidneyDisease": [KidneyDisease],
        "SkinCancer": [SkinCancer]
    })

    # Make a prediction using the trained model
    prediction = rf.predict(input_data)
    probability = rf.predict_proba(input_data)

    return prediction, probability


# Define the user interface
st.sidebar.subheader("Patient Information")
BMI = st.sidebar.slider("BMI", 12, 100, 28)
Smoking = st.sidebar.selectbox("Smoker", ["Yes", "No"])
if Smoking == "Yes":
    Smoking = 1
else:
    Smoking = 0
AlcoholDrinking = st.sidebar.selectbox("Alcohol Consumption", ["Yes", "No"])
if AlcoholDrinking == "Yes":
    AlcoholDrinking = 1
else:
    AlcoholDrinking = 0 
Stroke  = st.sidebar.selectbox("Stroke", ["Yes", "No"])
if Stroke  == "Yes":
    Stroke  = 1
else:
    Stroke  = 0 
PhysicalHealth = st.sidebar.slider("Physical Health", 0, 30, 3)

MentalHealth = st.sidebar.slider("Mental Health", 0, 30, 4)

DiffWalking = st.sidebar.selectbox("Difficulty Walking", ["Yes", "No"])
if DiffWalking  == "Yes":
    DiffWalking  = 1
else:
    DiffWalking  = 0 

Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
if Sex == "Male":
    Sex = 1
else:
    Sex = 0
age_dict = {'55-59': 7, '80 or older': 12, '65-69': 9, '75-79': 11, '40-44': 4, '70-74': 10, '60-64': 8, '50-54': 6, '45-49': 5, '18-24': 0, '35-39': 3, '30-34': 2, '25-29': 1}
AgeCategory = st.sidebar.selectbox("Age Category", ['55-59',
 '80 or older',
 '65-69',
 '75-79',
 '40-44',
 '70-74',
 '60-64',
 '50-54',
 '45-49',
 '18-24',
 '35-39',
 '30-34',
 '25-29',])
AgeCategory = age_dict[AgeCategory]

race_dict = {"White" : 5, "Other" : 4, "Hispanic" : 3, "Black" : 2, "Asian" : 1, "American Indian/Alaskan Native" : 0}
Race = st.sidebar.selectbox("Race", ["White", "Other", "Hispanic", "Black", "Asian", "American Indian/Alaskan Native"])
Race = race_dict[Race]

diabetes_dict = {"Yes (during pregnancy)" : 3, "Yes" : 2, "No, borderline diabetes" : 1, "No" : 0}
Diabetic = st.sidebar.selectbox("Diabetes", ["Yes (during pregnancy)", "Yes", "No, borderline diabetes", "No"])
Diabetic = diabetes_dict[Diabetic] 

PhysicalActivity = st.sidebar.selectbox("Physical Activity", ["Yes", "No"])
if PhysicalActivity  == "Yes":
    PhysicalActivity  = 1
else:
    PhysicalActivity  = 0 

gen_health = {'Excellent' : 0, 'Very good' : 4, 'Fair' : 1, 'Good' : 2, 'Poor' : 3 }
GenHealth = st.sidebar.selectbox("General Health", ['Excellent', 'Very good', 'Fair', 'Good', 'Poor'])
GenHealth = gen_health[GenHealth] 

SleepTime = st.sidebar.slider("Time Slept in a Day", 0, 24, 7)

Asthma = st.sidebar.selectbox("Asthma", ["Yes", "No"])
if Asthma  == "Yes":
    Asthma  = 1
else:
    Asthma  = 0 
    
KidneyDisease = st.sidebar.selectbox("Kidney Disease", ["Yes", "No"])
if KidneyDisease  == "Yes":
    KidneyDisease  = 1
else:
    KidneyDisease  = 0  

SkinCancer = st.sidebar.selectbox("SkinCancer", ["Yes", "No"])
if SkinCancer  == "Yes":
    SkinCancer  = 1
else:
    SkinCancer  = 0 

submit_button = st.sidebar.button("Predict")

# Make a prediction when the user clicks the "Predict" button
if submit_button:
    prediction, probability = make_prediction(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer )

    # Show the prediction and the probability of having heart disease
    st.subheader("Prediction")
    if prediction == 0:
        st.write("The patient is **not** likely to have heart disease.")
    else:
        st.write("The patient is **likely** to have heart disease.")
    # st.write("Probability: {:.2f}%".format(probability * 100))


    #Some information about Heart Disease
    
    st.subheader("Heart Attack Causes")
    st.write('''Heart attacks are caused by narrowing of the coronary arteries, which can lead to fat, calcium, proteins, and inflammatory cells building up in the arteries and forming plaques. Platelets (disc-shaped things in your blood) come to the area, and blood clots form around the plaque. If a blood clot blocks the artery, the heart muscle becomes starved for oxygen and the muscle cells die, causing permanent damage. The heart muscle starts to heal soon after a heart attack, but the new scar tissue doesn't move the way it should, so the ability to pump is affected.
How much that ability to pump is affected depends on the size and location of the scar.
''')
    
    st.subheader("Symptoms")
    st.write('''Typical symptoms of an underlying cardiovascular issue include:

    - pain or pressure in the chest, which may indicate angina
    - pain or discomfort in the arms, left shoulder, elbows, jaw, or back
    - shortness of breath
    - nausea and fatigue
    - lightheadedness or dizziness
    - cold sweats
''')

    st.subheader("Prevention")
    st.write('''The same lifestyle changes used to manage heart disease may also help prevent it. Try these heart-healthy tips:
    
    - Don't smoke.
    - Eat a diet that's low in salt and saturated fat.
    - Exercise at least 30 minutes a day on most days of the week.
    - Maintain a healthy weight.
    - Reduce and manage stress.
    - Control high blood pressure, high cholesterol and diabetes.
    - Get good sleep. Adults should aim for 7 to 9 hours daily.
''')
 

    # Show some statistics about the dataset
    st.subheader("Dataset Statistics")
    st.write("Number of rows : {}".format(heart_df.shape[0]))
    st.write("Number of columns : {}".format(heart_df.shape[1]))


    # st.write("Number of patients with heart disease: {}".format(heart_df["HeartDisease"].sum()))
    # st.write("Number of patients without heart disease: {}".format(heart_df.shape[0] - heart_df["HeartDisease"].sum()))



    # Show a histogram of the age distribution
    # st.subheader("Age Distribution")
    # st.bar_chart(heart_df["AgeCategory"])

    # Show a correlation matrix
    st.subheader("Correlation Matrix")
    st.write("This matrix shows how each feature is correlated with each other feature.")
    st.write(heart_df.corr(method='pearson'))
