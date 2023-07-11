import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Set up the home page layout
st.set_page_config(page_title="Disease Prediction", page_icon=":doctor:", layout="wide")
st.title("Disease Prediction")


#taking the dataset stored as csv file as input and storing it in a pandas df
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))


temp = pd.get_dummies(df.loc[:, df.columns != "Disease"], prefix = '', prefix_sep = '')
col_names = []
for x in temp:
    if x not in col_names:
        col_names.append(x)
temp = temp.groupby(by=temp.columns, axis = 1).sum()
Y = df.Disease
disease_list = list(Y.unique())
disease_dict = {}
for i in range(len(disease_list)):
    disease_dict[disease_list[i]] = i
disease_names = disease_dict.keys()
disease_labels = disease_dict.values()
symptom_list = list(temp.columns)
df1=pd.concat([Y,temp],axis=1)
Y = Y.map(disease_dict)
X = temp
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
# classifier = RandomForestClassifier(criterion = "entropy", n_estimators = 2)
# classifier.fit(X_train,Y_train)
disease_dict = dict(zip(disease_labels,disease_names))

classifier = pkl.load(open(os.path.join(os.path.dirname(__file__), "model_disease_pred.pkl"),'rb'))


def input_to_df(l):
    for i in range(len(l)):
        sym = l[i]
        sym = sym.lower()
        sym = sym.replace(' ', '_')
        l[i] = ' ' + sym
    #st.write(l)
    x_df = {}
    for i in symptom_list:
        x_df[i] = 0
    for i in l:
        if i in symptom_list:
            #st.write(i)
            x_df[i] = 1
            
    return pd.DataFrame.from_dict([x_df])

# Define a function to make predictions
def make_prediction(l):
    # Create a new DataFrame with the input data
    input_data = input_to_df(l)
    # Make a prediction using the trained model
    y_test_input = disease_dict[classifier.predict(input_data)[0]]

    return y_test_input.strip()

select_symptom = ["None"] + symptom_list

for i in range(1, len(select_symptom)):
    sym = select_symptom[i][1:]
    if sym == "tching":
        sym = 'i' + sym
    sym = sym.replace("_"," ")
    select_symptom[i] = sym.capitalize()


# Define the user interface
st.sidebar.subheader("Patient Symptoms\n")
symptom_1 = st.sidebar.selectbox("Symptom 1", select_symptom)
symptom_2 = st.sidebar.selectbox("Symptom 2", select_symptom)
symptom_3 = st.sidebar.selectbox("Symptom 3", select_symptom)
symptom_4 = st.sidebar.selectbox("Symptom 4", select_symptom)
symptom_5 = st.sidebar.selectbox("Symptom 5", select_symptom)
symptom_6 = st.sidebar.selectbox("Symptom 6", select_symptom)
dl = [symptom_1, symptom_2, symptom_3, symptom_4, symptom_5, symptom_6]
none_c = 0
for i in dl:
    if i == "None":
        none_c += 1
submit_button = st.sidebar.button("Predict")

# Make a prediction when the user clicks the "Predict" button


if submit_button and none_c > 3:
    st.write("\n **Warning : Please Enter more symptoms**\n")


if submit_button and none_c <= 3:

    prediction = make_prediction(dl)


    # Show the prediction and the probability of having heart disease
    st.subheader("Prediction")
    st.write("The patient is most likely to have **{}**\n\n".format(prediction))

    st.write()

    import requests

    url = "https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"

    page_title = prediction # replace with the title of the page you want to retrieve

    st.subheader("\nInformation about {}".format(page_title))

    response = requests.get(url.format(page_title=page_title))

    if response.status_code == 200:
        data = response.json()
        summary = data['extract']
        st.write(summary)
    else:
        st.write("Error: Could not retrieve page summary")

    # Show some statistics about the dataset
    st.subheader("Dataset Statistics")
    st.write("Number of rows : {}".format(df.shape[0]))
    st.write("Number of columns : {}\n".format(df.shape[1]))
    st.subheader("List of Diseases in Dataset : ")
    c = 1
    for i in disease_list:
        st.write(c,".",i)
        c += 1

    st.subheader("\n\n**Dataset**")
    st.write(df)

