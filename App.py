import streamlit as st
import joblib
import spacy
from sklearn.feature_extraction.text import CountVectorizer


# Model loading
Model = joblib.load('Email_classifier.joblib')
# vectorizer loading which is fitted on the training data
vectorizer = joblib.load('vectorizer_.joblib')
# title of the web app
st.title("Email_classifier")
# taking the input from the user
user_input = st.text_input("Enter the email for classification : ", 'Enter Email ',key="user_input")
input_email = list(user_input)

# fuction to preprocess the input email --> words to vectors
def preprocess_input(email):
    email_cv = vectorizer.transform(email)
    return email_cv
# if the user clicks on predict button
if st.button('Predict'):

    processed_input = preprocess_input(input_email)

    prediction = Model.predict(processed_input)[0]

    if prediction == 0:
        st.write("Predicted as : Not Spam Email")
    else:
        st.write("Predicted as : Spam Email")


    


 
