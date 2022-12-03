import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
from tensorflow.keras.models import load_model
import pickle
tf_model = load_model('tf_model.json')
rfc_model = pickle.load(open('RandomForestClassifier_model.json','rb'))

st.header("Credit Card Fraud Detection")

st.subheader("Enter an input array")
input = st.text_input("Enter your input array: ", key="name")


if st.button('Make Prediction'):
    temp_array = input.split(',')
    temp_array = np.array(temp_array, dtype=np.float32)
    tf_pred = tf_model.predict(np.array( [temp_array,]))
    rfc_pred = rfc_model.predict(np.array([temp_array]))
    tf_pred = st.number_input(tf_pred[0])
    rfc_pred = st.number_input(rfc_pred[0])
    st.write("tensorflow Model predicted: " , tf_pred)
    st.write("Random Forest Classifier Model predicted: " , rfc_pred)
