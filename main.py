import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Credit Card Fraud Detection")

st.subheader("Please select relevant features of your fish!")
input = st.text_input("Enter your input array: ", key="name")


if st.button('Make Prediction'):
    st.write(input)
