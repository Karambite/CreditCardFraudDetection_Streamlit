import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
st.header("Credit Card Fraud Detection")

st.subheader("Enter an input array")
input = st.text_input("Enter your input array: ", key="name")


if st.button('Make Prediction'):
    st.write(input)
