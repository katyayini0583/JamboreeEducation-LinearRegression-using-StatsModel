import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from sklearn.preprocessing import StandardScaler
st.header('Seat Prediction for Jumboree Education')

input = [[333, 118, 4, 4.5, 4.5, 9.65,1	]]
scaler = StandardScaler()
input = scaler.fit_transform(input)
with open('Jumbore_LinReg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

chance = loaded_model.predict(input)

st.write(f' Chance of getting seat in Jamboree is: {chance}')