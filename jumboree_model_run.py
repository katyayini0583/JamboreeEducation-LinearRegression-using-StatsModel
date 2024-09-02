import pandas as pd
import numpy as np
import pickle 
import streamlit as st

st.header('Seat Prediction for Jumboree Education')

with open('Jumbore_LinReg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

chance = loaded_model.predict(0.389986,	0.602418,	-0.098298,	0.126796,	0.564984,	0.415018,	0.895434)

st.write(f' Chance of getting seat in Jamboree is: {chance}')