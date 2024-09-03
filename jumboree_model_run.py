import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from sklearn.preprocessing import StandardScaler
st.header('Seat Prediction for Jumboree Education')

GRE_Score = st.slider('GRE Score',min_value= 260, max_value = 340)
TOFEL_Score = st.slider('Tofel Score',min_value= 260, max_value = 340)
Uni_Rating = st.selectbox('Select University Rating',(1,2,3,4,5))
SOP = 3
LOR = 3
CGPA = st.number_input('Enter CGPA between 6-10')
Research = st.selectbox('If you have a research select 1 else 0',(0,1))
input1 = [GRE_Score,TOFEL_Score,Uni_Rating, SOP, LOR, CGPA, Research]

input1_2d = np.array([input1])
st.write(input1_2d.shape)

#input = pd.DataFrame([333, 118, 4, 4.5, 4.5, 9.65,1])

with open('scaled_train.pkl', 'rb') as file:
    scaler = pickle.load(file) 
st.write(scaler)
st.write('new line')
#st.write(np.array([[333, 118, 4, 4.5, 4.5, 9.65,1]]).shape)
#st.write(scaler.transform([333, 118, 4, 4.5, 4.5, 9.65,1]))

st.write(input)
scaled_input1= scaler.transform(input)
st.write(scaled_input1)

scaled_input = scaler.transform(input1_2d)
st.write(scaled_input)
with open('Jumbore_LinReg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

chance = loaded_model.predict(scaled_input)
if st.button('Submit'):
    st.write(f' Chance of getting seat in Jamboree is: {chance}')

