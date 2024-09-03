import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.header('Welcome to Jumboree Education!!.. Check your chance of admission')

GRE_Score = st.slider('GRE Score',min_value= 260, max_value = 340)
TOFEL_Score = st.slider('Tofel Score',min_value= 260, max_value = 340)
Uni_Rating = st.selectbox('Select University Rating',(1,2,3,4,5))
SOP = 5
LOR = 5
CGPA = st.number_input('Enter CGPA between 6-10')
Research = st.selectbox('If you have a research select 1 else 0',(0,1))

# step-1.Input from user
input1 = [GRE_Score,TOFEL_Score,Uni_Rating, SOP, LOR, CGPA, Research]

# step-2.Change input to 2-d array to transform
input1_2d = np.array([input1])

#step-3. Load scaler from pickle
with open('scaled_train.pkl', 'rb') as file:
    scaler = pickle.load(file) 

# Step-4. Transform the input with scaler
scaled_input = scaler.transform(input1_2d)
st.write(scaled_input.shape)
scaled_input_2 = np.array([0.389986,0.602418,-0.098298,0.126796,0.564984,0.415018])
st.write(scaled_input_2.shape)

# Step-5:Load model from pickle
with open('Jumbore_LinReg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Step-6. Predict using model
chance1 = loaded_model.predict(scaled_input_2)
st.write(f' Chance of getting seat in Jamboree is: {chance1[0]}')

chance = loaded_model.predict(scaled_input)
if st.button('Submit'):
    st.write(f' Chance of getting seat in Jamboree is: {chance[0]}')

