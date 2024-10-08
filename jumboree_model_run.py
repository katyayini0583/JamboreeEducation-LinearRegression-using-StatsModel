import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.header('Welcome to Jumboree Education!!.. Check your chance of admission')

col1,col2 = st.columns(2)

GRE_Score = col1.slider('GRE Score',min_value= 260, max_value = 340)
TOFEL_Score = col2.slider('Tofel Score',min_value= 90, max_value = 120)
Uni_Rating = col1.selectbox('Select University Rating',(1,2,3,4,5))
SOP = 5
LOR = 5
CGPA = col2.number_input('Enter CGPA between 6-10')
Research = col1.selectbox('If you have a research select 1 else 0',(0,1))

# step-1.Input from user
input1 = [GRE_Score,TOFEL_Score,Uni_Rating, SOP, LOR, CGPA, Research]

# step-2.Change input to 2-d array to transform
input1_2d = np.array([input1])

#step-3. Load scaler from pickle
with open('scaled_train.pkl', 'rb') as file:
    scaler = pickle.load(file) 

# Step-4. Transform the input with scaler
scaled_input = scaler.transform(input1_2d)

# Step-5:Load model from pickle
with open('Jumbore_LinReg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Step-6. Predict using model
chance = loaded_model.predict(scaled_input)
if st.button('Submit'):
    st.write(f' Chance of getting seat in Jamboree is: {np.round(chance[0],2)}')

