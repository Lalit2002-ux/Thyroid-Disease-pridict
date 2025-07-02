# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 21:36:57 2025

@author: lalit
"""


import numpy as np
import pickle
import streamlit as st

#load the save model
Loaded_model=pickle.load(open('L:/ML DEPLOY/best_model.sav','rb'))




# create a function for prediction
def disease_detection(input_data):
    
    input_data_as_numpy_array=np.asarray(input_data)
    
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
    
    prediction=Loaded_model.predict(input_data_reshape)
    print(prediction)
    
    if(prediction[0]==0):
        return "Result: The individual is likely normal (no thyroid diease detected)."
    else:
        return "Result: The individual shows a chance of thyroid disorder. It is recommended to consult a healthcare professional for further evaluation and confirmation through medical tests."
 



def main():
    
    st.title("Thyroid Prediction ML App")
    
    #getting the input data from user

    age=st.text_input("AGE:")
    TSH=st.text_input("TSH value:")
    T3=st.text_input("T3 value:")
    T4=st.text_input("T4 value:")
    
    diagnose = ""
    
    if st.button("Thyroid Test Result"):
        diagnose = disease_detection([age,TSH,T3,T4])
        
    st.success(diagnose)



if __name__ == "__main__":
    main()