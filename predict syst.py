# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 21:32:23 2025

@author: lalit
"""

import numpy as np
import pickle

#load the save model
Loaded_model=pickle.load(open('best_model.sav','rb'))
# ✅ New (relative path for Streamlit Cloud)



def disease_detection():
  def case1():
    print("Enter the details of the Patient")

    a1=(int(input("Age:")))
    a2=(float(input("TSH:")))
    a3=(float(input("T3:")))
    a4=(float(input("T4:")))
   
    print("_")
    input_features=[[a1,a2,a3,a4]]
    if all(val == 0 for val in input_features[0]):
            print("_")
            print("Normal patient")
            return
    G = Loaded_model.predict(input_features)
    print("_")
    print(G)
    if(G==0):
      print(" Result: The individual is likely normal (no thyroid diease detected).")
    else:
      print("Result: The individual shows a chance of thyroid disorder. It is recommended to consult a healthcare professional for further evaluation and confirmation through medical tests.")
  case1()
disease_detection()
