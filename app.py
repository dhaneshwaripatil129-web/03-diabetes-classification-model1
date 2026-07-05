import numpy as np
import pandas as pd
import streamlit as st
import pickle

st.title("💊Diabetic Prediction")
st.subheader("📋Your Details:")
st.markdown('---')
pipe=pickle.load(open("CPP.pkl","rb"))
df=pd.read_csv("cleaned_data.csv")


# Simple textbox
name = st.sidebar.text_input("Enter your name")

Pregnancies = st.sidebar.number_input("Enter  Pregnancies count ", min_value=0, max_value=17, value=1,step=1)

DiabetesPedigreeFunction=st.sidebar.number_input("Enter a DiabetesPedigreeFunction", min_value=0.0, max_value= 0.637, value=0.10, step=0.01)

Age=st.sidebar.number_input("Enter your Age", min_value=0, max_value=120, value=3, step=1)

BMI = st.sidebar.slider("Select a BMI", min_value=2.0, max_value=100.0, value=6.70,step=0.1)

Glucose= st.sidebar.slider("Select a Glucose ", min_value=0.0, max_value=199.0, value=0.0,step=0.2)

BloodPressure= st.sidebar.slider("Select a BloodPressure:", min_value=24.0, max_value=122.0, value=66.0,step=2.0)

SkinThickness = st.sidebar.slider("Select a SkinThickness : ", min_value=7, max_value=10, value=9,step=1)

Insulin  = st.sidebar.slider("Select a Insulin :  ", min_value=14, max_value=846, value=54,step=5)



if st.sidebar.button("Predict"):
   

    st.markdown(f"**Patient Name :** {name}")
    st.markdown(f"**Pregnancies :** {Pregnancies}")
    st.markdown(f"**Diabetes Pedigree :** {DiabetesPedigreeFunction}")
    st.markdown(f"**Age :** {Age}")
    st.markdown(f"**BMI :** {BMI}")
    st.markdown(f"**Glucose :** {Glucose}")
    st.markdown(f"**Blood Pressure :** {BloodPressure}")
    st.markdown(f"**Skin Thickness :** {SkinThickness}")
    st.markdown(f"**Insulin :** {Insulin}")

    myinput = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age]]
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age']
    myinput = pd.DataFrame(data=myinput, columns=columns)

    result = pipe.predict(myinput)

    st.markdown("---")
    if result[0] == 1:
        st.error("**💊 Yes! You have Diabetes**")   # red box with bold text
    else:
        st.success("**💊 No! You don't have Diabetes**")  # green box with bold text
        st.balloons()

    
    



