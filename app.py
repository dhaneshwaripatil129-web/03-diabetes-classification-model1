import numpy as np
import pandas as pd
import streamlit as st
import pickle

# ---------------------- PAGE ----------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------------- LOAD MODEL ----------------------
pipe = pickle.load(open("CPP.pkl","rb"))

# ---------------------- CSS ----------------------
st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg, #E0F7FA, #BBDEFB, #E8F5E9);
}
.main-title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#0b7285;
}
.sub{
text-align:center;
color:gray;
font-size:18px;
margin-bottom:25px;
}
[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0px 5px 10px rgba(0,0,0,.12);
}
.stButton>button{
width:100%;
background:#009688;
color:white;
font-size:20px;
border-radius:10px;
height:55px;
border:none;
}
.stButton>button:hover{
background:#00695c;
color:white;
}
.result{
padding:20px;
border-radius:15px;
font-size:24px;
font-weight:bold;
text-align:center;
}
.good{
background:#2ecc71;
color:white;
}
.bad{
background:#e74c3c;
color:white;
}
</style>
""",unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.markdown("<div class='main-title'>🩺 Diabetes Prediction Dashboard</div>",unsafe_allow_html=True)
st.markdown("<div class='sub'>Machine Learning Based Health Prediction System</div>",unsafe_allow_html=True)

# ---------------------- METRICS ----------------------
m1,m2,m3,m4=st.columns(4)
with m1: st.metric("Model"," Logistic Regression / Pipeline")
with m2: st.metric("Inputs","8")
with m3: st.metric("Prediction","Binary")
with m4: st.metric("Status","Ready")

st.divider()

# ---------------------- INPUTS ----------------------
left,right=st.columns(2)

with left:
    st.subheader("👤 Personal Information")
    name = st.text_input("Enter your Name")
    Age = st.number_input("Age",1,120,3)
    Pregnancies = st.number_input("Pregnancies",0,17,1)

with right:
    st.subheader("🧪 Medical Information")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function",0.0,1.0,0.10,step=0.01)
    BMI = st.slider("BMI",2.0,100.0,6.70,step=0.1)
    Glucose = st.slider("Glucose",0.0,199.0,3.20,step=0.2)
    BloodPressure = st.slider("Blood Pressure",24.0,122.0,66.0,step=2.0)
    SkinThickness = st.slider("Skin Thickness",7,99,35,step=1)
    Insulin = st.slider("Insulin",14,846,89,step=5)

st.divider()

# ---------------------- BUTTON ----------------------
if st.button("🔍 Predict Diabetes"):

    myinput = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                             BMI, DiabetesPedigreeFunction, Age]],
                           columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                    'Insulin','BMI','DiabetesPedigreeFunction','Age'])

    result = pipe.predict(myinput)

    st.divider()
    if result[0]==1:
        st.markdown("<div class='result bad'>⚠️ Yes! You have Diabetes</div>",unsafe_allow_html=True)
        st.progress(90)
        st.warning("Consult a doctor for further diagnosis.")
    else:
        st.markdown("<div class='result good'>✅ No! You don't have Diabetes</div>",unsafe_allow_html=True)
        st.progress(20)
        st.success("Maintain a healthy lifestyle.")
        st.balloons()

st.divider()
st.caption("Developed using ❤️ Streamlit + Machine Learning")
