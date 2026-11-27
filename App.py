import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title='Salary Data',layout='centered')
model=joblib.load('salary_training_model.pkl')
st.title("Salary Prediction")
st.write('Predict the salary based on the experience')
st.sidebar.header('Emloyee Data')
exp=st.sidebar.number_input('Years of Experience',min_value=20,max_value=60,step=1)
if st.button('Predict salary'):
    input=np.array([[exp]])
    pred=model.predict(input)
    #To display function
    st.success(f"Predicted salary={pred}")
    