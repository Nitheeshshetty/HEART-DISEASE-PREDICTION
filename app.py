import numpy as np
import streamlit as st
import pickle
loaded_model=pickle.load(open('decision_tree.sav','rb'))

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    category=["No, you don't have heart Disease","Yes, you have heart Disease"]
    input_data_prediction_model = loaded_model.predict(input_data_reshaped)
    return category[int(input_data_prediction_model)]
    
def main():
    st.title('Heart Disease Prediction Web App')
      
    Age = st.text_input('Age')
    Sex = st.text_input('Sex')
    Chest_Pain = st.text_input('Chest Pain')
    RestingBP = st.text_input('TResting BP')
    Cholestrol = st.text_input('Cholestrol')
    Fasting_BS = st.text_input('Fasting BS')
    RestingECG = st.text_input('Resting ECG')
    Thalach = st.text_input('Thalach')
    Exercise_Angina = st.text_input('Exercise Angina')
    Old_peak = st.text_input('Old peak')
    ST_Slope = st.text_input('ST Slope')
    ca = st.text_input('ca')
    Thal = st.text_input('Thal')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_prediction([Age, Sex, Chest_Pain, RestingBP, Cholestrol, Fasting_BS, RestingECG, Thalach,Exercise_Angina,Old_peak,ST_Slope,ca,Thal])     
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
