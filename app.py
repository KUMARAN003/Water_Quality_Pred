import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
knn_model = joblib.load('knn_model.pkl')

# Streamlit web app
st.title("Water Quality Prediction Web App")

# Sidebar title and user input for features
st.sidebar.title('Water Quality Features')

# User input for features
ph = st.sidebar.slider('pH', min_value=0.0, max_value=14.0, step=0.01, value=7.0)
hardness = st.sidebar.slider('Hardness (mg/L)', min_value=47.0, max_value=323.0, step=0.01, value=200.0)
solids = st.sidebar.slider('Solids', min_value=320, max_value=50000, value=250)
chloramines = st.sidebar.slider('Chloramines (ppm)', min_value=0.0, max_value=15.0, step=0.1, value=1.0)
sulfate = st.sidebar.slider('Sulfate (mg/L)', min_value=120, max_value=500, value=50)
conductivity = st.sidebar.slider('Conductivity (μS/cm)', min_value=181, max_value=760, value=200)
organic_carbon = st.sidebar.slider('Organic Carbon (ppm)', min_value=2.0, max_value=30.0, step=0.1, value=2.0)
trihalomethanes = st.sidebar.slider('Trihalomethanes (μg/L)', min_value=0.0, max_value=124.0, step=0.1, value=30.0)
turbidity = st.sidebar.slider('Turbidity (NTU)', min_value=0.0, max_value=7.0, step=0.01, value=2.0)

# Make prediction
if st.sidebar.button('Predict Potability'):
    # Create a numpy array from user input
    user_input = np.array([ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity])

    # Ensure the model is a KNN model and has a predict method
    if hasattr(knn_model, 'predict') and callable(getattr(knn_model, 'predict')):
        # Reshape the input to match the expected format
        user_input = user_input.reshape(1, -1)

        # Make prediction using the pre-trained model
        result = knn_model.predict(user_input)[0]

        st.title('Water Quality Prediction Results')

        st.subheader('Input Features:')
        input_df = pd.DataFrame(data=[user_input.flatten()], columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
        st.write(input_df)

        st.subheader('Prediction:')
        potability_label = 'Potable' if result == 1 else 'Not Potable'
        st.success(f'The predicted water potability is: {potability_label}')

        st.subheader('Explanation:')
        st.write("The prediction is based on a pre-trained KNN model loaded from a .pkl file.")
    else:
        st.error("The loaded model does not have a valid predict method.")
