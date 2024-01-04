# Water Quality Prediction

## Overview
This project focuses on predicting the potability of water based on various water quality parameters. The prediction is made using machine learning algorithms.

## Features
The following features are considered for water quality prediction:
- pH (Acidity/Alkalinity)
- Hardness (mg/L)
- Solids
- Chloramines (ppm)
- Sulfate (mg/L)
- Conductivity (μS/cm)
- Organic Carbon (ppm)
- Trihalomethanes (μg/L)
- Turbidity (NTU)

## Machine Learning Model
The machine learning model used for prediction is a KNN .

## Usage
1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit web app:
    ```bash
    streamlit app.py
    ```

3. Use the sliders in the sidebar to input water quality parameters.
4. Click the "Predict Potability" button to see the prediction results.

## Model Training
The SVM model is trained using a cleaned water quality dataset. The dataset is split into training and testing sets, and the features are standardized.

## File Structure
- `app.py`: Streamlit web app for interactive prediction.
- `cleaned_data.csv`: Cleaned water quality dataset.
- `model_training.ipynb`: Jupyter Notebook for model training.
- `requirements.txt`: List of project dependencies.

## Results
The model achieved an accuracy of 64% on the test set.

## Future Improvements
- Explore additional machine learning algorithms for comparison.
- Enhance the user interface for a better user experience.


Feel free to contribute to this project by opening issues or submitting pull requests.
