#Prepare Libraries
import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

#Setting page
st.set_page_config(page_title="Modelling Page",
                   layout="wide")

#Introduction
st.write("""
         # Welcome to [ERIKA](https://www.linkedin.com/in/erika-budiarti/)'s Machine Learning Dashboard
        """)

st.write("""
        # This app predicts the **Car Mileage**.
        
        Data obtained from the [Car Mileage dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset)
        """)

#Collects user input features into dataframe
#st.sidebar.header('User Input Features:')
#uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#if uploaded_file is not None:
    #input_df = pd.read_csv(uploaded_file)
#else: 
def user_input_features():
        st.sidebar.header('Slide to set the value')
        cylinders = st.sidebar.slider('Number of cylinders in the engine', 0, 10, 0, step=1)
        displacement = st.sidebar.slider('The volume of the piston`s stroke in the engine (cubic inches)', 50, 500, 50, step=25)
        horsepower = st.sidebar.slider('The engine`s power output (hp)', 0, 10, 0, step=1)
        weight = st.sidebar.slider('The weight of the car (pounds)', 1500, 5500, 1500, step=200)
        acceleration = st.sidebar.slider('The time it takes for the car to accelerate from 0 to 60 km/hr (second)', 5, 30, 5, step=5)
        model_year = st.sidebar.slider('The year of manufacture (year 19xx)', 70, 85, 70, step=1)
        origin = st.sidebar.selectbox('The country or region of the car`s origin',("USA", "Europe", "Japan"))
        if origin == "USA":
                origin = 0
        elif origin == "Europe":
                origin = 1
        else:
                origin = 2 
        data = {'cylinders' : cylinders,
                'displacement' : displacement,
                'horsepower' : horsepower,
                'weight': weight,
                'acceleration' : acceleration,
                'model_year' : model_year,
                'origin' : origin}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

# Add picture  
img = Image.open("car_mileage.png")
st.image(img, width=600)

if st.sidebar.button('Click Here to Predict!'):
        with open("best_model_gbr.pkl", 'rb') as file:  
                loaded_model = pickle.load(file)
    
# Extract the input features from input_df
        cylinders = input_df['cylinders'].values[0]
        displacement = input_df['displacement'].values[0]
        horsepower = input_df['horsepower'].values[0]
        weight = input_df['weight'].values[0]
        acceleration = input_df['acceleration'].values[0]
        model_year = input_df['model_year'].values[0]
        origin = input_df['origin'].values[0]

# Create a list of input features for prediction
        input_features = [[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]]

# Make the prediction
        mpg = loaded_model.predict(input_features)
        mpg_formatted = round(mpg[0], 4)

        st.subheader('Prediction Result: ') 
        st.success(f"The mileage of the car (Miles Per Gallon) is: {mpg_formatted} ")
        
st.snow()
