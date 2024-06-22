import streamlit as st
import pandas as pd
from sklearn import datasets
import pickle

st.write("""
# California House Price Prediction Webapp
         
This app predicts the **California House Price**!
""")
st.write('---')

housing = datasets.fetch_california_housing()
x = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.DataFrame(housing.target, columns=['MedHouseVal'])

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    MEDINC = st.sidebar.slider('MEDINC', float(x.MedInc.min()), float(x.MedInc.max()), float(x.MedInc.mean()))
    HOUSEAGE = st.sidebar.slider('HOUSEAGE', float(x.HouseAge.min()), float(x.HouseAge.max()), float(x.HouseAge.mean()))
    AVEROOMS = st.sidebar.slider('AVEROOMS', float(x.AveRooms.min()), float(x.AveRooms.max()), float(x.AveRooms.mean()))
    AVEBEDRMS = st.sidebar.slider('AVEBEDRMS', float(x.AveBedrms.min()), float(x.AveBedrms.max()), float(x.AveBedrms.mean()))
    POP = st.sidebar.slider('POP', float(x.Population.min()), float(x.Population.max()), float(x.Population.mean()))
    AVEOCCUP = st.sidebar.slider('AVEOCCUP', float(x.AveOccup.min()), float(x.AveOccup.max()), float(x.AveOccup.mean()))
    LATITUDE = st.sidebar.slider('LATITUDE', float(x.Latitude.min()), float(x.Latitude.max()), float(x.Latitude.mean()))
    LONGITUDE = st.sidebar.slider('LONGITUDE', float(x.Longitude.min()), float(x.Longitude.max()), float(x.Longitude.mean()))
    data = {
        'MedInc': MEDINC,
        'HouseAge': HOUSEAGE,
        'AveRooms': AVEROOMS,
        'AveBedrms': AVEBEDRMS,
        'Population': POP,
        'AveOccup': AVEOCCUP,
        'Latitude': LATITUDE,
        'Longitude': LONGITUDE
        }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header("Specified User Input Parameters")
st.write(df)
st.write('---')

temp_df = df[['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']]

with open('random_forest_regressor.pkl', 'rb') as file:
    model = pickle.load(file)
prediction = model.predict(temp_df)

st.header('Prediction of MedHouseVal')
st.write(prediction)
st.write('---')
