import streamlit as st
import airbnb_class as ac
import pandas as pd
import numpy as np


#ac.airbnb_city()

st.set_page_config(page_title="Predict your house", page_icon="🏨")

tab_model, tab_mapas = st.tabs(["Predictive model","map of yuor city"])

with tab_model:

    st.title('Predictive model')

    # We need a method to call the DataFrame of a city

    city = st.selectbox('Select your city', options = ["Madrid", "Barcelona", "London"])

    #neighbourhood = st.selectbox('Select the neighbourhood', options = df[df["city"]==city]["neighbourhood_cleansed"].unique())

    host_total_listings_count = st.slider("Selectyour listings counts", 0, 100)

    accommodates =st.slider("Select the number of acomodates", 0, 15)

    bathrooms_text = st.slider("Select the number of bathrooms", 0, 5)

    bedrooms = st.slider("Select number of bedrooms", 0, 10)

    beds = st.slider("Select number of bed", 0, 20)

    minimum_nights = st.slider("Select minimun nights", 0, 365)

    maximum_nights = st.slider("Select maximun nights", 0, 365)

    availability_365 = st.slider("Select the days it will be avaliable in a year", 0, 365)

    number_of_reviews = st.slider("Select the number of reviews you have in air bnb", 0, 10000)

    reviews_per_month = st.slider("Select the number of reviews per month you have in air bnb", 0, 50)

    

    st.subheader("Amenities")

    col_1, col_2 = st.columns(2)

    check_amenitie0 = col_1.checkbox("Long term stays allowed",help="")
    check_amenitie1 = col_1.checkbox("Cooking basics",help="")
    check_amenitie2 = col_1.checkbox("Dishes and silverware",help="")
    check_amenitie3 = col_1.checkbox("Essentials",help="")
    check_amenitie4 = col_1.checkbox("Coffee maker",help="")

    
    check_amenitie5 = col_2.checkbox("Hair dryer",help="")
    check_amenitie6 = col_2.checkbox("Microwave",help="")
    check_amenitie7 = col_2.checkbox("Refrigerator",help="")
    check_amenitie8 = col_2.checkbox("Heating",help="")
    check_amenitie9 = col_2.checkbox("Air conditioning",help="")


l_user_features = [neighbourhood, city, accommodates, bathrooms_text, bedrooms, beds, minimum_nights, maximum_nights, availability_365, number_of_reviews, reviews_per_month, host_total_listings_count, check_amenitie0, check_amenitie1, check_amenitie2, check_amenitie3, check_amenitie4, check_amenitie5, check_amenitie6, check_amenitie7, check_amenitie8, check_amenitie9]




















#with open("model.sav", "rb") as file:
#    model = pickle.load(file)