import streamlit as st
import airbnb_class as ac
import pandas as pd
import numpy as np


# load the data

madrid = "datasets/madrid.csv"
barcelona = "datasets/barcelona.csv"
london = "datasets/london.csv"

d_csvs, d_names = dict(), dict()

d_csvs["csvs1"] = [madrid, barcelona]
d_csvs["csvs2"] = [london]

d_names["names1"] = ["madrid","barcelona"]
d_names["names2"] = ["london"]

df = ac.airbnb_city(csvs="barcelona.csv", city_names="barcelona")

df_madbar = df.return_initial_df()

# Set streamlit page

st.set_page_config(page_title="Predict your house", page_icon="🏨")

st.write(df_madbar)



tab_model, tab_mapas = st.tabs(["Predictive model","map of yuor city"])

with tab_model:

    st.title('Predictive model')

    city = st.selectbox('Select your city', options = ["Madrid", "Barcelona", "London"])

    neighbourhood = st.selectbox('Select the neighbourhood', options = df_madbar[df_madbar["city"]==city.lower()]["neighbourhood_cleansed"].unique())

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

# User DataFrame

l_columns = ['neighbourhood_cleansed', 'city', 'accommodates', 'availability_365', 'bathrooms_text', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'reviews_per_month', 
             'host_total_listings_count', 'Long term stays allowed', 'Cooking basics', 'Dishes and silverware', 'Essentials', 'Coffee maker', 'Hair dryer', 'Microwave', 'Refrigerator', 'Heating', 'Air conditioning', 
             'Entire home/apt', 'Private room', 'Shared room']

l_user_features = [neighbourhood, city, accommodates, availability_365, bathrooms_text, bedrooms, beds, minimum_nights, maximum_nights,  number_of_reviews, reviews_per_month, host_total_listings_count,
                   check_amenitie0, check_amenitie1, check_amenitie2, check_amenitie3, check_amenitie4, check_amenitie5, check_amenitie6, check_amenitie7, check_amenitie8, check_amenitie9]

st.write(l_columns[0])

df_user = pd.DataFrame()

for i in range(15):
    df_user[l_columns[i]] = l_user_features[i] 


st.write(df_user)




#model = df.load_model("model", ".sav")


# Prediction


#st.subheader(f"The prediction of the apartament price is: {prediction}")
















#with open("model.sav", "rb") as file:
#    model = pickle.load(file)