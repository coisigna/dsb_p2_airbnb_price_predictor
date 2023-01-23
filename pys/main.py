import streamlit as st
import airbnb_class as ac
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# load the data

madrid = "madrid.csv"
barcelona = "barcelona.csv"
london = "london.csv"

d_csvs, d_names = dict(), dict()

d_csvs["csvs1"] = [madrid, barcelona]
d_csvs["csvs2"] = [london]

d_names["names1"] = ["madrid","barcelona"]
d_names["names2"] = ["london"]



# Set streamlit page

st.set_page_config(page_title="Predict your house", page_icon="🏨", layout= "wide")


tab_model, tab_mapas = st.tabs(["Predictive model","map of yuor city"])

with tab_model:

    st.title('Predictive model')

    city = st.selectbox('Select your city', options = ["Madrid", "Barcelona", "London"])

    if city == "Madrid" or city == "Barcelona":

        city_instance = ac.airbnb(d_csvs["csvs1"], d_names["names1"], "csv")

        df_city = city_instance.return_initial_df()

    else: 

        city_instance = ac.airbnb(d_csvs["csvs2"], d_names["names2"], "csv")

        df_city = city_instance.return_initial_df()

    house_type = st.selectbox('Select the type of the space', options = ['Entire home/apt', 'Private room', 'Shared room'])

    d_room_type = {'Entire home/apt': 0 , 'Private room': 0, 'Shared room' : 0}

    d_room_type[house_type] = 1

    neighbourhood = st.selectbox('Select the neighbourhood', options = df_city[df_city["city"]==city.lower()]["neighbourhood_cleansed"].unique())

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


l_amenities = [check_amenitie0, check_amenitie1, check_amenitie2,check_amenitie3 , check_amenitie4,check_amenitie5 , check_amenitie6, check_amenitie7, check_amenitie8,  check_amenitie9 ]

for enum,i in enumerate(l_amenities):
    if i ==True:
        l_amenities[enum] = 1
    else:
        l_amenities[enum] = 0

# User DataFrame

d_columns = {'neighbourhood_cleansed': neighbourhood, 'city':city, 'accommodates': accommodates, 'availability_365': availability_365, 'bathrooms_text': bathrooms_text, 'bedrooms': bedrooms, 'beds': beds, 'minimum_nights':minimum_nights, 
             'maximum_nights':maximum_nights, 'number_of_reviews': number_of_reviews, 'reviews_per_month':reviews_per_month, 'host_total_listings_count':host_total_listings_count, 'Long term stays allowed':l_amenities[0], 'Cooking basics':l_amenities[1], 
             'Dishes and silverware':l_amenities[2], 'Essentials':l_amenities[3], 'Coffee maker':l_amenities[4], 'Hair dryer':l_amenities[5], 'Microwave':l_amenities[6], 'Refrigerator':l_amenities[7], 'Heating':l_amenities[8], 
             'Air conditioning':l_amenities[9], 'Entire home/apt':d_room_type["Entire home/apt"], 'Private room':d_room_type["Private room"], 'Shared room':d_room_type["Shared room"]}


# User DataFrame

df_user = pd.DataFrame(d_columns.items())
df_user = df_user.T
df_user = df_user.rename(columns=df_user.iloc[0])




if df_user.shape[0]>1:
    df_user.drop(df_user.index[0], inplace=True)

city_instance.clean_tested_columns()

df_city_cleaned = city_instance.return_cleaned()

instance_prediction = ac.airbnb(data= [df_city_cleaned, df_user] , file= "dataframe")

df_prediction = instance_prediction.label_encoding(instance_prediction.return_initial_df())

df_prediction = instance_prediction.normalize(df=df_prediction).tail(1)

df_prediction.drop("price", axis=1, inplace=True)

nparr_prediction = df_prediction.values


check_prediction = st.button("Ready, give me the prediction!!",help="")

if check_prediction:

    model_madrid_barcelona = instance_prediction.load_model("model_madrid_barcelona", "sav")
    model_london = instance_prediction.load_model("model_london", "sav")

    if city == "Madrid" or city == "Barcelona":

        instance_prediction.predict_model(nparr_prediction, model_madrid_barcelona)                                                   #([X_test.iloc[23,:].values], model)

        prediction = instance_prediction.return_prediction()

        st.header(f"The predicted price is: {round(prediction[0][0])}.00 euros")

    else: 

        instance_prediction.predict_model(nparr_prediction, model_london)                                                   #([X_test.iloc[23,:].values], model)

        prediction = instance_prediction.return_prediction()

        st.header(f"The predicted price is: {round(prediction[0][0]/100)}.00 euros")