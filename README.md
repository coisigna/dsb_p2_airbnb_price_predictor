# Introduction

In this Readme we're about to explain all the technical information related with the project development. If you're looking for the fundamentals, motivation or management, [this](https://github.com/coisigna/dsb-p2-ml/wiki) is your site!

We structured this repository separating ipynbs and pys, so you can use all the files in two different ways:

- Colab and try yourself the code used for analyzing, plotting or training the price prediction model.
- Clone it and develop your own webpage using the [Class.py](https://github.com/coisigna/dsb-p2-ml/blob/main/pys/airbnb_class.py) and the [Main.py](https://github.com/coisigna/dsb-p2-ml/blob/main/pys/main.py) with another datasets of different cities.

All of these datasets were downloaded from [Inside Airbnb](http://insideairbnb.com), a web with a lot of information of different airbnbs around the globe, sustained by Murray Cox, John Morris, Taylor Higgins, Alice Corona, Luca Lamonaca and Michael "Ziggy" Mintz, to those whom we greatly appreciate their work.

# EDA notebook

The development of the entire project, depended on the results obtained in the EDA (Exploratory Data Analysis) stage. [Here](https://github.com/coisigna/dsb-p2-ml/blob/main/ipynbs/EDA%20.ipynb) you can find every plot, dataframe, comparatives and conclusions that lead us to create [Class.py](https://github.com/coisigna/dsb-p2-ml/blob/main/pys/airbnb_class.py) and [Main.py](https://github.com/coisigna/dsb-p2-ml/blob/main/pys/main.py) exactly the way we did it.

# Class.py

## \_\_init\_\_

Creates an instance of the class for a list of csvs

#### Create one instance

```python
df5 = airbnb_city(d_csvs["csvs5"],d_names["names5"])
```

#### Create multiple instances at once

```python
d_csvs, d_names = dict(), dict()

d_csvs["csvs1"] = [madrid, barcelona]
d_csvs["csvs2"] = [madrid, barcelona, london]
d_csvs["csvs3"] = [madrid, barcelona, london, paris]
d_csvs["csvs4"] = [madrid, barcelona, london, paris, dublin]
d_csvs["csvs5"] = [madrid, barcelona, london, paris, dublin, rome]
d_csvs["csvs6"] = [madrid, barcelona, london, paris, dublin, rome, amsterdam]
d_csvs["csvs7"] = [madrid, barcelona,london, paris, dublin, rome, amsterdam, athens]
d_csvs["csvs8"] = [madrid, barcelona,london, paris, dublin, rome, amsterdam, athens, oslo]
d_csvs["csvs9"] = [madrid, barcelona,london, paris, dublin, rome, amsterdam, athens, oslo, geneva]
d_csvs["csvs10"] = [madrid, barcelona, paris, london, amsterdam, rome, dublin, geneva, athens, oslo]

d_names["names1"] = ["madrid", "barcelona"]
d_names["names2"] = ["madrid", "barcelona","london"]
d_names["names3"] = ["madrid", "barcelona","london", "paris"]
d_names["names4"] = ["madrid", "barcelona","london", "paris", "dublin"]
d_names["names5"] = ["madrid", "barcelona","london", "paris", "dublin", "rome"]
d_names["names6"] = ["madrid", "barcelona","london", "paris", "dublin", "rome", "amsterdam"]
d_names["names7"] = ["madrid", "barcelona","london", "paris", "dublin", "rome", "amsterdam", "athens"]
d_names["names8"] = ["madrid", "barcelona","london", "paris", "dublin", "rome", "amsterdam", "athens", "oslo"]
d_names["names9"] = ["madrid", "barcelona","london", "paris", "dublin", "rome", "amsterdam", "athens", "oslo", "geneva"]
d_names["names10"] = ["madrid", "barcelona","paris", "london", "amsterdam", "rome","dublin","geneva","athens","oslo"]
    

for i in range(1,10):

    d_dfs[f"instance{i}"] = airbnb_city(d_csvs[f"csvs{i}"],d_names[f"names{i}"])

```

## return_initial_df()

Returns a DataFrame with the csvs passed when the instance was created concatenated. This DataFrame has all the values and columns present in the csvs, without edits.

```python
df5.return_initial_df()
```

## display_initial_df()

Displays a DataFrame with the csvs passed when the instance was created concatenated. This DataFrame has all the values and columns present in the csvs, without edits.

```python
df5.display_initial_df()
```

## clean_columns_tested()

Edits the entire dataframe focusing in the relevant columns studied in the [EDA.ipynb]().

- Drops all columns but: 'neighbourhood_cleansed', 'city', 'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price','minimum_nights', 'maximum_nights', 'availability_365', 'number_of_reviews', 'reviews_per_month', 'host_total_listings_count'
- Price: is given as a string with the $ at the beginning, so it's changed to a float without the $.
- Bathrooms_text: Takes the number of baths from the inside of this string.
- Room_type: Separate it in dummies, and skip Hotel Room.
- Amenities: Takes just the relevant ones from the inside of the list.
- Drop nans

```python
df5.clean_columns_tested()
```

## return_cleaned()

Returns the dataframe cleaned

```python
df5.return_cleaned()
```

## display_cleaned()

Returns the dataframe cleaned

```python
df5.return_cleaned()
```

## remove_outliers()

There's a lot of outliers detailed in the [EDA.ipynb]().

With this method you can try different combinations of values for this column and see which gives you the best results.

Each value of the following columns will be interpreted as <=:

- accommodates
- minimum_nights
- maximum_nights
- nreviews
- reviews_pmonth
- price
- htlc
- bedrooms

```python
df5.remove_outliers(accommodates=8, bathrooms_min=1, bathrooms_max=2, bedrooms=4, beds_min=1, beds_max=5, minimum_nights=30, maximum_nights=500000, nreviews=300, reviews_pmonth=8, price=400, htlc=500000)

# Accommodates will be -> df5[df5["accommodates"] <= 8]
```

The following min values will be interpreted as the first value in a .between() pandas method, max values will be the last value:

- bathrooms_min
- bathrooms_max
- beds_min
- beds_max

```python
df5.remove_outliers(accommodates=8, bathrooms_min=1, bathrooms_max=2, bedrooms=4, beds_min=1, beds_max=5, minimum_nights=30, maximum_nights=500000, nreviews=300, reviews_pmonth=8, price=400, htlc=500000)

# beds_min and beds_max will be -> df5[df5["beds"].between(1,5)]
```

## display_outliers()

Displays a kdeplot for each column after the outliers cleanse.

```python
df5.display_outliers()
```

## label_encoding()

Uses the sklearn.preprocessing LabelEncoder to encode the columns "city" and "neighbourhood_cleansed" so they can be interpreted as numbers.

```python
df5.label_encoding()
```

## normalize()

Normalizes all the columns.

```python
df5.normalize()
```

## tts()

Divides the data in train(80%) and test(20%).

```python
df5.tts()
```

## train_model()

It trains with five different algorithms, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, SVR, AdaBoostRegressor and GradientBoostingRegressor and get the metrics of each.

```python
df5.train_model()
```

## return_metrics()

it returns the metrics of each algorithm. Focused on r**2 and MSE

```python
df5.return_metrics()
```

## display_metrics()

it displays the metrics of each algorithm.

```python
df5.display_metrics()
```

## model_feature_importances()

It gets and returns the feature importance sorted.

```python
df5.model_feature_importances()
```

## grid_search_cv_tuning()

It looks for the best params of the model following the metrics given.

```python
df5.grid_search_cv_tuning()
```
## return_model_result_gcv()

It returns the results of the grid_search.

```python
df5.return_model_result_gcv()
```


## grid_search_cv_validation()

It splits the data in different ways to check the model with different parts of the data in order to return the mean of the metrics and validate the model.

```python
df5.grid_search_cv_validation()
```

## return_validation_gcv()

It returns the results of validation.

```python
df5.return_validation_gcv()
```

## final_trial_model()

It trains the best model with the features recomended.

```python
df5.final_trial_model()
```

## train_final_model()
Returns the definitive model trained with the whole data given the definitive features.
***It must be equalized to the variable model***

```python
df5.train_final_model()
```

## predict()

It makes a prediction given an array with the features introduced by the user.

```python
df5.predict("array")

```

## return_prediction()

It returns the prediction

```python
df5.return_prediction()
```

## save_model()

It saves the model into a file given the name, the extension(.sav recomended) and the model

```python
df5.save_model(name = "modelairbnb", ext = ".sav", model = model)
```

## load_model()

It loads the model in case you want to use it without training it given the name, the extension(.sav recomended) and the model

```python
df5.load_model(name = "modelairbnb", ext = ".sav", model = model)

```

# Main.py

## Load data

It loads every single file the app will need and create instances to call the class

## Adding user features

- house_type: kind of space the user wants to check
- room_type: type of room the user wants to check
- neighbourhood: neighbourhood were the space is located
- host_total_listings_count: numbrer of spaces the host has on airbnb
- accommodates: number of accomodates
- bathrooms: number of bathrooms  
- bedrooms: number of bedrooms
- beds: number of beds 
- minimum_nights: minimun nights it is allowed to stay
- maximum_nights: maximun nights it is allowed to stay
- availability_365: number of days the space will be avaliable in a year
- number_of_reviews: number of reviews on airbnb plataform
- reviews_per_month: number of reviews per month on airbnb plataform
- amenities: amenities that will be available 

## Graphics

- Bar plot of the mean price by district
- Bar plot of the total price by neigbourhood and district
- Bar plot of the mean price by neigbourhood
- Map with a sample of 15 different sapces in the neighbourhood chosen
