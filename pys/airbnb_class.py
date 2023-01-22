import pandas as pd
import numpy as np
import re
import json
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import pickle


class airbnb_city:
    
    def __init__(self, csvs, city_names):
        
        if type(csvs) != list:
            
            self.csv = csvs

            self.df = pd.read_csv(self.csv)
            
            self.df["city"] = city_names
            
            print("Instance created!")
            
        else:
            
            for enum, dataset in enumerate(csvs):
    
                dataset.drop("source", axis = 1, inplace = True)
                
                dataset["city"] = city_names[enum].lower()
        
            self.df = pd.concat(csvs)
            
            print("Instance created!")
            
    def return_initial_df(self):
    
        return self.df
    
    def display__initial_df(self):
    
        display(self.df)

    def clean_tested_columns(self):
        
        """
        Sets predefined columns, transforms price to a float column and separates bathroom_text 
        into 3 different categories, private, shared and unknown.
        """
        
        # Sets predefined columns
        
        tested_cols = ['neighbourhood_cleansed', 'city',
                       'room_type', 'accommodates',
                       'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
                       'minimum_nights', 'maximum_nights', 'availability_365',
                       'number_of_reviews', 'reviews_per_month', 'host_total_listings_count']
        
        self.df_cleaned = self.df[tested_cols]
        
        # Transforms price to a float column
        
        self.df_cleaned["price"] = self.df_cleaned["price"].apply(lambda x: float(x.strip("$").replace(',', '')) if pd.notnull(x) else x).values
            
        # Get numbers out of bathroom_text columns
        
        self.df_cleaned = self.df_cleaned[self.df_cleaned["bathrooms_text"].isnull() == False]

        l_nums = [re.findall(r'\d+',i) for i in self.df_cleaned["bathrooms_text"].values]

        l_nums_completed = []

        for i in l_nums:

            if len(i) > 1:

                l_nums_completed.append('.'.join(i))

            elif len(i) == 0:

                l_nums_completed.append('0')

            else:

                l_nums_completed.append(i[0])
                
        # Create two different columns replacing bathroom_text
        
        self.df_cleaned["bathrooms_text"] = l_nums_completed

        self.df_cleaned["bathrooms_text"] = self.df_cleaned["bathrooms_text"].astype("float64")
        
        # Amenities
        
        l_amenities_cleaned = list()
        
        for i in self.df_cleaned["amenities"]:

            l_amenities_cleaned.append(json.loads(i))

        # Most relevant amenities, detailed analysis in the EDA file

        l_amenities_valuables = ['Long term stays allowed','Cooking basics','Dishes and silverware','Essentials','Coffee maker','Hair dryer','Microwave','Refrigerator','Heating','Air conditioning']

        for j in l_amenities_valuables:

            self.df_cleaned[j] = [1 if j in i else 0 for i in l_amenities_cleaned]

        self.df_cleaned.drop("amenities", axis =1, inplace=True)
        
        self.df_cleaned.dropna(inplace = True)
        
        # Room type
        
        self.df_cleaned = self.df_cleaned[self.df_cleaned["room_type"] != "Hotel room"]
        self.df_cleaned = pd.concat([self.df_cleaned, pd.get_dummies(data = self.df_cleaned["room_type"])], axis = 1).drop("room_type", axis = 1)
        
        return self.df_cleaned
        
    def label_encoding(self):
        
        city_encoder = LabelEncoder()
        self.df_cleaned["city"] = city_encoder.fit_transform(self.df_cleaned["city"])
        neighbourhood_encoder = LabelEncoder()
        self.df_cleaned["neighbourhood_cleansed"] = neighbourhood_encoder.fit_transform(self.df_cleaned["neighbourhood_cleansed"])
        
        return self.df_cleaned
    
    def remove_outliers(self, accommodates, bathrooms_min, bathrooms_max, bedrooms, beds_min, beds_max, minimum_nights,
                       maximum_nights, nreviews, reviews_pmonth, price, htlc):

        self.df_cleaned = self.df_cleaned[self.df_cleaned["accommodates"] <= accommodates]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["bathrooms_text"].between(bathrooms_min, bathrooms_max)]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["bedrooms"] <= bedrooms]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["beds"].between(beds_min, beds_max)]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["minimum_nights"] <= minimum_nights]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["maximum_nights"] <= maximum_nights]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["number_of_reviews"] <= nreviews]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["reviews_per_month"] <= reviews_pmonth]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["price"] <= price]
        self.df_cleaned = self.df_cleaned[self.df_cleaned["host_total_listings_count"] <= htlc]

        return self.df_cleaned

    def normalize(self):
        
        x_scaler = MinMaxScaler()
        self.df_cleaned[self.df_cleaned.drop("price", axis = 1).columns] = x_scaler.fit_transform(self.df_cleaned[self.df_cleaned.drop("price", axis = 1).columns])

        y_scaler = MinMaxScaler()
        self.df_cleaned["price"] = y_scaler.fit_transform(self.df_cleaned[["price"]]).flatten()
        
        return self.df_cleaned
    
    def tts(self):
        
        self.X = self.df_cleaned.drop(["price"], axis = 1)
        self.y = self.df_cleaned["price"]
                
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

        print(f"X_train: {self.X_train.shape} | y_train: {self.y_train.shape}")
        print(f"X_test: {self.X_test.shape} | y_test: {self.y_test.shape}")
    
    def train_model(self):
        
        models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(),
                  RandomForestRegressor(), SVR(), AdaBoostRegressor(), GradientBoostingRegressor()]
        
        metrics = list()
        
        for model in models:
            
            # fit
            
            model.fit(self.X_train, self.y_train)

            # predict
            
            self.yhat = model.predict(self.X_test)
            
            # metrics
            
            r2 = r2_score(self.y_test, self.yhat)
            mse = mean_squared_error(self.y_test, self.yhat)
        
            metrics.append([str(model), r2, mse, model])
            
        self.df_metrics = pd.DataFrame(data = metrics, columns = ["model_name", "r2", "mse", "model"])
        self.df_metrics.sort_values(by = "r2", ascending = False, inplace= True)
        
    def return_metrics(self):
        
        return self.df_metrics
    
    def display_metrics(self):
        
        display(self.df_metrics)
        
    def model_feature_importances(self, model):
        
        importances = np.argsort(model.feature_importances_)[::-1]
        d_importances = dict()
        
        for i in importances:

            d_importances[i] = [model.feature_importances_[i]*100, self.df_cleaned.drop("price", axis = 1).columns[i]]
            print(i, model.feature_importances_[i]*100, self.df_cleaned.drop("price", axis = 1).columns[i])
            
        return d_importances
    
    def grid_search_cv_tuning(self):
        
        model = RandomForestRegressor()
        
        params = {"n_estimators" : [i for i in range(100, 1001, 50)],
                  "max_depth"    : [8, 10, 12, 14, 16],
                  "max_features" : ["log2", "sqrt"]}

        scorers = {"r2", "neg_mean_squared_error"}

        grid_solver = GridSearchCV(estimator  = model, 
                                   param_grid = params, 
                                   scoring    = scorers,
                                   cv         = 10,
                                   refit      = "r2",
                                   n_jobs     = -1, 
                                   verbose    = 2)

        self.model_result = grid_solver.fit(self.X_train, self.y_train)
        
    def grid_search_cv_validation(self):
        
        l_validations = [self.model_result.best_estimator_,
                         self.model_result.cv_results_["mean_test_r2"].max(),
                         self.model_result.best_score_]
        self.df_validations = pd.DataFrame(data = l_validations, columns = ["Best Estimator",
                                                                            "Mean Test R**2",
                                                                            "Best Score"])
        return self.df_validations
    
    def save_model(self, name, ext, model):
    
        with open(f"{name}.{ext}", "wb") as file:
            pickle.dump(model)
            
    def load_model(self, name, ext):
        
        with open(f"{name}.{ext}", "rb") as file:
            self.model = pickle.load(file)
            
        