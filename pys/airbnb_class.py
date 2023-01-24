import pandas as pd
import numpy as np
import re
import json
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pprint import pprint
import folium
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from IPython.display import display
import joblib

# Modelos
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Métricas
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import pickle


class airbnb:
    
    def __init__(self, data, city_names = None, file = "csv"):
                    
        if (file == "csv") and (city_names is not None):
            
            self.l_dfs = list()
            
            for enum, dataset in enumerate(data):
                
                self.l_dfs.append(pd.read_csv(dataset))
                
                self.l_dfs[enum].drop("source", axis = 1, inplace = True)
                
                self.l_dfs[enum]["city"] = city_names[enum].lower()
        
            self.df = pd.concat(self.l_dfs)
            
            print("Instance created!")
            
        elif file == "dataframe":
            
            self.l_dfs = list()

            for enum, dataframe in enumerate(data):
                
                self.l_dfs.append(dataframe)
                                        
            self.df = pd.concat(self.l_dfs)
            
            print("Instance created!")
            
        else:
            
            print("Only csv or dataframe are valid inputs, and city_names cannot be empty")
            
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
                       'room_type', 'accommodates', 'availability_365',
                       'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
                       'minimum_nights', 'maximum_nights',
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
                
        # Replace bathrooms_text with floats
        
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
    
        # Room type
        
        self.df_cleaned = self.df_cleaned[self.df_cleaned["room_type"] != "Hotel room"]
        self.df_cleaned = pd.concat([self.df_cleaned, pd.get_dummies(data = self.df_cleaned["room_type"])], axis = 1).drop("room_type", axis = 1)
        
        self.df_cleaned.dropna(inplace = True)
        
    def return_cleaned(self):
        
        return self.df_cleaned
    
    def display_cleaned(self):
        
        display(self.df_cleaned)
    
    def remove_outliers(self, accommodates = 8, bathrooms_min = 1, bathrooms_max = 2, bedrooms = 4, beds_min = 1, beds_max = 5, minimum_nights = 30,
                       maximum_nights = 70000, nreviews = 375, reviews_pmonth = 9, price = 350, htlc = 50000):

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
    
    def display_outliers(self):
        
        for i in self.df_cleaned.columns:
    
            print(i)
            sns.kdeplot(self.df_cleaned[i])
            plt.show()

    def label_encoding(self, df = None):
        
        if df is None:
            df = self.df_cleaned
            
        self.city_encoder = LabelEncoder()
        df["city"] = self.city_encoder.fit_transform(df["city"])
        self.neighbourhood_encoder = LabelEncoder()
        df["neighbourhood_cleansed"] = self.neighbourhood_encoder.fit_transform(df["neighbourhood_cleansed"])
        
        return df
    
    def normalize(self, df = None, price = None):
        
        if df is None:
            df = self.df_cleaned
            
        if price is None:

            self.x_scaler = MinMaxScaler()
            df[df.drop("price", axis = 1).columns] = self.x_scaler.fit_transform(df[df.drop("price", axis = 1).columns])

            self.y_scaler = MinMaxScaler()
            df["price"] = self.y_scaler.fit_transform(df[["price"]]).flatten()

        else:

            self.x_scaler = MinMaxScaler()
            df = self.x_scaler.fit_transform(df)

        
        return df

    # def lb_enc_norm_prediction(self, df_prediction):

    #     df_prediction["city"] = self.city_encoder.fit_transform(df_prediction["city"])
    #     df_prediction["neighbourhood_cleansed"] = self.neighbourhood_encoder.fit_transform(df_prediction["neighbourhood_cleansed"])
    #     df_prediction = self.x_scaler.fit_transform(df_prediction)

    #     return df_prediction
  
    
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
        
        d_validations = {"Best Estimator" : self.model_result.best_estimator_,
                         "Mean Test R**2" : self.model_result.cv_results_["mean_test_r2"].max(),
                         "Best Score"     : self.model_result.best_score_}
        
        self.df_validations = pd.DataFrame(data    = d_validations.items(), 
                                           columns = ["Validation","Result"])
        
    def return_model_result_gcv(self):
        
        return self.model_result
        
    def return_validations_gcv(self):
        
        return self.df_validations

    
    def final_trial_model(self, md = 16, mf = 'sqrt', ne = 800, rs = 42):
        
        '''It trains the best model with the features recomended'''
        
        model = RandomForestRegressor(max_depth=md, max_features=mf, n_estimators=ne, random_state=rs)
        model.fit(self.X_train, self.y_train)
        
        self.yhat = model.predict(self.X_test)
    
        return f"r**2 = {r2_score(self.y_test, self.yhat)}"
    
    def train_final_model(self, md, mf, ne, rs):
        
        '''Returns the definitive model'''
        
        self.X_def = self.df_cleaned.drop(["price"], axis = 1)
        self.y_def = self.df_cleaned["price"]
        
        model = RandomForestRegressor(max_depth = md, max_features = mf, n_estimators = ne, random_state = rs)

        model.fit(self.X_def, self.y_def)
        
        return  model

    def norm_enc_prediction(self, enc, xs, dataframe, cols):

        dataframe[cols[0]] = enc[0].transform(dataframe[cols[0]])
        dataframe[cols[1]] = enc[1].transform(dataframe[cols[1]])

        dataframe = xs.transform(dataframe)

        return dataframe

    
    def predict_price(self, array, model, y_s):  
        
        '''Predicts the price given a cleaned array with te features needed'''
        
        self.price_predicted = y_s.inverse_transform([model.predict(array)])
    
    def return_prediction(self):
        
        return self.price_predicted
    
    def save_model_pickle(self, name, ext, pkl_model):
    
        with open(f"{name}.{ext}", "wb") as file:
            pickle.dump(pkl_model, file, -1)
            
    def load_model_pickle(self, name, ext):
        
        with open(f"{name}.{ext}", "rb") as file:
            self.pkl_model = pickle.load(file)
            
        return self.pkl_model

    def save_model_joblib(self, name, jlib_model):

        joblib.dump(jlib_model, name +'.gz', compress= ('gzip', 3))

    def load_model_joblib(self, name):

        self.jlib_model = joblib.load(name + '.gz')

        return self.jlib_model


   