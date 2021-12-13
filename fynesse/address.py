# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

from .access import *

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns # visualization
from sklearn.model_selection import train_test_split # data split
from sklearn.metrics import explained_variance_score as evs # evaluation metric

"""Address a particular question that arises from the data"""

TAGS = {"amenity": True, 
        "leisure": True, 
        "shop": True, 
        "healthcare": True, 
        "sport": True, 
        "public_transport": True}

def train(dataset, max_training_size, tags, pois_radius):
    training_data = dataset.sample(n = min(max_training_size, dataset.shape[0]), random_state=0)
    x = []
    y = []
    for house in training_data.iterrows():
        pois_data = get_pois_features(float(house[1].latitude), float(house[1].longitude), tags=TAGS, box_radius=pois_radius)
        x.append(pois_data)
        y.append(house[1].price)
    print(x)
    print(y)
    fitted_model = sm.GLM(y, x, family = sm.families.Poisson()).fit()
    return fitted_model

def predict(fitted_model, latitude, longitude, tags, pois_radius):
    x_pred = get_pois_features(latitude=latitude, longitude=longitude, tags=tags, box_radius=pois_radius)
    print(x_pred)
    y_pred = fitted_model.get_prediction(x_pred).summary_frame(alpha=0.05)['mean'][0]
    print(int(y_pred))
    return y_pred

def make_prediction(conn, latitude, longitude, property_type, date, date_range=180, data_distance=0.03, tags=TAGS, pois_radius=0.005, max_training_size=15):
    prices_coordinates_rows = join_price_coordinates_with_date_location(conn, latitude=latitude, longitude=longitude, date=date, 
                                                                        property_type=property_type, date_range=date_range, box_radius=data_distance)
    prices_coordinates_data = pd.DataFrame(prices_coordinates_rows, columns=["price", "date_of_transfer", "postcode", "property_type", "new_build_flag", "tenure_type", 
                                                                            "locality", "town_city", "district", "county", "country", "latitude", "longitude"])
    print(prices_coordinates_data)
    fitted_model = train(dataset=prices_coordinates_data, max_training_size=max_training_size, tags=tags, pois_radius=pois_radius)
    y_pred = predict(fitted_model=fitted_model, latitude=latitude, longitude=longitude, tags=tags, pois_radius=pois_radius)
    return int(y_pred)

def test(conn, latitude, longitude, date, property_type, date_range=180, data_distance=0.03, pois_radius=0.005, max_training_size=15):
    prices_coordinates_rows = join_price_coordinates_with_date_location(conn, latitude=latitude, longitude=longitude, date=date, 
                                                                        property_type=property_type, date_range=date_range, box_radius=data_distance)
    prices_coordinates_data = pd.DataFrame(prices_coordinates_rows, columns=["price", "date_of_transfer", "postcode", "property_type", "new_build_flag", "tenure_type", 
                                                                            "locality", "town_city", "district", "county", "country", "latitude", "longitude"])
    if prices_coordinates_data.shape[0] == 0:
        print(f'No data points found. Cannot make prediction.')
    if prices_coordinates_data.shape[0] < 10:
        print(f'Few data points warning: Model created from only {prices_coordinates_data.shape[0]} data points.')

    train_data, test_data = train_test_split(prices_coordinates_data, test_size=0.1, random_state=0)
    fitted_model = train(dataset=train_data, max_training_size=max_training_size, tags=TAGS, pois_radius=pois_radius)
    y_pred = []
    for pred in test_data.iterrows():
        y_pred.append(int(predict(fitted_model=fitted_model, latitude=float(pred[1].latitude), longitude=float(pred[1].longitude), tags=TAGS, pois_radius=pois_radius)))
    test_results = test_data
    test_results['price_prediction'] = y_pred
    return test_results
  
def eval(test_results):
    return evs(test_results.price_prediction, test_results.price)

def plot(test_results):
    plt.figure(figsize=(10,6))
    plt.title("Predicted Price vs Actual Price")
    test_results = test_results.reset_index()
    sns.barplot(x=test_results.index, y=test_results.price, alpha=0.9, edgecolor='red', color='pink', linewidth=2, label='actual price')
    sns.barplot(x=test_results.index, y=test_results.price_prediction, alpha=0.5, edgecolor='blue', color='darkblue', linewidth=3, label='predicted price')
    plt.xlabel("Prediction number")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
