from .config import *

from .access import *
from .address import *

"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

import os
import numpy as np
import pandas as pd
import osmnx as ox
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns # visualization
from sklearn.model_selection import train_test_split # data split
from sklearn.metrics import explained_variance_score as evs # evaluation metric

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def assess_house_prices(region):
    print(f'Assessing dataframe...')
    required_columns = ['price', 'date_of_transfer', 'property_type', 'latitude', 'longitude', 'postcode']
    try: 
        for col in required_columns:
            assert not region[col].isnull().any()

        assert pd.to_numeric(region['price'], errors='coerce').notnull().all()

        prop_types = ['D', 'S', 'T', 'F', 'O']
        for uniq in region['property_type'].unique():
            assert uniq in prop_types
        
    except Exception as e:
        raise e
    print('Assessment is finished.')
    return region

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

def view_prediction_accuracy(conn, latitude, longitude, date, property_type, date_range=180, data_distance=0.03, tags=TAGS, pois_radius=0.005, max_training_size=15):
    test_results = test(conn, latitude, longitude, date, property_type, date_range, data_distance, tags, pois_radius, max_training_size)
    plot(test_results)
    return evs(test_results.price_prediction, test_results.price)

def view_pois_points(latitude, longitude, tags=TAGS, box_radius=0.005):
    pois_list = get_pois_features(latitude, longitude, tags, box_radius)
    pois_list.reverse()
    print("Around point with latitude " + str(pois_list.pop()) + " and logitutde " + str(pois_list.pop()) + " on a radius of " +  box_radius*111 + "km there are a number of pois points of type: ")
    for tag in tags:
        print(tag + ": " + str(pois_list.pop()))