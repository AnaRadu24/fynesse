from .config import *

from . import access
from .access import *
from . import address
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
from math import radians, cos, sin, asin, sqrt

TAGS = {"amenity": True, 
        "leisure": True, 
        "shop": True, 
        "healthcare": True, 
        "sport": True, 
        "public_transport": True}

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

def view_pois_points(latitude, longitude, tags=TAGS, box_radius=0.005):
    pois_list = access.get_pois_features(latitude, longitude, tags, box_radius)
    res = pois_list
    pois_list.reverse()
    print("Around point with latitude " + str(pois_list.pop()) + " and logitutde " + str(pois_list.pop()) + " on a radius of " +  str(box_radius*111) + "km there are a number of pois points of type: ")
    for tag in tags:
        print(tag + ": " + str(pois_list.pop()))

def plot_price_histograms(conn, latitude, longitude, date, property_type, date_range=180, box_radius=0.04):
    access.upload_prices_coordinates_data(conn, latitude, longitude, date, property_type, date_range, box_radius)
    df = pd.read_csv('prices_coordinates_data.csv', names=access.PRICES_COORDINATES_COLUMNS)
    df = df[['price', 'latitude', 'longitude', 'date_of_transfer', 'town_city',	'new_build_flag',	'tenure_type', 'property_type', 'town_city',	'district',	'county']]
    plt.hist(df.price, label='series1', alpha=.8, edgecolor='red')
    plt.show()

def plot_lat_long_price(conn, latitude, longitude, date, property_type, date_range=180, box_radius=0.04):
    rows = access.select_cached(conn, city, district, property_type, date, date_range)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                        'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.7, s=df['price']/10000,
    figsize=(12, 8), c='price', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    plt.show()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lons = np.array([radians(lon) for lon in lon2])
    lats = np.array([radians(lat) for lat in lat2])

    # haversine formula 
    dlon = np.subtract(lons, lon1) 
    dlat = np.subtract(lats, lat1) 
    a = np.square(np.sin(np.divide(dlat, 2)))
    b = np.multiply(np.square(np.sin(np.divide(dlon, 2))), np.cos(lons))
    b = np.multiply(b, cos(lat1))
    a = np.add(a, b)
    c = np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    return np.multiply(c, 2*6371)

def plot_price_distance(conn, latitude, longitude, date, property_type, date_range=180, box_radius=0.04):
    #access.upload_prices_coordinates_data(conn, latitude, longitude, date, property_type, date_range, box_radius)
    df = pd.read_csv('prices_coordinates_data.csv', names=access.PRICES_COORDINATES_COLUMNS)
    df.latitude = df.latitude.astype("float")
    df.longitude = df.longitude.astype("float")
    df['distance'] = haversine(longitude, latitude, df.longitude, df.latitude)
    df.plot(kind='scatter', x='distance', y='price')
    plt.legend()
    plt.show()

def plot_price_in_time(conn, latitude, longitude, date, property_type, date_range=7000, box_radius=0.01):
    #access.upload_prices_coordinates_data(conn, latitude, longitude, date, property_type, date_range, box_radius)
    df = pd.read_csv('prices_coordinates_data.csv', names=access.PRICES_COORDINATES_COLUMNS)
    df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
    plt.figure(figsize=(14,6))
    #df.plot(kind='scatter', x='date_of_transfer', y='price', alpha=0.7)
    sns.lineplot(data=df, x='date_of_transfer', y='price', color='blue')
    plt.title("Price variation in time for houses in Waltham Forest")
    plt.legend()
    plt.show()

def plot_test_bars(test_results):
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
    test_results = address.test(conn, latitude, longitude, date, property_type, date_range, data_distance, tags, pois_radius, max_training_size)
    plot_test_bars(test_results)
    return evs(test_results.price_prediction, test_results.price)

def plot_monthly_price(conn, city, district, property_type, date, date_range=3650):
    rows = access.select_cached(conn, city, district, property_type, date, date_range)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                        'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
    df['month'] = df['date_of_transfer'].dt.month
    df['year'] = df['date_of_transfer'].dt.year
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df_2020 = df.loc[(df["year"] == "2020")]
    df_2019 = df.loc[(df["year"] == "2019")]
    
    grouped_year = pd.DataFrame(df.groupby('year')['price'].mean())
    grouped_year.reset_index(inplace=True)
    grouped_month_2020 = pd.DataFrame(df_2019.groupby('month')['price'].mean())
    grouped_month_2020.reset_index(inplace=True)
    grouped_month_2019 = pd.DataFrame(df_2019.groupby('month')['price'].mean())
    grouped_month_2019.reset_index(inplace=True)

    plt.subplot(221)
    df.plot(kind='scatter', x='year', y='price', alpha=0.7)
    sns.lineplot(data=grouped_year, x='year', y='price', color='blue')
    plt.title("Price variation in time for houses in " + city)
    plt.legend()
    plt.show()

    plt.subplot(222)
    df_2020.plot(kind='scatter', x='month', y='price', alpha=0.7)
    sns.lineplot(data=grouped_month_2020, x='month', y='price', color='blue')
    plt.title("Price variation in time for houses in Waltham Forest")
    plt.legend()
    plt.show()

    plt.subplot(223)
    df_2019.plot(kind='scatter', x='month', y='price', alpha=0.7)
    sns.lineplot(data=grouped_month_2019, x='month', y='price', color='blue')
    plt.title("Price variation in time for houses in Waltham Forest")
    plt.legend()
    plt.show()

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()


def query(data):
    """Request user input for some aspect of the data."""
    

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
