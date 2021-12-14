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

def plot_property_type_boxplot(conn, city, district):
    cur = conn.cursor()
    cur.execute(f"""
                    SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, 
                    locality, town_city, district, county, country, latitude, longitude  
                    FROM prices_coordinates_data
                    WHERE town_city = '{city}' AND
                    district = '{district}'
                    """)

    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                                'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df['price'] = np.log(df['price'])
    grouped = df.groupby('property_type')
    grouped.boxplot(column='price', figsize=(8,10), subplots=False)
    plt.title('Log Price in ' + city + ", " + district + " for the different types of houses")
    plt.xlabel('house type')
    plt.ylabel('log price')
    plt.show()

def plot_price_histograms(conn, city, district, property_type, date, date_range):
    rows = access.select_cached(conn, city, district, property_type, date, date_range)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                            'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    plt.figure(figsize=(10,6))
    plt.hist(df.price, label='series1', alpha=.7, edgecolor='red', linewidth=2)
    plt.title('Price Histogram ' + city + ", " + district + " for houses of type " + property_type)
    plt.ylabel('Price')
    plt.ylabel('Frequency')
    plt.show()

def plot_lat_long_price(conn, city, district, property_type, date, date_range=3650):
    rows = access.select_cached(conn, city, district, property_type, date, date_range)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                        'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.7, s=df['price']/10000,
    figsize=(12, 8), c='price', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.title('Price Map in ' + city + ", " + district + " for houses of type " + property_type)
    plt.xlabel('latitude')
    plt.ylabel('longitude')
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
    rows = access.join_price_coordinates_with_date_location(conn, latitude, longitude, date, property_type, date_range, box_radius)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                                'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df.latitude = df.latitude.astype("float")
    df.longitude = df.longitude.astype("float")
    df['distance'] = haversine(longitude, latitude, df.longitude, df.latitude)
    plt.figure(figsize=(14,6))
    df.plot(kind='scatter', x='distance', y='price', c='distance', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.title('Price vs Distance from point with latitude ' + str(latitude) + " and longitude " + str(longitude) + " for houses of type " + property_type)
    plt.xlabel('distance')
    plt.ylabel('price')
    plt.show()

def plot_price_in_time(conn, latitude, longitude, date, property_type, date_range=7000, box_radius=0.01):
    rows = access.join_price_coordinates_with_date_location(conn, latitude, longitude, date, property_type, date_range=180, box_radius=0.04)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                                'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
    plt.figure(figsize=(14,6))
    sns.lineplot(data=df, x='date_of_transfer', y='price', color='blue')
    plt.title("Price variation in time for houses around latitude " + str(latitude) + " and longitude " + str(longitude) + " for houses of type " + property_type)
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()

def plot_yearly_price(conn, city, district, property_type, date, date_range=365):
    rows = access.select_cached(conn, city, district, property_type, date, date_range)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                        'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df['price'] = np.log(df['price'])
    df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
    df['year'] = df['date_of_transfer'].dt.year
    grouped = pd.DataFrame(df.groupby('year')['price'].mean())
    grouped.reset_index(inplace=True)
    plt.figure(figsize=(14,6))
    df.plot(kind='scatter', x='year', y='price', alpha=0.7, color = 'purple', label='prices in that year')
    sns.lineplot(data=grouped, x='year', y='price', color='blue', label='average yearly price')
    plt.title("Yearly price variation in " + city + ", " + district + " for houses of type " + property_type)
    plt.xlabel('price')
    plt.ylabel('year')
    plt.legend()
    plt.show()

def plot_monthly_price(conn, city, district, property_type, date, date_range=3650):
  rows = access.select_cached(conn, city, district, property_type, date, date_range)
  df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                    'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
  df['price'] = np.log(df['price'])
  df.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
  df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
  df['month'] = df['date_of_transfer'].dt.month

  grouped = pd.DataFrame(df.groupby('month')['price'].mean())
  grouped.reset_index(inplace=True)
  plt.figure(figsize=(14,6))
  df.plot(kind='scatter', x='month', y='price', alpha=0.7, color = 'purple', label='prices in that month')
  plt.title("Monthly price variation in " + str(date.dt.year) + ", " + city + ", " + district + " for houses of type " + property_type)
  plt.xlabel('price')
  plt.ylabel('month')
  sns.lineplot(data=grouped, x='month', y='price', color='blue', label='average monthly price')
  plt.title("Monthly price variation in time for houses in " + city + ", " + district + " for houses of type " + property_type)
  plt.legend()
  plt.show()

def plot_corr(conn, city, district, property_type, date, date_range):
    rows = access.select_cached(conn, city, district, property_type, date, date_range)
    data = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                        'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    data = data.sample(n = 40, random_state=0)
    data.to_csv('selected_prices_coordinates_data.csv', header=False, index=False)
    x = []
    y = []
    z = []
    for house in data.iterrows():
        pois_data = access.get_pois_features(float(house[1].latitude), float(house[1].longitude))
        x.append(pois_data)
        y.append(house[1].price)
        z.append([house[1].date_of_transfer, house[1].new_build_flag, house[1].tenure_type, house[1].locality, house[1].town_city, house[1].district])
    df = pd.DataFrame(y, columns=['price'])
    df[["latitude", "longitude", "amenity", "leisure", "shop", "healthcare", "sport", "public_transport"]] = pd.DataFrame(x, columns=["latitude", "longitude", "amenity", "leisure", "shop", "healthcare", "sport", "public_transport"])
    df[["date_of_transfer", "new_build_flag", "tenure_type", "locality", "town_city", "district"]] = pd.DataFrame(z, columns=["date_of_transfer", "new_build_flag", "tenure_type", "locality", "town_city", "district"])
    
    for tag in address.TAGS: 
        correlation = df['price'].corr(df[tag])
        plt.figure(figsize=(25,15))
        plt.scatter(df[tag],df['price'], label=tag)
        slope, intercept, r, p, stderr = scipy.stats.linregress(df[tag], df['price'])
        line = f'Regression line: y={intercept:.1f}+{slope:.1f}x, r={r:.1f}'
        plt.plot(df[tag], intercept + slope * df[tag], label=line, color = 'black')
        print("Correlation for tag " + tag + ": " + str(correlation))
    plt.title("Price vs no. of pois points in " + city + ", " + district + " for houses of type " + property_type)
    plt.xlabel('number of pois points')
    plt.ylabel('price')
    plt.legend()
    plt.show()

def plot_test_bars(test_results):
    plt.figure(figsize=(10,6))
    plt.title("Predicted Price vs Actual Price for " + test_results.town_city[0] + ", " + test_results.district[0] + " - Variance Score: " + str(evs(test_results.price_prediction, test_results.price)))
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

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()


def query(data):
    """Request user input for some aspect of the data."""
    

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
