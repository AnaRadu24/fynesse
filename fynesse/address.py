# This file contains code for suporting addressing questions in the data
"""Address a particular question that arises from the data"""

from . import access
from .access import *
from . import assess
from .assess import *

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # data split

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
    z = []
    for house in training_data.iterrows():
        pois_data = access.get_pois_features(float(house[1].latitude), float(house[1].longitude), tags=tags, box_radius=pois_radius)
        x.append(pois_data)
        y.append(house[1].price)
        z.append([house[1].date_of_transfer, house[1].new_build_flag, house[1].tenure_type, house[1].locality, house[1].town_city, house[1].district])
    df = pd.DataFrame(y, columns=['price'])
    df[["latitude", "longitude", "amenity", "leisure", "shop", "healthcare", "sport", "public_transport"]] = pd.DataFrame(x, columns=["latitude", "longitude", "amenity", "leisure", "shop", "healthcare", "sport", "public_transport"])
    df[["date_of_transfer", "new_build_flag", "tenure_type", "locality", "town_city", "district"]] = pd.DataFrame(z, columns=["date_of_transfer", "new_build_flag", "tenure_type", "locality", "town_city", "district"])
    #display(df)
    fitted_model = sm.GLM(y, x, family = sm.families.Poisson()).fit()
    return fitted_model

def predict(fitted_model, latitude, longitude, tags, pois_radius):
    x_pred = access.get_pois_features(latitude=latitude, longitude=longitude, tags=tags, box_radius=pois_radius)
    y_pred = fitted_model.get_prediction(x_pred).summary_frame(alpha=0.05)['mean'][0]
    return y_pred

def make_prediction(conn, latitude, longitude, property_type, date, date_range=180, data_distance=0.018, tags=TAGS, pois_radius=0.005, max_training_size=20):
    prices_coordinates_rows = access.join_price_coordinates_with_date_location(conn, latitude=latitude, longitude=longitude, date=date, 
                                                                        property_type=property_type, date_range=date_range, box_radius=data_distance)
    prices_coordinates_data = pd.DataFrame(prices_coordinates_rows, columns=["price", "date_of_transfer", "postcode", "property_type", "new_build_flag", "tenure_type", 
                                                                            "locality", "town_city", "district", "county", "country", "latitude", "longitude"])
    fitted_model = train(dataset=prices_coordinates_data, max_training_size=max_training_size, tags=tags, pois_radius=pois_radius)
    y_pred = predict(fitted_model=fitted_model, latitude=latitude, longitude=longitude, tags=tags, pois_radius=pois_radius)
    #assess.view_pois_points(latitude, longitude)
    print("Predicted price for a house of type " + property_type + "at latitude " + str(latitude) + " and longitude " + str(longitude) + " in " + date + 
        " based on the houses sold in the previous " + str(date_range) + " days on a radius of " + str(pois_radius*111) + "km is " + str(int(y_pred)))
    return int(y_pred)

def test(conn, latitude, longitude, date, property_type, date_range=180, data_distance=0.018, tags=TAGS, pois_radius=0.005, max_training_size=20):
    prices_coordinates_rows = access.join_price_coordinates_with_date_location(conn, latitude=latitude, longitude=longitude, date=date, 
                                                                        property_type=property_type, date_range=date_range, box_radius=data_distance)
    prices_coordinates_data = pd.DataFrame(prices_coordinates_rows, columns=["price", "date_of_transfer", "postcode", "property_type", "new_build_flag", "tenure_type", 
                                                                            "locality", "town_city", "district", "county", "country", "latitude", "longitude"])
    if prices_coordinates_data.shape[0] == 0:
        print(f'No data points found. Cannot make prediction.')
    if prices_coordinates_data.shape[0] < 10:
        print(f'Few data points warning: Model created from only {prices_coordinates_data.shape[0]} data points.')

    train_data, test_data = train_test_split(prices_coordinates_data, test_size=0.1, random_state=0)
    fitted_model = train(dataset=train_data, max_training_size=max_training_size, tags=tags, pois_radius=pois_radius)
    y_pred = []
    for pred in test_data.iterrows():
        y_pred.append(int(predict(fitted_model=fitted_model, latitude=float(pred[1].latitude), longitude=float(pred[1].longitude), tags=tags, pois_radius=pois_radius)))
    test_results = pd.DataFrame(test_data[['date_of_transfer', 'new_build_flag', 'tenure_type', 'property_type', 'town_city', 'district', 'county', 'longitude', 'latitude', 'price']], columns=['date_of_transfer', 'new_build_flag', 'tenure_type', 'property_type', 'town_city', 'district', 'county', 'longitude', 'latitude', 'price'])
    test_results['price_prediction'] = y_pred
    display(test_results)
    return test_results