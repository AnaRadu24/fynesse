from .config import *
from .address import *
from .assess import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

import os
import urllib.request
import pandas as pd
import pymysql
import yaml
from ipywidgets import interact_manual, Text, Password
import urllib.request
import datetime
import osmnx as ox
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # visualization
import zipfile
from sklearn.model_selection import train_test_split # data split
from sklearn.metrics import explained_variance_score as evs # evaluation metric


# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. 
Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side 
also think about the ethical issues around this data. """

# Write code for requesting and storing credentials (username, password) here. 
@interact_manual(username=Text(description="Username:"), 
                  password=Password(description="Password:"))
def write_credentials(username, password):
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username, 
                            'password': password}
        yaml.dump(credentials_dict, file)

# to protect the passoword, create a credentials.yaml file locally that will store the username and password so that 
# the client can access the server without ever showing your password in the notebook.

def create_connection(database_details):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    with open("credentials.yaml") as file:
      credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    database_name = database_details["database"]
    url = database_details["url"]
    port = database_details["port"]
    conn = None
    try:
      # when connecting to the database we used the flag local_infile=1 to ensure we could load local files into the database
      conn = pymysql.connect(user=username, passwd=password, host=url, port=port, local_infile=1, db=database_name)
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return initialise_db(conn, database_name)

def initialise_db(conn, database_name):
    with conn.cursor() as cur:
        cur.execute('SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";')
        cur.execute('SET time_zone = "+00:00";')

        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}` DEFAULT \
        CHARACTER SET utf8 COLLATE utf8_bin;")
        cur.execute(f"USE `{database_name}`;")

    conn.commit()
    return conn

def price_paid_schema(conn):
    cur = conn.cursor()
    # The schema tells the database server what to expect in the columns of the table.
    cur.execute("""DROP TABLE IF EXISTS `pp_data`;""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `pp_data` (
            `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
            `price` int(10) unsigned NOT NULL,
            `date_of_transfer` date NOT NULL,
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
            `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
            `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
            `street` tinytext COLLATE utf8_bin NOT NULL,
            `locality` tinytext COLLATE utf8_bin NOT NULL,
            `town_city` tinytext COLLATE utf8_bin NOT NULL,
            `district` tinytext COLLATE utf8_bin NOT NULL,
            `county` tinytext COLLATE utf8_bin NOT NULL,
            `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
            `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;
    """.replace("\n", " "))
    cur.execute("""ALTER TABLE `pp_data` ADD PRIMARY KEY (`db_id`);""")
    cur.execute(""" ALTER TABLE `pp_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;""")
    cur.execute("""CREATE INDEX `pp.postcode` USING HASH ON `pp_data` (postcode);""")
    cur.execute("""CREATE INDEX `pp.date` USING HASH ON `pp_data` (date_of_transfer);""")
    conn.commit()

PP_COLUMNS = ['transaction_unique_identifier', 'price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
            'primary_addressable_object_name', 'secondary_addressable_object_name', 'street', 'locality', 'town_city', 'district', 'county',
            'record_status', 'ppd_category_type', 'db_id']

def postcode_schema(conn):
    cur = conn.cursor()
    cur.execute("""DROP TABLE IF EXISTS `postcode_data`;""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `postcode_data` (
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `status` enum('live','terminated') NOT NULL,  
            `usertype` enum('small', 'large') NOT NULL,
            `easting` int unsigned,
            `northing` int unsigned,
            `positional_quality_indicator` int NOT NULL,
            `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
            `latitude` decimal(11,8) NOT NULL,
            `longitude` decimal(10,8) NOT NULL,
            `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
            `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
            `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
            `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
            `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
            `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
            `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
            `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin; 
    """.replace("\n", " "))
    #cur.execute("""ALTER TABLE `postcode_data` ADD PRIMARY KEY (`db_id`);""")
    #conn.commit()
    #cur.execute("""ALTER TABLE `postcode_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;
    #    CREATE INDEX `po.postcode` USING HASH
    #      ON `postcode_data`
    #        (postcode);
    #    """.replace("\n", " "))
    cur.execute("""CREATE INDEX `po.postcode` USING HASH ON `postcode_data` (postcode);""")
    cur.execute("""
        LOAD DATA LOCAL INFILE 'postcode_data_folder/open_postcode_geo.csv' INTO TABLE `postcode_data`
        FIELDS TERMINATED BY ',' 
        LINES STARTING BY '' TERMINATED BY '\n';
    """.replace("\n", " "))
    conn.commit()

POSTCODE_COLUMNS = ['postcode', 'status', 'usertype', 'easting', 'northing', 'positional_quality_indicator', 'country', 'latitude', 'longitude',
            'postcode_no_space', 'postcode_fixed_width_seven', 'postcode_fixed_width_eight', 'postcode_area', 'postcode_district', 'postcode_sector',
            'outcode', 'incode', 'db_id']

def prices_coordinates_schema(conn):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS `prices_coordinates_data`;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
            `price` int(10) unsigned NOT NULL,
            `date_of_transfer` date NOT NULL,
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
            `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `locality` tinytext COLLATE utf8_bin NOT NULL,
            `town_city` tinytext COLLATE utf8_bin NOT NULL,
            `district` tinytext COLLATE utf8_bin NOT NULL,
            `county` tinytext COLLATE utf8_bin NOT NULL,
            `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
            `latitude` decimal(11,8) NOT NULL,
            `longitude` decimal(10,8) NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY
            ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;
        """.replace("\n", " "))
    #cur.execute("""ALTER TABLE `prices_coordinates_data` ADD PRIMARY KEY (`db_id`);""")
    cur.execute("""
        ALTER TABLE `prices_coordinates_data`
        MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;
    """.replace("\n", " "))
    conn.commit()

PRICES_COORDINATES_COLUMNS = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
            'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude', 'db_id']

def load_data(conn, csv_name, table_name):
    conn.cursor().execute(f"""
        LOAD DATA LOCAL INFILE '{csv_name}' INTO TABLE {table_name}
        FIELDS TERMINATED BY ','
        OPTIONALLY ENCLOSED BY '"'
        LINES STARTING BY '' TERMINATED BY '\n';""")
    conn.commit()

def upload_pp_database(conn):
    pp_data_url = 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/'
    price_paid_schema(conn)
    for year in range(1995, 2022):
        filename = 'pp-' + str(year) + '.csv'
        urllib.request.urlretrieve(pp_data_url + filename, 'pp-data.csv')
        load_data(conn, 'pp_data.csv', 'pp_data')
        print("Uploaded dataset from " + str(year))
    return 

def upload_postcode_data(conn):
    postcode_data_url = 'https://www.getthedata.com/downloads/open_postcode_geo.csv.zip'
    urllib.request.urlretrieve(postcode_data_url, 'postcode_data.zip')
    with zipfile.ZipFile('postcode_data.zip', 'r') as zip_ref:
        zip_ref.extractall('postcode_data_folder')
    print("data downloaded")
    load_data(conn, 'postcode_data_folder/open_postcode_geo.csv', 'postcode_data')
    print("Uploaded postcode dataset ")

def select_top(conn, table,  n):
    """
    Query n first rows of the table
    :param conn: the Connection object
    :param table: The table to query
    :param n: Number of rows to query
    """
    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM {table} LIMIT {n}')
    rows = cur.fetchall()
    return rows

def print_rows(rows):
  for r in rows:
    print(r)

def execute_query(conn, query):
  with conn.cursor() as cur:
        cur.execute(query)
  rows = cur.fetchall()
  return rows

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    database_details = {"url": 'database-amr97.cgrre17yxw11.eu-west-2.rds.amazonaws.com', 
                    "port": 3306, "database": 'property_prices'}
    conn = create_connection(database_details=database_details)
    upload_pp_database(conn)
    upload_postcode_data(conn)
    house_prices = pd.DataFrame(execute_query(conn, 'SELECT * FROM pp_data'), columns=PP_COLUMNS)
    poscode_data = pd.DataFrame(execute_query(conn, 'SELECT * FROM postcode_data'), columns=POSTCODE_COLUMNS)
    return pd.merge(house_prices, poscode_data, on = 'postcode', how = 'inner')  # the big merge

def join_price_coordinates_with_date_location(conn, latitude, longitude, date, property_type, date_range=180, box_radius=0.04):

    d1 = datetime.datetime.strptime(date, "%Y-%m-%d")
    d2 = d1
    d1 = d1 - datetime.timedelta(days=date_range)
    d1 = d1.strftime("%Y-%m-%d")
    d2 = d2.strftime("%Y-%m-%d")

    lat1 = latitude - box_radius
    lat2 = latitude + box_radius
    lon1 = longitude - box_radius
    lon2 = longitude + box_radius

    cur = conn.cursor()
    cur.execute(f"""
                SELECT price, date_of_transfer, pp_data.postcode as postcode, property_type, new_build_flag, tenure_type, 
                locality, town_city, district, county, country, latitude, longitude 
                FROM pp_data
                INNER JOIN postcode_data
                ON pp_data.postcode = postcode_data.postcode
                WHERE date_of_transfer BETWEEN '{d1}' AND '{d2}' AND
                property_type = '{property_type}' AND
                latitude BETWEEN {lat1} AND {lat2} AND
                longitude BETWEEN {lon1} AND {lon2}
                """)

    rows = cur.fetchall()
    return rows

def select_town_city(conn, city, district, property_type, date, date_range, limit):
    
    d1 = datetime.datetime.strptime(date, "%Y-%m-%d")
    d2 = d1
    d1 = d1 - datetime.timedelta(days=date_range)
    d1 = d1.strftime("%Y-%m-%d")
    d2 = d2.strftime("%Y-%m-%d")

    cur = conn.cursor()
    cur.execute(f"""
                SELECT price, date_of_transfer, pp_data.postcode as postcode, property_type, new_build_flag, tenure_type, 
                locality, town_city, district, county, country, latitude, longitude  
                FROM pp_data
                INNER JOIN postcode_data
                ON pp_data.postcode = postcode_data.postcode
                WHERE town_city = '{city}' AND
                district = '{district}' AND
                property_type = '{property_type}' AND
                date_of_transfer BETWEEN '{d1}' AND '{d2}'
                ORDER BY RAND ( )
                LIMIT {limit}
                """)
    rows = cur.fetchall()
    return rows

def select_cached(conn, city, district, property_type, date, date_range):
    
    d1 = datetime.datetime.strptime(date, "%Y-%m-%d")
    d2 = d1
    d1 = d1 - datetime.timedelta(days=date_range)
    d1 = d1.strftime("%Y-%m-%d")
    d2 = d2.strftime("%Y-%m-%d")

    cur = conn.cursor()
    cur.execute(f"""
                SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, 
                locality, town_city, district, county, country, latitude, longitude  
                FROM prices_coordinates_data
                WHERE town_city = '{city}' AND
                district = '{district}' AND
                property_type = '{property_type}' AND
                date_of_transfer BETWEEN '{d1}' AND '{d2}'
                """)

    rows = cur.fetchall()
    return rows

def cache_prices_coordinates_data(conn, city, district, property_type, date, date_range, limit):
    rows = select_town_city(conn, city, district, property_type, date, date_range, limit)
    df = pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                    'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude'])
    df.to_csv('cached_prices_coordinates_data.csv', header=False, index=False)
    access.load_data(conn, 'cached_prices_coordinates_data.csv', 'prices_coordinates_data')

def upload_prices_coordinates_data(conn, date_range=3650, limit=1000000):
    prices_coordinates_schema(conn)
    cache_prices_coordinates_data(conn, 'LONDON', 'WALTHAM FOREST', 'S', '2020-06-30', date_range, limit)
    cache_prices_coordinates_data(conn, 'LONDON', 'WALTHAM FOREST', 'D', '2020-06-30', date_range, limit)
    cache_prices_coordinates_data(conn, 'LONDON', 'WALTHAM FOREST', 'T', '2020-06-30', date_range, limit)
    cache_prices_coordinates_data(conn, 'LONDON', 'WALTHAM FOREST', 'O', '2020-06-30', date_range, limit)
    cache_prices_coordinates_data(conn, 'LONDON', 'WALTHAM FOREST', 'F', '2020-06-30', date_range, limit)
    cache_prices_coordinates_data(conn, 'CAMBRIDGE', 'CAMBRIDGE', 'D', '2020-06-30', date_range, limit)
    cache_prices_coordinates_data(conn, 'CAMBRIDGE', 'CAMBRIDGE', 'S', '2020-06-30', date_range=3650, limit=1000000)
    cache_prices_coordinates_data(conn, 'CAMBRIDGE', 'CAMBRIDGE', 'T', '2020-06-30', date_range=3650, limit=1000000)
    cache_prices_coordinates_data(conn, 'CAMBRIDGE', 'CAMBRIDGE', 'O', '2020-06-30', date_range=3650, limit=1000000)
    cache_prices_coordinates_data(conn, 'CAMBRIDGE', 'CAMBRIDGE', 'F', '2020-06-30', date_range=3650, limit=1000000)

def select_cached(conn, city, district, property_type, date, date_range):
    
    d1 = datetime.datetime.strptime(date, "%Y-%m-%d")
    d2 = d1
    d1 = d1 - datetime.timedelta(days=date_range)
    d1 = d1.strftime("%Y-%m-%d")
    d2 = d2.strftime("%Y-%m-%d")

    cur = conn.cursor()
    cur.execute(f"""
                SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, 
                locality, town_city, district, county, country, latitude, longitude  
                FROM prices_coordinates_data
                WHERE town_city = '{city}' AND
                district = '{district}' AND
                property_type = '{property_type}' AND
                date_of_transfer BETWEEN '{d1}' AND '{d2}'
                """)

    rows = cur.fetchall()
    return rows

# 0.02 degrees wide approx 2.2km, 1 degree is around 111km
def get_pois_features(latitude, longitude, tags=TAGS, box_radius=0.005):
    north = latitude + box_radius
    south = latitude - box_radius
    west = longitude - box_radius
    east = longitude + box_radius
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    count_map = pois.count()
    count_list = []
    count_list.append(latitude)
    count_list.append(longitude)
    for tag in tags:
        if tag in count_map:
            count_list.append(float(min(15, count_map[tag])))
        else:
            count_list.append(float(0))
    return count_list