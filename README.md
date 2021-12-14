# Fynesse Template

This repo provides a python template repo for doing data analysis according to the Fynesse framework.

One challenge for data science and data science processes is that they do not always accommodate the real-time and evolving nature of data science advice as required, for example in pandemic response or in managing an international supply chain. The Fynesse paradigm is inspired by experience in operational data science both in the Amazon supply chain and in the UK Covid-19 pandemic response.

The Fynesse paradigm considers three aspects to data analysis, Access, Assess, Address. 

## Access

Gaining access to the data, including overcoming availability challenges (data is distributed across architectures, called from an obscure API, written in log books) as well as legal rights (for example intellectual property rights) and individual privacy rights (such as those provided by the GDPR).

It seems a great challenge to automate all the different aspects of the process of data access, but this challenge is underway already through the process of what is commonly called *digital transformation*. The process of digital transformation takes data away from physical log books and into digital devices. But that transformation process itself comes with challenges. 

Legal complications around data are still a major barrier though. In the EU and the US database schema and indices are subject to copyright law. Companies making data available often require license fees. As many data sources are combined, the composite effect of the different license agreements often makes the legal challenges insurmountable. This was a common challenge in the pandemic, where academics who were capable of dealing with complex data predictions were excluded from data access due to challenges around licensing. A nice counter example was the work led by Nuria Oliver in Spain who after a call to arms in a national newspaper  was able to bring the ecosystem together around mobility data.

However, even when organisation is fully digital, and license issues are overcome, there are issues around how the data is managed stored, accessed. The discoverability of the data and the recording of its provenance are too often neglected in the process of digtial transformation. Further, once an organisation has gone through digital transformation, they begin making predictions around the data. These predictions are data themselves, and their presence in the data ecosystem needs recording. Automating this portion requires structured thinking around our data ecosystems.

The access model creates the database connection, fetches data and stores it in the database tables, lists, or data frames. 
     - create_connection(database_details) creates the database connetcion given database_details that contains the url, port, and database name - this implementation hides the username and password in a local file by asking the user to introduce credentials when running the interact manual
     - upload_pp_database(conn), upload_postcode_data(conn), upload_prices_coordinates_data(conn) specify the database schema, download the data and upload it into the database tables
     - get_pois_features(latitude, longitude) returns a list of features, including the latitude, longitude, and number of POIS features of each tag type
     - table_head(connection, table_name, limit=5) loads the head of an SQL table - useful for previewing if table had been loaded successfully
     - cache_prices_coordinates_data(conn, city, district, property_type, date) will cache datapoints from the joined pp_data and postcode_data tables from a specified city & district, in order to save time when assessing and plotting the data by using select_cached(conn, city, district, property_type, date, date_range) to select directly from the cached data instead of performing joins each time

## Assess

Understanding what is in the data. Is it what it's purported to be, how are missing values encoded, what are the outliers, what does each variable represent and how is it encoded.

Data that is accessible can be imported (via APIs or database calls or reading a CSV) into the machine and work can be done understanding the nature of the data. The important thing to say about the assess aspect is that it only includes things you can do *without* the question in mind. This runs counter to many ideas about how we do data analytics. The history of statistics was that we think of the question *before* we collect data. But that was because data was expensive, and it needed to be excplicitly collected. The same mantra is true today of *surveillance data*. But the new challenge is around *happenstance data*, data that is cheaply available but may be of poor quality. The nature of the data needs to be understood before its integrated into analysis. Unfortunately, because the work is conflated with other aspects, decisions are sometimes made during assessment (for example approaches to imputing missing values) which may be useful in one context, but are useless in others. So the aim in *assess* is to only do work that is repeatable, and make that work available to others who may also want to use the data.

The assess module contains the assess_houses method checking columns needed for processing the data are not empty (NaN), view_pois_points method for understanding how the number of POIS points influence the house prices, and many plotting functions that helped in chosing the parameters for the prediction model. 
    - plot_price_histograms gives a general idea of how the data looks like and which distribution the price follows
    - plot_price_distance helps in chosing the maximum distance from the data point we want to predict the price for in the training set
    - plot_lat_long_price shows there is a strong dependency between the price and (latitude, longitude) values 
    - plot_property_type_boxplot and plot_house_types_distributions show, as in Simpson's paradox, the data has property_type as a confounding variable as, when the data is grouped based on the property_type value, it follows different distributions 
    - plot_price_in_time, plot_yearly_price, plot_monthly_price help in chosing the data_range for gathering the training data by looking at the price trends in time
    - view_prediction_accuracy, plot_test_bars is a way of visualizing the predictions when testing, compared to the actual values and measuring the variance score

## Address

The final aspect of the process is to *address* the question. We'll spend the least time on this aspect here, because it's the one that is most widely formally taught and the one that most researchers are familiar with. In statistics, this might involve some confirmatory data analysis. In machine learning it may involve designing a predictive model. In many domains it will involve figuring out how best to visualise the data to present it to those who need to make the decisions. That could involve a dashboard, a plot or even summarisation in an Excel spreadsheet.

The address module contains the make_prediction(conn, latitude, longitude, property_type, date) function with optional parameters that are initially set to date_range=180, data_distance=0.018, tags=TAGS, pois_radius=0.005, max_training_size=20 designed for experimenting with the parameters. This outputs the predicted house price for a point at a given latitude, longitude, house type and date by training a Poission regression model with data gathered in the date_range before the given date, from a radius of data_distance and taking into consideration the number of POIS points around pois_radius. Because of the time limitation, the training data is chosen as a random sample of a maximum size max_training_size from the eligible datapoints for training. 

The steps to do the prediction:
    - join housing data and postcode data to find geographical coordinates of houses (which are features for the model), but only on the houses that are in the desired bounding box and date range 
    - for each point in the training set find the number of pois points for each defined tag and form a matrix of features
    - train the model
    - get the features for the desired prediction point and predict the price

The test(conn, latitude, longitude, date, property_type) function does the same thing as the make_predict, but by also splitting the dataset into a training and validation ones in order to test the accuracy of the prediction model. The results can be plotted with the view_prediction_accuracy(conn, latitude, longitude, date, property_type) method from Assess, including the variance score.