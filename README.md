# python-api-challenge
#For both of these I used https://openweathermap.org/appid as a resource on loading them
Weather PY
# WeatherPy

---

## Starter Code to Generate Random Geographic Coordinates and a List of Cities

!pip install citipy

# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress

# Impor the OpenWeatherMap API key
from api_keys import weather_api_key

# Import citipy to determine the cities based on latitude and longitude
from citipy import citipy

### Generate the Cities List by Using the `citipy` Library

# Empty list for holding the latitude and longitude combinations
lat_lngs = []

# Empty list for holding the cities names
cities = []

# Range of latitudes and longitudes
lat_range = (-90, 90)
lng_range = (-180, 180)

# Create a set of random lat and lng combinations
lats = np.random.uniform(lat_range[0], lat_range[1], size=1500)
lngs = np.random.uniform(lng_range[0], lng_range[1], size=1500)
lat_lngs = zip(lats, lngs)

# Identify nearest city for each lat, lng combination
for lat_lng in lat_lngs:
    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    
    # If the city is unique, then add it to a our cities list
    if city not in cities:
        cities.append(city)

# Print the city count to confirm sufficient count
print(f"Number of cities in the list: {len(cities)}")

---

## Requirement 1: Create Plots to Showcase the Relationship Between Weather Variables and Latitude

### Use the OpenWeatherMap API to retrieve weather data from the cities list generated in the started code

# Set the API base URL
#Copied the URL from the Weather_Stats assignment
url = "http://api.openweathermap.org/data/2.5/weather?"

# Define an empty list to fetch the weather data for each city
city_data = []

# Print to logger
print("Beginning Data Retrieval     ")
print("-----------------------------")

# Create counters
record_count = 1
set_count = 1

# Loop through all the cities in our list to fetch weather data
for i, city in enumerate(cities):
        
    # Group cities in sets of 50 for logging purposes
    if (i % 50 == 0 and i >= 50):
        set_count += 1
        record_count = 0

    # Create endpoint URL with each city
    city_url = f"{url}appid={weather_api_key}&q={city}"
    
    # Log the url, record, and set numbers
    print("Processing Record %s of Set %s | %s" % (record_count, set_count, city))

    # Add 1 to the record count
    record_count += 1

    # Run an API request for each of the cities
    try:
        # Parse the JSON and retrieve data
        city_weather = requests.get(city_url).json()

        # Parse out latitude, longitude, max temp, humidity, cloudiness, wind speed, country, and date
        city_lat = city_weather["coord"]["lat"]
        city_lng = city_weather["coord"]["lon"]
        city_max_temp = city_weather["main"]["temp_max"]
        city_humidity = city_weather["main"]["humidity"]
        city_clouds = city_weather["clouds"]["all"]
        city_wind = city_weather["wind"]["speed"]
        city_country = city_weather["sys"]["country"]
        city_date = city_weather["dt"]

        # Append the City information into city_data list
        city_data.append({"City": city, 
                          "Lat": city_lat, 
                          "Lng": city_lng, 
                          "Max Temp": city_max_temp,
                          "Humidity": city_humidity,
                          "Cloudiness": city_clouds,
                          "Wind Speed": city_wind,
                          "Country": city_country,
                          "Date": city_date})

    # If an error is experienced, skip the city
    except:
        print("City not found. Skipping...")
        pass
              
# Indicate that Data Loading is complete 
print("-----------------------------")
print("Data Retrieval Complete      ")
print("-----------------------------")

# Convert the cities weather data into a Pandas DataFrame
city_data_df = pd.DataFrame(city_data)

# Show Record Count
city_data_df.count()

# Display sample data
city_data_df.head()

import os

# Create the output_data directory if it doesn't exist
output_data_dir = "output_data"
os.makedirs(output_data_dir, exist_ok=True)

# Export the City_Data into a CSV file
city_data_df.to_csv(os.path.join(output_data_dir, "cities.csv"), index_label="City_ID")

# Read saved data
city_data_df = pd.read_csv("output_data/cities.csv", index_col="City_ID")

# Display sample data
city_data_df.head()

### Create the Scatter Plots Requested

#### Latitude Vs. Temperature

# Build scatter plot for latitude vs. temperature
#These followed the logic used in class assignments
plt.scatter(city_data_df["Lat"], city_data_df["Max Temp"], edgecolors="black", alpha=0.8)

# Incorporate the other graph properties
plt.title("City Latitude vs. Max Temperature (Date)")
plt.xlabel("Latitude")
plt.ylabel("Max Temperature (F)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig1.png")

# Show plot
plt.show()

#### Latitude Vs. Humidity

# Build scatter plot for latitude vs. humidity
plt.scatter(city_data_df["Lat"], city_data_df["Humidity"], edgecolors="black", alpha=0.8)

# Incorporate the other graph properties
plt.title("City Latitude vs. Humidity (Date)")
plt.xlabel("Latitude")
plt.ylabel("Humidity (%)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig2.png")

# Show plot
plt.show()

#### Latitude Vs. Cloudiness

# Build scatter plot for latitude vs. cloudiness
plt.scatter(city_data_df["Lat"], city_data_df["Cloudiness"], edgecolors="black", alpha=0.8)

# Incorporate the other graph properties
plt.title("City Latitude vs. Cloudiness (Date)")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness (%)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig3.png")

# Show plot
plt.show()

#### Latitude vs. Wind Speed Plot

# Build scatter plot for latitude vs. wind speed
plt.scatter(city_data_df["Lat"], city_data_df["Wind Speed"], edgecolors="black", alpha=0.8)

# Incorporate the other graph properties
plt.title("City Latitude vs. Wind Speed (Date)")
plt.xlabel("Latitude")
plt.ylabel("Wind Speed (mph)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig4.png")

# Show plot
plt.show()

---

## Requirement 2: Compute Linear Regression for Each Relationship


# Define a function to create Linear Regression plots
def plot_linear_regression(x_values, y_values, x_label, y_label):
    # Perform linear regression
    (slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)

    # Calculate the regression line
    regress_values = x_values * slope + intercept

    # Create equation of the line
    line_eq = f"y = {round(slope, 2)}x + {round(intercept, 2)}"

    # Create scatter plot
    plt.scatter(x_values, y_values, edgecolors="black", alpha=0.8)

    # Plot the regression line
    plt.plot(x_values, regress_values, "r-")

    # Annotate the line equation
    plt.annotate(line_eq, (min(x_values), min(y_values)), fontsize=12, color="red")

    # Label plot
    plt.title(f"{x_label} vs. {y_label} Linear Regression")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Display R-squared value
    print(f"The r-squared is: {rvalue**2}")

    # Show plot
    plt.show()

# Create a DataFrame with Northern Hemisphere data
northern_hemi_df = city_data_df[city_data_df['Lat'] >= 0]

# Display sample data
northern_hemi_df.head()

# Create a DataFrame with the Southern Hemisphere data (Latitude < 0)
# Create a DataFrame with Northern Hemisphere data
southern_hemi_df = city_data_df[city_data_df['Lat'] < 0]

# Display sample data
southern_hemi_df.head()

###  Temperature vs. Latitude Linear Regression Plot

# Linear regression on Northern Hemisphere
def plot_linear_regression(x_values, y_values, title, text_coordinates):
    # Perform linear regression
    (slope, intercept, r_value, p_value, std_err) = linregress(x_values, y_values)

    # Get regression values
    regress_values = x_values * slope + intercept

    # Create line equation string
    line_eq = "y = " + str(round(slope, 2)) + "x + " + str(round(intercept, 2))

    # Create scatter plot
    plt.scatter(x_values, y_values)

    # Plot regression line
    plt.plot(x_values, regress_values, "r-")

    # Annotate the linear equation on the plot
    plt.annotate(line_eq, text_coordinates, fontsize=15, color="red")

    # Label plot
    plt.xlabel("Latitude")
    plt.ylabel(title)

    # Print r-value (correlation coefficient)
    print(f"The r-value is: {r_value}")

    # Show plot
    plt.show()

# Call the function to create the Northern Hemisphere plots
plot_linear_regression(northern_hemi_df["Lat"], northern_hemi_df["Max Temp"], "Max Temp (F) vs. Latitude Linear Regression", (10, -20))

# Linear regression on Southern Hemisphere
def plot_linear_regression(x_values, y_values, title, text_coordinates):
    # Perform linear regression
    (slope, intercept, r_value, p_value, std_err) = linregress(x_values, y_values)

    # Get regression values
    regress_values = x_values * slope + intercept

    # Create line equation string
    line_eq = "y = " + str(round(slope, 2)) + "x + " + str(round(intercept, 2))

    # Create scatter plot
    plt.scatter(x_values, y_values)

    # Plot regression line
    plt.plot(x_values, regress_values, "r-")

    # Annotate the linear equation on the plot
    plt.annotate(line_eq, text_coordinates, fontsize=15, color="red")

    # Label plot
    plt.xlabel("Latitude")
    plt.ylabel(title)

    # Print r-value (correlation coefficient)
    print(f"The r-value is: {r_value}")

    # Show plot
    plt.show()

# Call the function to create the Northern Hemisphere plots
plot_linear_regression(southern_hemi_df["Lat"], southern_hemi_df["Max Temp"], "Max Temp (F) vs. Latitude Linear Regression", (10, -20))

For both the Northern and the Southern Hemispheres the distance from the Equator negatively correlated with Max Temperatures, as for both they decreased, the further latitudes from the equator.

### Humidity vs. Latitude Linear Regression Plot

# Northern Hemisphere
plot_linear_regression(northern_hemi_df["Lat"], northern_hemi_df["Humidity"], "Humidity (%) vs. Latitude Linear Regression", (40, 15))

# Southern Hemisphere
plot_linear_regression(southern_hemi_df["Lat"], southern_hemi_df["Humidity"], "Humidity (%) vs. Latitude Linear Regression", (40, 15))

Humdity is positively correlated with distance from the Equator.

### Cloudiness vs. Latitude Linear Regression Plot

# Northern Hemisphere
plot_linear_regression(northern_hemi_df["Lat"], northern_hemi_df["Cloudiness"], "Cloudiness (%) vs. Latitude Linear Regression", (40, 15))

# Southern Hemisphere
plot_linear_regression(southern_hemi_df["Lat"], southern_hemi_df["Cloudiness"], "Cloudiness (%) vs. Latitude Linear Regression", (40, 15))

Cloudiness is positively correlated with distance from the Equator.

### Wind Speed vs. Latitude Linear Regression Plot

# Northern Hemisphere
plot_linear_regression(northern_hemi_df["Lat"], northern_hemi_df["Wind Speed"], "Wind Speed (mph) vs. Latitude Linear Regression", (40, 25))

# Southern Hemisphere
plot_linear_regression(southern_hemi_df["Lat"], southern_hemi_df["Wind Speed"], "Wind Speed (mph) vs. Latitude Linear Regression", (40, 25))

Windiness does not have a strong correlation either way.

VacationPy
# VacationPy
---

## Starter Code to Import Libraries and Load the Weather and Coordinates Data

# Dependencies and Setup
import hvplot.pandas
import pandas as pd
import requests

# Import API key
from api_keys import geoapify_key

# Load the CSV file created in Part 1 into a Pandas DataFrame
city_data_df = pd.read_csv("output_data/cities.csv")
# Convert Max Temp from Kelvin to Celsius
#Got this from https://stackoverflow.com/questions/19477324/how-do-i-calculate-the-temperature-in-celsius-returned-in-openweathermap-org-jso.
I tried to do what had been done in class before but it wasn't working so I resorted to this.
city_data_df["Max Temp (C)"] = city_data_df["Max Temp"] - 273.15

# Display sample data
city_data_df.head()

---

### Step 1: Create a map that displays a point for every city in the `city_data_df` DataFrame. The size of the point should be the humidity in each city.

%%capture --no-display

# Configure the map plot_1
map_plot_1 = city_data_df.hvplot.points(
    "Lng",
    "Lat",
    geo = True,
    tiles="OSM"
)

# Display the map plot_1
map_plot_1

# Configure the map plot
map_plot = city_data_df.hvplot.points(
    'Lng',
    'Lat',
    c='Humidity',
    hover_cols=['City', 'Humidity'],
    geo=True,
    tiles="OSM",
    size='Humidity',
    title='City Map with Humidity',
    frame_width=800,
)

# Display the map
map_plot

### Step 2: Narrow down the `city_data_df` DataFrame to find your ideal weather condition

# Narrow down cities that fit criteria
ideal_weather_df = city_data_df[
    (city_data_df["Max Temp (C)"] > 21) & (city_data_df["Max Temp (C)"] < 27) &
    (city_data_df["Wind Speed"] < 4.5) & (city_data_df["Cloudiness"] == 0)
]

# Drop any rows with null values
ideal_weather_df = ideal_weather_df.dropna()

# Display sample data
ideal_weather_df.head()

### Step 3: Create a new DataFrame called `hotel_df`.

# Use the Pandas copy function to create DataFrame called hotel_df
hotel_df = ideal_weather_df.copy()

# Add an empty column, "Hotel Name," to the DataFrame
hotel_df["Hotel Name"] = ""

# Display sample data
hotel_df.head()

### Step 4: For each city, use the Geoapify API to find the first hotel located within 10,000 metres of your coordinates.
#Got the premise of how to do this from geoviews_demo and other class assignments

# Set parameters to search for a hotel
radius = 10000  # You can adjust the radius as needed
params = {
    "type": "lodging",
    "radius": radius,
}

# Print a message to follow up the hotel search
print("Starting hotel search")

# Iterate through the hotel_df DataFrame
#Got how to do the base url and others from 
for index, row in hotel_df.iterrows():
    # Get latitude, longitude from the DataFrame
    lat = row["Lat"]
    lng = row["Lng"]
    
    # Add filter and bias parameters with the current city's latitude and longitude to the params dictionary
    params["filter"] = f"point:{lng},{lat}"
    params["bias"] = f"point:{lng},{lat}"
    
    # Set base URL
    base_url = "https://api.geoapify.com/v2/places"

    # Make an API request using the params dictionary
    name_address = requests.get(base_url, params=params).json()
    
    # Grab the first hotel from the results and store the name in the hotel_df DataFrame
    try:
        hotel_df.loc[index, "Hotel Name"] = name_address["features"][0]["properties"]["name"]
    except (KeyError, IndexError):
        # If no hotel is found, set the hotel name as "No hotel found".
        hotel_df.loc[index, "Hotel Name"] = "No hotel found"
        
    # Log the search results
    print(f"{hotel_df.loc[index, 'City']} - nearest hotel: {hotel_df.loc[index, 'Hotel Name']}")

# Display sample data
hotel_df.head()

### Step 5: Add the hotel name and the country as additional information in the hover message for each city in the map.

# Configure the map plot_2
map_plot_2 = hotel_df.hvplot.points(
    x="Lng",
    y="Lat",
    c="Humidity",
    hover_cols=["Humidity", "City", "Hotel Name", "Country"],
    geo=True,
    tiles="OSM"
)

# Display the map plot_2
map_plot_2



