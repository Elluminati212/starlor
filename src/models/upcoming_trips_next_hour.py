
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta

predicted_trips = pd.read_csv("/home/ubuntu/Map_Today.csv")

predicted_trips['datetime'] = pd.to_datetime(predicted_trips['ds'])

# Get the current time and the next hour
current_time = datetime.now()
next_hour = current_time + timedelta(hours=1)

upcoming_trips = predicted_trips[
    (predicted_trips['datetime'] >= current_time) & 
    (predicted_trips['datetime'] < next_hour)
]

# Create a map centered at the high demand coordinates of upcoming trips
if not upcoming_trips.empty:
    center_lat = upcoming_trips['latitude'].mean()
    center_lon = upcoming_trips['longitude'].mean()
else:
    # Fallback center if no trips
    center_lat = predicted_trips['latitude'].mean()
    center_lon = predicted_trips['longitude'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

if upcoming_trips.empty:
    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.Icon(color='gray', icon='info-sign'),
        popup='Insufficient data'
    ).add_to(m)
else:

    cluster = MarkerCluster(
        supercluster_options={
            'radius': 40,
            'minPoints': 2,
            'color': '#ff0000' 
        }
    ).add_to(m)

    # Add upcoming trips to the cluster
    for _, row in upcoming_trips.iterrows():
        min_price = row.get('price_min', 'N/A')  # Fallback to 'N/A' if column not found
        max_price = row.get('price_max', 'N/A')  # Fallback to 'N/A' if column not found

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=(
                f"Predicted Trip<br>Time: {row['datetime']}<br>"
                f"Price Range: {min_price} - {max_price}"
            )
        ).add_to(cluster)

        # Add a light red circle around the trip marker
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=2000,
            color='#ff0000',  # Light red color
            fill=True,
            fill_opacity=0.2
        ).add_to(m)


# Save the map to an HTML file
m.save('upcoming_trips_next_hour.html')
print("upcoming trips saved as 'upcoming_trips_next_hour.html'.")

import pandas as pd
import json
from datetime import datetime, timedelta

# Load data
predicted_trips = pd.read_csv("/home/ubuntu/Map_Today.csv")

# Ensure datetime column is present
predicted_trips['datetime'] = pd.to_datetime(predicted_trips['ds'])  # Adjust 'ds' to match the actual column name

# Get the current time and the next hour
current_time = datetime.now()
next_hour = current_time + timedelta(hours=1)

# Filter trips for the upcoming one hour
upcoming_trips = predicted_trips[
    (predicted_trips['datetime'] >= current_time) & 
    (predicted_trips['datetime'] < next_hour)
]

# Prepare the JSON data
map_data = {
    "trips": []
}

# Add trip details to the JSON
for _, row in upcoming_trips.iterrows():
    trip = {
        "latitude": row['latitude'],
        "longitude": row['longitude'],
        "datetime": row['datetime'].isoformat(),
        "price_min": row.get('price_min', 'N/A'),
        "price_max": row.get('price_max', 'N/A')
    }
    map_data["trips"].append(trip)

# Save the data to a JSON file
with open('upcoming_trips_next_hour.json', 'w') as json_file:
    json.dump(map_data, json_file, indent=4)

print("Upcoming trips saved as 'upcoming_trips_next_hour.json'.")





'''
import pymongo
import pandas as pd
from pymongo import MongoClient
# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/", socketTimeoutMS=360000, connectTimeoutMS=360000, serverSelectionTimeoutMS=360000)

# Select the database and collection
db = client["WriteDB"]
collection = db["prediction"]

# Read the CSV file
df = pd.read_csv("/home/ubuntu/Map_Today.csv")
print(df)
# Insert the data into MongoDB
collection.insert_many(df.to_dict("records"))


# Close the MongoDB connection
client.close()
'''

