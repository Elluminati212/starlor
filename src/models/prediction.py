import pymongo
import pandas as pd
import requests

# MongoDB connection details
response = requests.get('http://apiplatform.elluminatiinc.net/api/service_config/credentials?product=STARLOR&service=AI_PREDICTION')
if response.status_code == 200:
    data = response.json().get("data", {})
    db_url = data.get("dbUrl", "")
    db = data.get("dbName", "")
    collections = data.get("collections", [])
else:
    print(f"Failed to fetch credentials from service config API: {response.status_code}")
    db_url = "mongodb+srv://read_access:g4WJ9ZstQzwsa3lm@rydexnew.issgelo.mongodb.net/RydeX_history?retryWrites=true&w=majority"
    db = "RydeX_history"
    collections = ["booking_histories", "booking_invoice_histories"]

# Connect to MongoDB
client = pymongo.MongoClient(db_url)
db = client[db]
pipeline = [
    {
        "$lookup": {
            "from": "booking_invoice_histories",
            "localField": "bookingInvoiceId",
            "foreignField": "_id",
            "as": "invoice"
        }
    },
    {
        "$project": {
            "_id": 1,
            "bookingType": 1,
            "status": 1,
            "cityId": 1,
            "countryId": 1,
            "pickupTime": "$pickupAddress.createdAt",
            "dropTime": "$destinationAddresses.createdAt",
            "source": [{
                "latitude": "$pickupAddress.latitude",
                "longitude": "$pickupAddress.longitude"
            }],
            "destination": [{
                "latitude": "$destinationAddresses.latitude",
                "longitude": "$destinationAddresses.longitude"
            }],
            "distance": "$driverToCustomerEstimation.distance",
            "total": { "$arrayElemAt": [ "$invoice.estimated.total", 0 ] } ,
        }
    }
]

# Execute the aggregation
results = db.booking_histories.aggregate(pipeline)


# for doc in results:
#     print(doc)

data = list(results)

# Create a pandas DataFrame from the list of dictionaries
df1 = pd.json_normalize(data)
df1.to_csv('doc.csv', index=False)

# Remap the fields of the existing DataFrame `df` to the specified fields
df = df1.rename(columns={
    "_id": "_id",
    "countryId": "countryId",
    "cityId": "cityId",
    "status": "is_trip_completed",
    "bookingType": "booking_type",
    "pickupTime": "provider_trip_start_time",
    "dropTime": "provider_trip_end_time",
    "source": "sourceLocation",
    "distance": "total_distance",
    "total": "total",
    "destination": "destinationLocation"
})
# df.to_csv("doc.csv", index=False)

import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from prophet import Prophet
import branca.colormap as cm

# First, let's check the available columns
print("Available columns:", df.columns.tolist())

# Filter for completed trips and booking type
df_completed = df[df['is_trip_completed'] == '1']
df_completed = df[df['booking_type'] == '1']
df_completed = df[~((df['booking_type'] == '2') & (df['booking_type'] == '4'))]

# Extract latitude and longitude
df_completed['latitude'] = df['sourceLocation'].apply(lambda x: x[0].get('latitude') if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) else None)
df_completed['longitude'] = df['sourceLocation'].apply(lambda x: x[0].get('longitude') if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) else None)

# Check if we have a timestamp column (might have a different name)
timestamp_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
print("Possible timestamp columns:", timestamp_columns)

# Assuming we find the correct timestamp column, let's use it
# Replace 'provider_trip_start_time' with the actual column name from your DataFrame
timestamp_column = timestamp_columns[0] if timestamp_columns else None

if timestamp_column is None:
    raise ValueError("No timestamp column found. Please specify the correct column name.")

# Convert timestamp and create datetime features
df_completed['datetime'] = pd.to_datetime(df_completed[timestamp_column])
df_completed['date'] = df_completed['datetime'].dt.date
df_completed['hour'] = df_completed['datetime'].dt.hour
df_completed['minute'] = df_completed['datetime'].dt.minute

# Rest of the clustering code
X = df_completed[['latitude', 'longitude']]
coords = X[['latitude', 'longitude']].dropna().to_numpy()

# DBSCAN clustering
epsilon = 0.01
min_samples = 10
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coords)
df_completed['cluster'] = dbscan.labels_

# Filter for high-demand clusters (exclude noise: label -1)
high_demand_clusters = df_completed[df_completed['cluster'] != -1]

# Aggregate trip data
aggregated_data = df_completed.groupby(['date', 'hour', 'minute']).size().reset_index(name='trip_count')

# Prepare data for Prophet
aggregated_data['datetime'] = pd.to_datetime(
    aggregated_data['date'].astype(str) + ' ' +
    aggregated_data['hour'].astype(str) + ':' +
    aggregated_data['minute'].astype(str)
)

# Apply log transformation
aggregated_data['y_log'] = np.log1p(aggregated_data['trip_count'])
prophet_data = aggregated_data[['datetime', 'y_log']].rename(columns={'datetime': 'ds', 'y_log': 'y'})

# Prophet model
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.fit(prophet_data)

# Create future predictions
future = model.make_future_dataframe(periods=17520, freq='15min')
forecast = model.predict(future)

# Inverse log transformation
forecast['yhat'] = np.expm1(forecast['yhat'])
forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

# Monthly aggregation
forecast['month'] = forecast['ds'].dt.to_period('M')
monthly_trip_summary = forecast.groupby('month')['yhat'].sum().reset_index()
monthly_trip_summary.rename(columns={'yhat': 'predicted_trip_count'}, inplace=True)
monthly_trip_summary['predicted_trip_count'] *= 0.9  # Scaling factor

# Prepare for visualization
forecast['date'] = forecast['ds'].dt.date
forecast['hour'] = forecast['ds'].dt.hour
forecast['minute'] = forecast['ds'].dt.minute

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
predicted_trips = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'date', 'hour', 'minute']]
predicted_trips['cluster'] = kmeans.fit_predict(predicted_trips[['hour', 'minute']])

# Add cityId and countryId
predicted_trips = predicted_trips.merge(df_completed[['cluster', 'cityId', 'countryId']].drop_duplicates(), on='cluster', how='left')

# Add price range
price_range = df_completed.groupby('cluster')['total'].agg(['min', 'max']).reset_index()
price_range.rename(columns={'min': 'price_min', 'max': 'price_max'}, inplace=True)
predicted_trips = pd.merge(predicted_trips, price_range, on='cluster', how='left')

# Create visualization
def get_color(predicted_trips):
    if predicted_trips < 10:
        return 'red'
    elif predicted_trips > 7:
        return 'yellow'
    elif predicted_trips > 2:
        return 'green'
    else:
        return 'gray'

# Create map
m = folium.Map(location=[df_completed['latitude'].mean(), df_completed['longitude'].mean()], zoom_start=12)
colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=10)
colormap.caption = 'Predicted Trips (cluster)'

for _, row in predicted_trips.iterrows():
    color = get_color(row['yhat'])
    folium.CircleMarker(
        location=[df_completed['latitude'].mean(), df_completed['longitude'].mean()],
        radius=row['yhat'] / 10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        tooltip=f"""
            Time: {row['ds']}<br>
            Predicted Trips: {int(row['yhat'])}<br>
            Cluster: {row['cluster']}<br>
            Price Range: {row['price_min']} - {row['price_max']}
        """
    ).add_to(m)

colormap.add_to(m)

# Save outputs
predicted_trips.to_csv("all_predictions.csv", index=False)
m.save("all_predictions.html")



# Visualize the forecast
model.plot(forecast).show()
model.plot_components(forecast).show()

print("Predicted trips for next month saved to 'all predition.csv'")

# # Extract the month and year from the 'ds' column
# predicted_trips['month'] = predicted_trips['ds'].dt.to_period('M')  # Year-Month format

# # Group predicted trips by month and calculate the sum of predicted trips in each month
# monthly_trip_summary = predicted_trips.groupby('month')['yhat'].sum().reset_index()

# # Rename the columns for clarity
# monthly_trip_summary.rename(columns={'yhat': 'predicted_trip_count'}, inplace=True)

# # Display the summary
# print("Predicted trips per month:")
# print(monthly_trip_summary)

# # Group predicted trips by cluster and calculate the sum of predicted trips in each cluster
# clustered_trips_summary = predicted_trips.groupby('cluster')['yhat'].sum().reset_index()

# # Rename the columns for clarity
# clustered_trips_summary.rename(columns={'yhat': 'predicted_trip_count'}, inplace=True)

# # Display the summary
# print(clustered_trips_summary)



import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import pandas as pd
from datetime import datetime

# Assuming 'high_demand_clusters' and 'predicted_trips' are already defined
# Prepare the data for clustering, including 'ds' (date/time)
cluster_data = high_demand_clusters[['latitude', 'longitude', 'cluster', 'booking_type', 'cityId', 'countryId']].merge(
    predicted_trips[['ds', 'price_min', 'price_max']], left_index=True, right_index=True, how='left'
)

# Convert timestamps to datetime
cluster_data['ds'] = pd.to_datetime(
    cluster_data['ds'], format="%d-%m-%Y %H:%M", errors='coerce'
)

# Filter to include only today's trips
today = datetime.now().date()
cluster_data['date_stamp'] = cluster_data['ds'].dt.strftime('%d-%m-%Y %H:%M')
cluster_data = cluster_data[cluster_data['ds'].dt.date == today]

# Save the filtered data to a CSV file
cluster_data.drop(columns=['ds', 'cluster']).to_csv("/home/vasu/starlor/starlor/Map_Today.csv", index=False)

# Initialize the map
m = folium.Map(
    location=[cluster_data['latitude'].mean(), cluster_data['longitude'].mean()],
    zoom_start=12
)

# Define a color palette for clusters
colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'orange', 'red'], vmin=-1, vmax=cluster_data['cluster'].max())
colormap.caption = 'Clusters'

# Add the color map to the map
colormap.add_to(m)

# Add clusters to the map
marker_cluster = MarkerCluster().add_to(m)

# Add predicted trips (pink color) and new trips (black color)
for _, row in cluster_data.iterrows():
    # Define color based on whether it's a predicted trip or a new trip
    if pd.notna(row['ds']):  # This assumes predicted trips have a valid timestamp in 'ds'
        cluster_color = 'Red'  # Predicted trips
    else:
        cluster_color = 'purple'  # New trips (assuming these don't have a 'ds' value)

    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=6,
        color=cluster_color,
        fill=True,
        fill_color=cluster_color,
        fill_opacity=0.7,
        tooltip=(
            f"Cluster: {row['cluster']}<br>"
            f"Timestamp: {row['ds']}<br>"
	        f"Price Range: {row['price_min']} - {row['price_max']}<br>"
            f"Booking Type: {row['booking_type']}<br>"
            f"City: {row['cityId']}<br>"
            f"Country: {row['countryId']}<br>"
        )
    ).add_to(marker_cluster)

# Save the map to an HTML file
m.save("predicted_Map.html")

print("Map with today's trips saved as 'predicted_Map.html'.")
print("Filtered cluster data saved as 'Map_Today.csv'.")



import pandas as pd
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta

# Load data
predicted_trips = pd.read_csv("/home/vasu/starlor/starlor/Map_Today.csv")

# Ensure datetime column is present
predicted_trips['datetime'] = pd.to_datetime(predicted_trips['date_stamp'])  # Adjust 'ds' to match the actual column name

# Get the current time and the next hour
current_time = datetime.now()
next_hour = current_time + timedelta(hours=1)

# Filter trips for the upcoming one hour
upcoming_trips = predicted_trips[
    (predicted_trips['datetime'] >= current_time) & 
    (predicted_trips['datetime'] < next_hour)
]

# Create a map centered at the average coordinates of upcoming trips
if not upcoming_trips.empty:
    center_lat = upcoming_trips['latitude'].mean()
    center_lon = upcoming_trips['longitude'].mean()
else:
    # Fallback center if no upcoming trips
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
    # Create a MarkerCluster instance with custom supercluster options
    cluster = MarkerCluster(
        supercluster_options={
            'radius': 40,
            'minPoints': 2,
            'color': '#ff0000'  # Red color
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
                f"Price Range: {min_price} - {max_price}<br>"
            	f"Booking Type: {row['booking_type']}<br>"
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
m.save('upcoming_trips_next_hour_red.html')
print("upcoming trips saved as 'upcoming_trips_next_hour_red.html'.")


# Add documents to WriteDB


import pymongo
import pandas as pd
from pymongo import MongoClient
# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/", socketTimeoutMS=360000, connectTimeoutMS=360000, serverSelectionTimeoutMS=360000)

# Select the database and collection
db = client["starlorDB"]
collection = db["prediction"]

# Read the CSV file
df = pd.read_csv("/home/vasu/starlor/starlor/Map_Today.csv")
print(df)
# Insert the data into MongoDB
collection.insert_many(df.to_dict("records"))


# Close the MongoDB connection
client.close()





