#connected with API
from pymongo import MongoClient
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import calendar
import requests
# from fields import FIELDS_TO_FETCH

# Function to create a MongoDB client
def create_mongo_client(mongo_uri):
    return MongoClient(mongo_uri)

# Function to fetch a chunk of data for a specific date range
def fetch_chunk_by_date_range(mongo_uri, db_name, collection_name, start_date, end_date, skip, limit):
    client = create_mongo_client(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    query = {
        "provider_trip_start_time": {
            "$gte": start_date,
            "$lt": end_date
        }
    }
    # fields_to_fetch = FIELDS_TO_FETCH
    fields_to_fetch = {
        "_id": 1,
        "is_trip_completed": 1,
        "trip_type": 1,
        "provider_trip_start_time": 1,
        "provider_trip_end_time": 1,
        "sourceLocation": 1,
        "total_distance": 1,
        "total": 1,
        "destinationLocation": 1
    }
def get_nested_value(doc, path):
    """Helper function to safely get nested dictionary values"""
    current = doc
    for part in path.split('.'):
        if isinstance(current, dict):
            current = current.get(path)
        else:
            return None
    return current

def transform_document(doc):
    """Transform MongoDB document to desired format"""
    transformed = {
        "_id": str(doc.get("_id")),
        "bookingType": doc.get("bookingType"),
        "status": doc.get("status"),  # 70 indicates completed trip
        "pickupTime": doc.get("pickupAddress", {}).get("createdAt"),
        "dropTime": doc.get("destinationAddresses", [{}])[0].get("createdAt") if doc.get("destinationAddresses") else None,
        "source": {
            "latitude": doc.get("pickupAddress", {}).get("latitude"),
            "longitude": doc.get("pickupAddress", {}).get("longitude"),
            "address": doc.get("pickupAddress", {}).get("address")
        },
        "destination": {
            "latitude": doc.get("destinationAddresses", [{}])[0].get("latitude"),
            "longitude": doc.get("destinationAddresses", [{}])[0].get("longitude"),
            "address": doc.get("destinationAddresses", [{}])[0].get("address")
        },
        "distance": None,  # Will need to be calculated from directionPath if needed
        "totalFare": None  # Will need to be fetched from bookingInvoiceId if needed
    }
    return transformed

FIELDS_TO_FETCH = {
    "_id": 1,
    "bookingType": 1,
    "status": 1,
    "pickupAddress.createdAt": 1,
    "destinationAddresses.createdAt": 1,
    "pickupAddress.latitude": 1,
    "pickupAddress.longitude": 1,
    "pickupAddress.address": 1,
    "destinationAddresses.latitude": 1,
    "destinationAddresses.longitude": 1,
    "destinationAddresses.address": 1
}

# Function to process data in parallel
def fetch_data_by_month(mongo_uri, db_name, collection_name, start_date, end_date, chunk_size):
    client = create_mongo_client(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    total_docs = collection.count_documents({
        "provider_trip_start_time": {
            "$gte": start_date,
            "$lt": end_date
        }
    })
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, total_docs, chunk_size):
            futures.append(executor.submit(fetch_chunk_by_date_range, mongo_uri, db_name, collection_name, start_date, end_date, i, chunk_size))

        # Combine results from all threads
        all_data = []
        for future in futures:
            all_data.extend(future.result())
        return all_data

# Function to iterate through each month and fetch data
def fetch_data_for_all_months(mongo_uri, db_name, collection_name, start_date, end_date, chunk_size):
    current_date = start_date
    all_data = []

    while current_date < end_date:
        # Calculate the start and end of the current month
        first_day_of_month = current_date.replace(day=1)
        last_day_of_month = current_date.replace(
            day=calendar.monthrange(current_date.year, current_date.month)[1]
        )
        next_month = last_day_of_month + timedelta(days=1)

        print(f"Fetching data from {first_day_of_month} to {next_month}")
        monthly_data = fetch_data_by_month(mongo_uri, db_name, collection_name, first_day_of_month, next_month, chunk_size)
        all_data.extend(monthly_data)
        current_date = next_month

    return all_data

# Function to call the API and get credentials, with a fallback to default credentials
def get_api_credentials():
    url = 'http://apiplatform.elluminatiinc.net/api/service_config/credentials?product=STARLOR&service=AI_PREDICTION'
    try:
        response = requests.get(url, timeout=10)  # Adding a timeout for better error handling
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API call failed with status code {response.status_code}. Using default credentials.")
    except (requests.RequestException, ValueError) as e:
        print(f"Error during API call: {e}. Using default credentials.")
    
    # Default credentials
    return {
        "data": {
            "serviceConfig": {
                "dbUrl": "mongodb+srv://read_access:g4WJ9ZstQzwsa3lm@rydexnew.issgelo.mongodb.net/RydeX_history?retryWrites=true&w=majority",
                "dbName": "RydeX_history",
                "collection": "booking_histories"
            }
        }
    }

# Example usage of the API credentials function
if __name__ == "__main__":
    credentials = get_api_credentials()
    print(credentials)

    # Extract MongoDB credentials from the API response
    try:
        mongo_uri = credentials['data']['serviceConfig']['dbUrl']
        db_name = credentials['data']['serviceConfig']['dbName']
        collection_name = credentials['data']['serviceConfig']['collection']
    except KeyError as e:
        print(f"KeyError: {e} not found in the credentials")
        # Handle the missing key error appropriately
        mongo_uri = "default_mongo_uri"
        db_name = "default_db_name"
        collection_name = "default_collection_name"

    # Define the overall date range for your query
    start_date = datetime(2024, 12, 1)  # Adjust as needed
    end_date = datetime(2024, 12, 2)  # Adjust as needed
    # end_date = datetime.today()  # Use today's date
    chunk_size = 1000  # Number of documents per thread

    # Fetch data for all months
    all_data = fetch_data_for_all_months(mongo_uri, db_name, collection_name, start_date, end_date, chunk_size)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(all_data)

    # Display the DataFrame
    print(df.head())





import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from prophet import Prophet
import branca.colormap as cm

# Filter for completed trips and booking type
df_completed = df[df['is_trip_completed'] == '1']
df_completed = df[df['booking_type'] == '1']
df_completed = df[~((df['booking_type'] == '2') & (df['booking_type'] == '4'))]

df['latitude'] = df['sourceLocation'].apply(lambda x: x[0] if isinstance(x, list) else None)
df['longitude'] = df['sourceLocation'].apply(lambda x: x[1] if isinstance(x, list) else None)

# If the extraction worked, proceed with KMeans clustering
X = df[['latitude', 'longitude']]

# The number of clusters
n_clusters = 10  # change this based on how many clusters

coords = df[['latitude', 'longitude']].dropna().to_numpy()
print(coords.size)

# Define DBSCAN parameters
epsilon = 0.01  # Maximum radius of a neighborhood in kilometers
min_samples = 10  # Minimum number of points in a cluster

# Convert epsilon from kilometers to radians (for geospatial data)
epsilon_rad = epsilon / 6371.0  # Earth radius in kilometers

# Convert start time to datetime and filter completed trips
df['provider_trip_start_time'] = pd.to_datetime(df['provider_trip_start_time'])
df_completed = df[df['is_trip_completed'] == True]

# Extract hour and drop rows with missing lat/lon
df_completed['hour'] = df_completed['provider_trip_start_time'].dt.hour
df_completed = df_completed.dropna(subset=['latitude', 'longitude'])

print(df_completed[['latitude', 'longitude']].head())
print(df_completed[['latitude', 'longitude']].isnull().sum())

# Clustering high-demand areas using DBSCAN
coords = df_completed[['latitude', 'longitude']].values
dbscan = DBSCAN(eps=0.01, min_samples=10).fit(coords)  # Tune `eps` and `min_samples`
df_completed['cluster'] = dbscan.labels_

# Filter for high-demand clusters (exclude noise: label -1)
high_demand_clusters = df_completed[df_completed['cluster'] != -1]

# PROPHET
# Extract relevant columns for Prophet: date and trip count
df_completed['datetime'] = pd.to_datetime(df_completed['provider_trip_start_time'])
df_completed['date'] = df_completed['datetime'].dt.date
df_completed['hour'] = df_completed['datetime'].dt.hour
df_completed['minute'] = df_completed['datetime'].dt.minute

# Aggregate trip data by date, hour, and minute
aggregated_data = df_completed.groupby(['date', 'hour', 'minute']).size().reset_index(name='trip_count')

# Prepare data for Prophet
aggregated_data['datetime'] = pd.to_datetime(
    aggregated_data['date'].astype(str) + ' ' +
    aggregated_data['hour'].astype(str) + ':' +
    aggregated_data['minute'].astype(str)
)

# Apply log transformation to the target variable
aggregated_data['y_log'] = np.log1p(aggregated_data['trip_count'])  # Log transformation

prophet_data = aggregated_data[['datetime', 'y_log']].rename(columns={'datetime': 'ds', 'y_log': 'y'})

# Initialize Prophet model with adjusted settings
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True  # Disable yearly seasonality if not relevant
)

# Fit the model
model.fit(prophet_data)

# Create a DataFrame for future predictions
future = model.make_future_dataframe(periods=17520, freq='15min')

# Forecast trips
forecast = model.predict(future)

# Inverse log transformation
forecast['yhat'] = np.expm1(forecast['yhat'])
forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

# Extract month from 'ds' for monthly aggregation
forecast['month'] = forecast['ds'].dt.to_period('M')

# Group by month and calculate predicted trip count
monthly_trip_summary = forecast.groupby('month')['yhat'].sum().reset_index()

# Rename columns for clarity
monthly_trip_summary.rename(columns={'yhat': 'predicted_trip_count'}, inplace=True)

# Scale down if predictions are too high
scaling_factor = 0.9  # Adjust based on validation
monthly_trip_summary['predicted_trip_count'] *= scaling_factor

# Display adjusted monthly predictions
print("Adjusted Predicted trips per month:")
print(monthly_trip_summary)



from prophet.make_holidays import make_holidays_df

# Create holidays dataframe for India
india_holidays = make_holidays_df(
    year_list=[2024 + i for i in range(10)],  # Generate years from 2024 to 2033
    country='IN'
)

# Display the first 10 rows of the holiday dataframe
print(india_holidays.head(10))

from prophet import Prophet

# Initialize Prophet model with Indian holidays
m_india = Prophet(holidays=india_holidays)



# Filter predictions for the same dates and times
forecast['date'] = forecast['ds'].dt.date
forecast['hour'] = forecast['ds'].dt.hour
forecast['minute'] = forecast['ds'].dt.minute

# Merge forecast with original historical data
predicted_trips = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'date', 'hour', 'minute']]

# Apply K-Means clustering to predicted trips for identifying high-demand areas
kmeans = KMeans(n_clusters=10, random_state=42)  # You can adjust the number of clusters as needed
predicted_trips['cluster'] = kmeans.fit_predict(predicted_trips[['hour', 'minute']])

# Add price range to clusters
price_range = df_completed.groupby('cluster')['total'].agg(['min', 'max']).reset_index()
price_range.rename(columns={'min': 'price_min', 'max': 'price_max'}, inplace=True)

predicted_trips = pd.merge(
    predicted_trips,
    price_range,
    on='cluster',
    how='left'  # Use left join to retain all predicted trips even if some clusters lack price range
)

# Merge booking_type and trip_type into predicted_trips
predicted_trips = pd.merge(
    predicted_trips,
    df_completed[['date', 'hour', 'minute', 'booking_type', 'trip_type']],
    on=['date', 'hour', 'minute'],  # Merge on matching date, hour, and minute
    how='left'  # Use left join to retain all predicted trips even if some rows lack booking_type or trip_type
)

# Function to categorize trips into colors based on predicted trip number
def get_color(predicted_trips):
    if predicted_trips < 10:
        return 'red'
    elif predicted_trips > 7:
        return 'yellow'
    elif predicted_trips > 2:
        return 'green'
    else:
        return 'gray'  # For less than 2 trips, use gray

# Create a map centered on the average latitude and longitude of the historical data
m = folium.Map(location=[df_completed['latitude'].mean(), df_completed['longitude'].mean()], zoom_start=12)

# Define a color map using `branca.colormap`
colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=10)
colormap.caption = 'Predicted Trips (cluster)'

# Plot the predicted trips on the map as colored circles
for _, row in predicted_trips.iterrows():
    color = get_color(row['yhat'])  # Get the color based on predicted trips
    folium.CircleMarker(
        location=[df_completed['latitude'].mean(), df_completed['longitude'].mean()],  # Update with actual locations
        radius=row['yhat'] / 10,  # Scale radius based on the predicted number of trips
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        tooltip=f"""
            Time: {row['ds']}<br>
            Predicted Trips: {int(row['yhat'])}<br>
            Cluster: {row['cluster']}<br>
            Price Range: {row['price_min']} - {row['price_max']}
            f"Booking Type: {row['booking_type']}<br>"
            f"Trip Type: {row['trip_type']}"
        """
    ).add_to(m)

# Add the colormap to the map
colormap.add_to(m)

# Save the map as an HTML file
"""m.save("all prediction.html")"""

# Save the predicted trips with clusters to a CSV file
predicted_trips.to_csv("all prediction.csv", index=False)



# Visualize the forecast
model.plot(forecast).show()
model.plot_components(forecast).show()

print("Predicted trips for next month saved to 'all predition.csv'")

# Extract the month and year from the 'ds' column
predicted_trips['month'] = predicted_trips['ds'].dt.to_period('M')  # Year-Month format

# Group predicted trips by month and calculate the sum of predicted trips in each month
monthly_trip_summary = predicted_trips.groupby('month')['yhat'].sum().reset_index()

# Rename the columns for clarity
monthly_trip_summary.rename(columns={'yhat': 'predicted_trip_count'}, inplace=True)

# Display the summary
print("Predicted trips per month:")
print(monthly_trip_summary)

# Group predicted trips by cluster and calculate the sum of predicted trips in each cluster
clustered_trips_summary = predicted_trips.groupby('cluster')['yhat'].sum().reset_index()

# Rename the columns for clarity
clustered_trips_summary.rename(columns={'yhat': 'predicted_trip_count'}, inplace=True)

# Display the summary
print(clustered_trips_summary)



import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import pandas as pd

# Assuming 'high_demand_clusters' and 'predicted_trips' are already defined
# Calculate price_min and price_max for each cluster
cluster_prices = high_demand_clusters.groupby('cluster')['total'].agg(price_min='min', price_max='max').reset_index()

# Merge booking_type and trip_type and price_min and price-max into cluster_data
cluster_data = high_demand_clusters[['latitude', 'longitude', 'cluster', 'booking_type', 'trip_type']].merge(
    predicted_trips[['ds']], left_index=True, right_index=True, how='left'
).merge(
    cluster_prices, on='cluster', how='left'
)

# Save the data to a CSV file
cluster_data.to_csv("Map.csv", index=False)

# Initialize the map
m = folium.Map(
    location=[high_demand_clusters['latitude'].mean(), high_demand_clusters['longitude'].mean()],
    zoom_start=12
)

# Define a color palette for clusters
colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'orange', 'red'], vmin=-1, vmax=cluster_data['cluster'].max())
colormap.caption = 'Clusters'

# Add the color map to the map
colormap.add_to(m)

# Add clusters to the map
marker_cluster = MarkerCluster().add_to(m)

for _, row in cluster_data.iterrows():
    cluster_color = colormap(row['cluster'])  # Assign color based on cluster ID
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
            f"Trip Type: {row['trip_type']}"
        )
    ).add_to(marker_cluster)

# Save the map to an HTML file
m.save("Map.html")

print("Map with clusters saved as 'Map.html'.")
print("Cluster data saved as 'Map.csv'.")



import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import pandas as pd
from datetime import datetime

# Assuming 'high_demand_clusters' and 'predicted_trips' are already defined
# Prepare the data for clustering, including 'ds' (date/time)
cluster_data = high_demand_clusters[['latitude', 'longitude', 'cluster', 'booking_type', 'trip_type']].merge(
    predicted_trips[['ds', 'price_min', 'price_max']], left_index=True, right_index=True, how='left'
)

# Convert timestamps to datetime
cluster_data['ds'] = pd.to_datetime(
    cluster_data['ds'], format="%d-%m-%Y %H:%M", errors='coerce'
)

# Filter to include only today's trips
today = datetime.now().date()
cluster_data['ds'] = pd.to_datetime(cluster_data['ds'])
cluster_data = cluster_data[cluster_data['ds'].dt.date == today]

# Save the filtered data to a CSV file
cluster_data.to_csv("Map_Today.csv", index=False)

# Save the filtered data to a JSON file
json_file_path = "Map_Today.json"
cluster_data.to_json(json_file_path, orient='records', date_format='iso')

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
            f"Trip Type: {row['trip_type']}"
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
predicted_trips = pd.read_csv("/home/vasu/app/Map_Today.csv")

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
            	f"Trip Type: {row['trip_type']}"
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





