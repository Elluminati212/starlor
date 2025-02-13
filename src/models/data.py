

# import calendar
# from concurrent.futures import ThreadPoolExecutor
# import pymongo
# import pandas as pd
# import requests
# from datetime import datetime, timedelta


# # Function to call the API and get credentials, with a fallback to default credentials
# def get_api_credentials():
#     url = 'http://apiplatform.elluminatiinc.net/api/service_config/credentials?product=STARLOR&service=AI_PREDICTION'
#     try:
#         response = requests.get(url, timeout=10)  # Adding a timeout for better error handling
#         if response.status_code == 200:
#             return response.json()
#         else:
#             print(f"API call failed with status code {response.status_code}. Using default credentials.")
#     except (requests.RequestException, ValueError) as e:
#         print(f"Error during API call: {e}. Using default credentials.")
    
#     # Default credentials
#     return {
#         "data": {
#             "serviceConfig": {
#                 "dbUrl": "mongodb+srv://read_access:g4WJ9ZstQzwsa3lm@rydexnew.issgelo.mongodb.net/RydeX_history?retryWrites=true&w=majority",
#                 "dbName": "RydeX_history",
#                 "collection": ["booking_histories", "booking_invoice_histories"]
#             }
#         }
#     }

# def fetch_chunk_by_date_range(mongo_uri, db_name, collection_name, start_date, end_date, skip, limit):
#     client = pymongo.MongoClient(mongo_uri)
#     db = client[db_name]
#     collection = db[collection_name]
#     cursor = collection.find(
#         {
#             "provider_trip_start_time": {
#                 "$gte": start_date,
#                 "$lt": end_date
#             }
#         }
#     ).skip(skip).limit(limit)
#     return list(cursor)

# # MongoDB connection details
# dbUrl = "mongodb+srv://read_access:g4WJ9ZstQzwsa3lm@rydexnew.issgelo.mongodb.net/RydeX_history?retryWrites=true&w=majority"
# dbName = "RydeX_history"
# collections = ["booking_histories", "booking_invoice_histories"]

# # Connect to MongoDB
# client = pymongo.MongoClient(dbUrl)
# dbName = client[dbName]
# pipeline = [
#     {
#         "$lookup": {
#             "from": "booking_invoice_histories",
#             "localField": "bookingInvoiceId",
#             "foreignField": "_id",
#             "as": "invoice"
#         }
#     },
#     {
#         "$project": {
#             "_id": 1,
#             "bookingType": 1,
#             "status": 1,
#             "pickupTime": "$pickupAddress.createdAt",
#             "dropTime": "$destinationAddresses.createdAt",
#             "source": [{
#                 "latitude": "$pickupAddress.latitude",
#                 "longitude": "$pickupAddress.longitude"
#             }],
#             "destination": [{
#                 "latitude": "$destinationAddresses.latitude",
#                 "longitude": "$destinationAddresses.longitude"
#             }],
#             "distance": "$driverToCustomerEstimation.distance",
#             "total": { "$arrayElemAt": [ "$invoice.estimated.total", 0 ] } ,
#         }
#     }
# ]

# # Execute the aggregation
# results = dbName.booking_histories.aggregate(pipeline)


# # for doc in results:
# #     print(doc)

# data = list(results)

# # Create a pandas DataFrame from the list of dictionaries
# df1 = pd.json_normalize(data)
# df1.to_csv('doc.csv', index=False)

# # Remap the fields of the existing DataFrame `df` to the specified fields
# df = df1.rename(columns={
#     "_id": "_id",
#     "status": "is_trip_completed",
#     "bookingType": "booking_type",
#     "pickupTime": "provider_trip_start_time",
#     "dropTime": "provider_trip_end_time",
#     "source": "sourceLocation",
#     "distance": "total_distance",
#     "total": "total",
#     "destination": "destinationLocation"
# })
# # df.to_csv("doc.csv", index=False)



# def fetch_data_by_month(mongo_uri, db_name, collection_name, start_date, end_date, chunk_size):
#     client = pymongo.MongoClient(mongo_uri)
#     db = client[db_name]
#     collection = db[collection_name]
#     total_docs = collection.count_documents({
#         "provider_trip_start_time": {
#             "$gte": start_date,
#             "$lt": end_date
#         }
#     })
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         for i in range(0, total_docs, chunk_size):
#             futures.append(executor.submit(fetch_chunk_by_date_range, mongo_uri, db_name, collection_name, start_date, end_date, i, chunk_size))

#         # Combine results from all threads
#         all_data = []
#         for future in futures:
#             all_data.extend(future.result())
#         return all_data

# def fetch_data_for_all_months(mongo_uri, db_name, collection_name, start_date, end_date, chunk_size):
#     current_date = start_date
#     all_data = []

#     while current_date < end_date:
#         # Calculate the start and end of the current month
#         first_day_of_month = current_date.replace(day=1)
#         last_day_of_month = current_date.replace(
#             day=calendar.monthrange(current_date.year, current_date.month)[1]
#         )
#         next_month = last_day_of_month + timedelta(days=1)

#         print(f"Fetching data from {first_day_of_month} to {next_month}")
#         monthly_data = fetch_data_by_month(mongo_uri, db_name, collection_name, first_day_of_month, next_month, chunk_size)
#         all_data.extend(monthly_data)
#         current_date = next_month

#     return all_data

# if __name__ == "__main__":
#     credentials = get_api_credentials()
#     print(credentials)

#     # Extract MongoDB credentials from the API response
#     try:
#         mongo_uri = credentials['data']['serviceConfig']['dbUrl']
#         db_name = credentials['data']['serviceConfig']['dbName']
#         collection_name = credentials['data']['serviceConfig']['collection']
#     except KeyError:
#         mongo_uri = "mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/"
#         db_name = "RydeX_history"
#         collection_name = "booking_histories"

#     # Define the overall date range for your query
#     start_date = datetime(2024, 12, 1)  # Adjust as needed
#     end_date = datetime(2024, 12, 31)  # Adjust as needed
#     chunk_size = 1000  # Number of documents per thread

#     # Fetch data for all months
#     all_data = fetch_data_for_all_months(mongo_uri, db_name, str(collection_name), start_date, end_date, chunk_size)

#     # Convert to Pandas DataFrame
#     df = pd.DataFrame(all_data)

#     # Save to CSV file
#     df.to_csv('doc.csv', index=False)

#     # Display the DataFrame
#     print(df.head())



import pymongo
import pandas as pd
import requests
from datetime import datetime, timedelta

# API URL to get DB credentials
api_url = "https://example.com/get_db_credentials"

# Function to fetch DB credentials from API
def get_db_credentials(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data['dbUrl'], data['dbName'], data['collections']
    except Exception as e:
        print("Failed to fetch DB credentials:", e)
        # Use default credentials in case of error
        return {
        "data": {
            "serviceConfig": {
                "dbUrl": "mongodb+srv://read_access:g4WJ9ZstQzwsa3lm@rydexnew.issgelo.mongodb.net/RydeX_history?retryWrites=true&w=majority",
                "dbName": "RydeX_history",
                "collection": "booking_histories"
            }
        }
        }
    
# Get DB details from API
db_url, db_name, collections = get_db_credentials(api_url)

# Credential check
if not db_url or not db_name or not collections:
    raise ValueError("Database URL, Name, and Collections must be provided.")

# Connect to MongoDB
client = pymongo.MongoClient(db_url)
db = client[db_name]

# Function to fetch data by month range
def fetch_data_by_month(start_date, end_date):
    pipeline = [
        {
            "$match": {
                "pickupAddress.createdAt": {
                    "$gte": start_date,
                    "$lt": end_date
                }
            }
        },
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
                "total": {"$arrayElemAt": ["$invoice.estimated.total", 0]},
            }
        }
    ]

    # Execute the aggregation
    results = db.booking_histories.aggregate(pipeline)
    return list(results)

# Set the date range for data fetching
start_date = datetime(2023, 1, 1)  # Start from January 2023
end_date = datetime(2023, 12, 31)  # Until December 2023

# Iterate through each month
current_date = start_date
all_data = []

while current_date < end_date:
    next_month = current_date + timedelta(days=31)
    next_month = next_month.replace(day=1)
    
    print(f"Fetching data from {current_date} to {next_month}")
    monthly_data = fetch_data_by_month(current_date, next_month)
    all_data.extend(monthly_data)
    
    # Move to the next month
    current_date = next_month

# Create a pandas DataFrame from the list of dictionaries
df1 = pd.json_normalize(all_data)
df1.to_csv('doc.csv', index=False)

# Remap the fields of the existing DataFrame `df1`
df = df1.rename(columns={
    "_id": "_id",
    "status": "is_trip_completed",
    "bookingType": "booking_type",
    "pickupTime": "provider_trip_start_time",
    "dropTime": "provider_trip_end_time",
    "source": "sourceLocation",
    "distance": "total_distance",
    "total": "total",
    "destination": "destinationLocation"
})

# Save to CSV
df.to_csv("doc_remapped.csv", index=False)

print("Data fetching and transformation completed successfully.")