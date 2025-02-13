
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from pymongo import MongoClient
from bson.json_util import dumps, loads
import requests
import os
import math
from typing import Optional
from datetime import datetime, timedelta
import pytz
from fastapi import Request
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI()

# @app.get("/map")
# def read_root():
#     with open("/home/vasu/app/src/templetes/map.html", "r") as f:
#          return 
#     HTMLResponse(content=f.read())

#     print(HTMLResponse)

@app.get("/test_connection")
async def test_connection():
    try:
        # Check MongoDB connection
        db.command("ping")
        return {"status": "Connection Successful"}
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
    
@app.post("/update_url")
async def update_url(url: str):
    global CREDENTIALS_URL
    CREDENTIALS_URL = url
    return {"message": "URL updated successfully"}

# Helper function to handle both full timestamps and date-only filtering
def parse_datetime(ds: str) -> Optional[datetime]:
    try:
        # Try parsing the full timestamp "YYYY-MM-DD HH:MM:SS"
        return datetime.strptime(ds, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # If it's just a date "YYYY-MM-DD", return the start of that day
            return datetime.strptime(ds, "%Y-%m-%d")
        except ValueError:
            # If it doesn't match either format, return None
            return None    

# MongoDB connection
MONGO_URI = "mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/"  # WriteDB MongoDB URI
DB_NAME = "starlorDB"  # Database name

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["prediction"]

# Fetch credentials dynamically from API
CREDENTIALS_URL = "http://apiplatform.elluminatiinc.net/api/service_config/credentials?product=STARLOR&service=AI_PREDICTION"
response = requests.get(CREDENTIALS_URL)
if response.status_code == 200:
    credentials = response.json().get("data", {})
    EXPECTED_ACCESS_TOKEN = credentials.get("accessToken", "")
    EXPECTED_SERVICE_PROVIDER = credentials.get("serviceProvider", "")
    # EXPECTED_IPS = credentials.get("allowedIPs", "")
else:
    raise Exception("Failed to fetch credentials from service config API")
# print(f"Credentials: {EXPECTED_ACCESS_TOKEN}, {EXPECTED_SERVICE_PROVIDER}, {EXPECTED_IPS}")

# Dependency to authenticate requests
def authenticate_request(
    access_token: str = Header(...),
    service_provider: str = Header(...)
    # ips: str = Header(...)
):
    if access_token != EXPECTED_ACCESS_TOKEN or service_provider != EXPECTED_SERVICE_PROVIDER:
    # if access_token != EXPECTED_ACCESS_TOKEN or service_provider != EXPECTED_SERVICE_PROVIDER or ips != EXPECTED_IPS:

        raise HTTPException(status_code=401, detail="Unauthorized")

# # List of allowed IP addresses
# ALLOWED_IPS = ["192.168.1.100", "203.0.113.5"]  

# # Dependency to authenticate requests with IP-based restrictions
# def authenticate_request(
#     access_token: str = Header(...),
#     service_provider: str = Header(...),
#     request: Request = None  # Access request object to get client IP
# ):
#     client_ip = request.client.host
#     if client_ip not in ALLOWED_IPS:
#         raise HTTPException(status_code=403, detail="Forbidden: Unauthorized IP address")
    
#     if access_token != EXPECTED_ACCESS_TOKEN or service_provider != EXPECTED_SERVICE_PROVIDER:
#         raise HTTPException(status_code=401, detail="Unauthorized")



# Helper function to sanitize float values
def sanitize_data(data):
    for record in data:
        for key, value in record.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                record[key] = None  # Replace invalid float with None
    return data

@app.get("/prediction")
async def get_prediction(
    auth: None = Depends(authenticate_request),
    trip_type: Optional[int] = Query(None),
    booking_type: Optional[str] = Query(None),
    ds: Optional[str] = Query(None),  # Expecting ISO format date-time or specific date (yyyy-mm-dd)
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    hour: Optional[int] = Query(None, ge=0, le=23),  # Specific hour filter (0-23)
):
    try:
        # Build query dynamically
        query = {}

        # Filter by trip_type and booking_type if provided
        if trip_type is not None:
            query["trip_type"] = trip_type
        if booking_type is not None:
            query["booking_type"] = booking_type
        
        # Filter by price range if provided
        if price_min is not None:
            query["price_min"] = {"$gte": price_min}
        if price_max is not None:
            query["price_max"] = {"$lte": price_max}

        if ds is not None:
            try:
                date_obj = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                start_of_day = date_obj
                end_of_day = date_obj + timedelta(days=1)

                # If `ds` is stored as string, use regex matching
                if isinstance(collection.find_one({"ds": {"$exists": True}}).get("ds"), str):
                    query["ds"] = {"$regex": f"^{ds}"}
                else:  # Assume `ds` is a datetime
                    query["ds"] = {"$gte": start_of_day, "$lt": end_of_day}
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        if hour is not None:
            try:
                if ds:
                    query["ds"] = {
                    "$gte": datetime.strptime(f"{ds} {hour}:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC),
                    "$lt": datetime.strptime(f"{ds} {hour + 1}:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC),
                    }
                else:
                    # If only hour is provided, match records with the specified hour across all dates
                    query["ds"] = {"$regex": f".* {hour:02d}:"}  # Match any date with the specified hour
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        # Fetch data from MongoDB
        prediction = list(collection.find(query, {
            "_id": 0,
            "latitude": 1,
            "longitude": 1,
            "cluster": 1,
            "trip_type": 1,
            "booking_type": 1,
            "ds": 1,
            "price_min": 1,
            "price_max": 1
        }))

        # Sanitize data for JSON compliance
        sanitized_prediction = sanitize_data(prediction)
        return sanitized_prediction

    except Exception as e:
        # Log and raise HTTP exception
        print(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
