# #   OpenAPI Documentation Available at:
# # - Swagger UI: http://192.168.0.125:8000/docs
# # - ReDoc: http://192.168.0.125:8000/redoc
# # - OpenAPI JSON Schema: http://192.168.0.125:8000/openapi.json
from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
import math
from typing import Optional
from datetime import datetime
import pytz
from fastapi.middleware.cors import CORSMiddleware
 
app = FastAPI()
 
origins = [
    "http://localhost:4200",
    "http://192.168.0.125:8000",
    "http://localhost",
    "http://192.168.0.180:6000"  # Example IP
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
MONGO_URI = "mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/"
DB_NAME = "starlorDB"
 
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["prediction"]
 
def sanitize_data(data):
    # Sanitize NaN and Inf AFTER fetching data
    for record in data:
        for key, value in record.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                record[key] = None
    return data
 
@app.get("/prediction")
async def get_prediction(
    # Date range
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    # Time range (handle either 24-hour or 12-hour format with AM/PM)
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    # Price range
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    # Booking type (existing)
    booking_type: Optional[str] = Query(None),
    # Additional filters
    country_id: Optional[str] = Query(None),
    city_id: Optional[str] = Query(None),
):
    try:
        query = {}

        # 1. Add filter for booking_type
        if booking_type:
            query["booking_type"] = int(booking_type)

        # 2. Add filter for country_id
        if country_id is not None:
            query["countryId"] = country_id

        # 3. Add filter for city_id
        if city_id is not None:
            query["cityId"] = city_id

        # 4. Price Range Filter
        if min_price is not None or max_price is not None:
            price_filter = {}
            if min_price is not None:
                price_filter["$gte"] = min_price
            if max_price is not None:
                price_filter["$lte"] = max_price
            query["price_min"] = price_filter
            query["price_max"] = price_filter

        # 5. Date and Time Filter
        if start_date or end_date:
            date_filter = {}

            def parse_datetime_safely(date_str: Optional[str], time_str: Optional[str], is_end: bool = False) -> Optional[str]:
                """
                Parse date and time strings and return in the format matching the dataset: "DD-MM-YYYY HH:mm"
                """
                if not date_str:
                    return None

                try:
                    # Parse input date (accept either DD-MM-YYYY or MM-DD-YYYY)
                    try:
                        # Try DD-MM-YYYY first
                        date_obj = datetime.strptime(date_str, "%d-%m-%Y")
                    except ValueError:
                        # Try MM-DD-YYYY if DD-MM-YYYY fails
                        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                        # Convert to DD-MM-YYYY format
                        date_obj = date_obj.replace(day=int(date_str.split('-')[1]), 
                                                 month=int(date_str.split('-')[0]))

                    # Handle time component
                    if time_str:
                        try:
                            # Try 24-hour format first
                            time_obj = datetime.strptime(time_str, "%H:%M").time()
                        except ValueError:
                            # Try 12-hour format
                            time_obj = datetime.strptime(time_str, "%I:%M %p").time()
                    else:
                        # Default times: 00:00 for start, 23:59 for end
                        time_obj = datetime.strptime("23:59" if is_end else "00:00", "%H:%M").time()

                    # Combine date and time
                    dt = datetime.combine(date_obj, time_obj)
                    
                    # Return in the format matching your dataset
                    return dt.strftime("%d-%m-%Y %H:%M")

                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid date/time format: {str(e)}. Use DD-MM-YYYY or MM-DD-YYYY for dates and HH:MM (24h) or HH:MM AM/PM (12h) for times."
                    )

            # Parse start and end datetimes
            start_dt = parse_datetime_safely(start_date, start_time, is_end=False)
            end_dt = parse_datetime_safely(end_date, end_time, is_end=True)

            # Add date filters to query
            if start_dt:
                date_filter["$gte"] = start_dt
            if end_dt:
                date_filter["$lte"] = end_dt
            
            if date_filter:
                query["date_stamp"] = date_filter

        # Fetch data from Mongo
        projection = {
            "_id": 0,
            "latitude": 1,
            "longitude": 1,
            "booking_type": 1,
            "date_stamp": 1,
            "price_min": 1,
            "price_max": 1,
            "countryId": 1,
            "cityId": 1,
        }

        print("Query:", query)
        prediction = list(collection.find(query, projection))
        return sanitize_data(prediction)

    except Exception as e:
        print(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
 







# from fastapi import FastAPI, HTTPException, Query
# from pymongo import MongoClient
# import math
# from typing import Optional
# from datetime import datetime
# import pytz
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# origins = [
#     "http://localhost:4200",
#     "http://192.168.0.125:8000",
#     "http://localhost",
#     "http://192.168.0.180:6000"  # Example IP
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MONGO_URI = "mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/"
# DB_NAME = "starlorDB"

# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db["prediction"]

# def sanitize_data(data):
#     # Sanitize NaN and Inf AFTER fetching data
#     for record in data:
#         for key, value in record.items():
#             if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
#                 record[key] = None
#     return data

# @app.get("/prediction")
# async def get_prediction(
#     start_date: Optional[str] = Query(None),
#     end_date: Optional[str] = Query(None),
#     start_time: Optional[str] = Query(None),
#     end_time: Optional[str] = Query(None),
#     price_min: Optional[float] = Query(None),
#     price_max: Optional[float] = Query(None),
#     booking_type: Optional[str] = Query(None),
#     country_id: Optional[str] = Query(None),
#     city_id: Optional[str] = Query(None),
# ):
#     try:
#         query = {}

#         if booking_type:
#             query["booking_type"] = booking_type

#         #  Add filter for country_id
#         if country_id is not None:
#             query["countryId"] = country_id
 
#         # Add filter for city_id
#         if city_id is not None:
#             query["cityId"] = city_id    

#         # Combined Price Range Filter
#         price_filter = {}
#         if price_min is not None:
#             price_filter["$gte"] = price_min
#         if price_max is not None:
#             price_filter["$lte"] = price_max
#         if price_filter:  # Add only if not empty
#             query["$and"] = [{"price_min": price_filter}, {"price_max": price_filter}] # Corrected price query

#         # Date and Time Filter
#         if start_date and end_date:
#             start = datetime.strptime(f"{start_date} {start_time or '00:00'}", "%d-%m-%Y %H:%M")
#             end = datetime.strptime(f"{end_date} {end_time or '23:59'}", "%d-%m-%Y %H:%M")
#             query["date_stamp"] = {
#                 "$gte": start,
#                 "$lte": end
#             }

#         prediction = list(collection.find(query, {
#             "_id": 0,
#             "latitude": 1,
#             "longitude": 1,
#             "booking_type": 1,
#             "date_stamp": 1,
#             "price_min": 1,
#             "price_max": 1,
#             "cityId" : 1,
#             "countryId" : 1
#         }))

#         return sanitize_data(prediction)

#     except Exception as e:
#         print(f"Error fetching predictions: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)









