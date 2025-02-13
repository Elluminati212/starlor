from pymongo import MongoClient
from pydantic_settings import BaseSettings

class Settings(MongoClient):
    db_url: str = "mongodb+srv://vpatel179:LqpC2zz4rO7pmtkR@cluster0.ejksq.mongodb.net/"
    db_name: str = "WriteDB"

    class Config:
        env_file = ".env"

settings = Settings()




# def get_mongo_client():
#     try:
#         # Correctly create a MongoDB client
#         client = MongoClient(settings.db_url)
#         return client
#     except Exception as e:
#         raise RuntimeError(f"Failed to connect to MongoDB: {e}")


# def get_prediction():
#     try:
#         # Connect to MongoDB
#         client = get_mongo_client()
#         db = client[settings.db_name]  # Access the database using db_name
#         collection = db['prediction']  # Access the collection named 'prediction'

#         # Fetch all predictions from the database
#         predictions = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB `_id` field
#         print(predictions)  # Print the fetched predictions
#         return predictions
#     except Exception as e:
#         return {"error": str(e)}


# # Call the function to test
# if __name__ == "__main__":
#     all_predictions = get_prediction()
#     print("All Predictions:", all_predictions)
