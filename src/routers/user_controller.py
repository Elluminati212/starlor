from fastapi import APIRouter, Depends
from src.middlewares.authentication import get_current_user
from src.models import user
from src.models.user import User

router = APIRouter() 

@router.get("/")
async def get_users(current_user: User = Depends(get_current_user)):
    # Logic to fetch all users from MongoDB goes here
    # Use MongoDB driver to query the collection
    # Return a list of User objects
    return {"users": user}

@router.get("/{user_id}")
async def get_user(user_id: str, current_user: User = Depends(get_current_user)):
    # Logic to fetch a single user by ID from MongoDB goes here
    # Use MongoDB driver to query the collection
    # Return a User object
    return {"user": user}

@router.post("/")
async def create_user(user: User):
    # Logic to create a new user in MongoDB goes here
    # Use MongoDB driver to insert the user data
    # Return the created user
    return {"user": User}

@router.put("/{user_id}")
async def update_user(user_id: str, user: User):
    # Logic to update an existing user in MongoDB goes here
    # Use MongoDB driver to update the user data
    # Return the updated user
    return {"user": User}

@router.delete("/{user_id}")
async def delete_user(user_id: str):
    # Logic to delete a user from MongoDB goes here
    # Use MongoDB driver to delete the user data
    # Return a success message
    return {"message": "User deleted successfully"}

# @router.get("/users")
# async def read_users(current_user: User = Depends(get_current_user)):
#     return [{"username": "user1"}, {"username": "user2"}]



# filepath: /home/vasu/app/src/routers/user_controller.py
from fastapi import APIRouter, HTTPException, Depends
from pymongo import MongoClient
from src.config.config import Settings
from src.models.user import User

router = APIRouter()

def get_mongo_client():
    try:
        client = MongoClient(Settings().db_url)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MongoDB: {e}")

@router.post("/users")
async def create_user(user: User):
    try:
        client = get_mongo_client()
        db = client[Settings().db_name]
        collection = db['users']
        collection.insert_one(user.dict())
        return {"status": "success", "data": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}")
async def get_user(user_id: str):
    try:
        client = get_mongo_client()
        db = client[Settings().db_name]
        collection = db['users']
        user_data = collection.find_one({"id": user_id})
        if user_data:
            return {"status": "success", "data": user_data}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))