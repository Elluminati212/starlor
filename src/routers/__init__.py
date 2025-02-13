from fastapi import APIRouter

user_rputer = APIRouter(prefix="/user", tags=["user"])  

#from .user_controller import router as user_router
from .import user_controller
__all__ = ["user_router"]