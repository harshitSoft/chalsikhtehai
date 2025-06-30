from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import SessionLocal
from models.user import User
from schemas.user_schema import UserCreate, UserLogin, ShowUser

router = APIRouter(
    prefix="/user",
    tags=["Users"]
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Register new user
@router.post("/register", response_model=ShowUser)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    new_user = User(
        username=user.username,
        email=user.email,
        zone=user.zone,
        meter_number=user.meter_number,
        contact_number=user.contact_number,
        address=user.address
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Login existing user
@router.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    # Remove password check from login_user or disable login if not needed
    return {"message": "Login successful", "user_id": user.username}
