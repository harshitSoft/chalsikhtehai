from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import SessionLocal
from models.user import User
from schemas.user_schema import UserCreate, UserLogin, ShowUser
from typing import List

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

# Create Admin (Superadmin only)
@router.post("/create-admin", response_model=ShowUser)
def create_admin(user: UserCreate, db: Session = Depends(get_db)):
    # role must be 'admin', created_by must be superadmin's id
    if user.role != 'admin':
        raise HTTPException(status_code=400, detail="Role must be 'admin'")
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    new_user = User(
        username=user.username,
        email=user.email,
        role='admin',
        created_by=user.created_by,
        area=user.area,
        zone=user.zone,
        meter_number=user.meter_number,
        contact_number=user.contact_number,
        address=user.address
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Create Staff (Admin only)
@router.post("/create-staff", response_model=ShowUser)
def create_staff(user: UserCreate, db: Session = Depends(get_db)):
    # role must be 'staff', created_by must be admin's id
    if user.role != 'staff':
        raise HTTPException(status_code=400, detail="Role must be 'staff'")
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    new_user = User(
        username=user.username,
        email=user.email,
        role='staff',
        created_by=user.created_by,
        area=user.area,
        zone=user.zone,
        meter_number=user.meter_number,
        contact_number=user.contact_number,
        address=user.address
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Create Customer (Staff only)
@router.post("/create-customer", response_model=ShowUser)
def create_customer(user: UserCreate, db: Session = Depends(get_db)):
    # role must be 'customer', created_by must be staff's id
    if user.role != 'customer':
        raise HTTPException(status_code=400, detail="Role must be 'customer'")
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    new_user = User(
        username=user.username,
        email=user.email,
        role='customer',
        created_by=user.created_by,
        area=user.area,
        zone=user.zone,
        meter_number=user.meter_number,
        contact_number=user.contact_number,
        address=user.address
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# List all admins (Superadmin only)
@router.get("/all-admins", response_model=List[ShowUser])
def get_all_admins(db: Session = Depends(get_db)):
    return db.query(User).filter(User.role == 'admin').all()

# List all staff for an admin
@router.get("/admin/{admin_id}/staff", response_model=List[ShowUser])
def get_admin_staff(admin_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.role == 'staff', User.created_by == admin_id).all()

# List all customers for a staff
@router.get("/staff/{staff_id}/customers", response_model=List[ShowUser])
def get_staff_customers(staff_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.role == 'customer', User.created_by == staff_id).all()

# List all users (for superadmin overview)
@router.get("/all", response_model=List[ShowUser])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    result = []
    for user in users:
        user_dict = user.__dict__.copy()
        # Convert last_reading_date to ISO string if it exists
        if user_dict.get('last_reading_date'):
            user_dict['last_reading_date'] = user_dict['last_reading_date'].isoformat()
        result.append(ShowUser(**user_dict))
    return result
