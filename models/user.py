from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # New columns for meter readings and billing
    current_unit = Column(Float, nullable=True)
    last_unit = Column(Float, nullable=True)
    unit_consumed = Column(Float, nullable=True)
    total_amount = Column(Float, nullable=True)
    last_reading_date = Column(DateTime(timezone=True), nullable=True)
