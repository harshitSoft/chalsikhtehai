from typing import Optional
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    role: str = 'customer'
    created_by: Optional[int] = None
    area: Optional[str] = None
    zone: Optional[str] = None
    meter_number: Optional[str] = None
    contact_number: Optional[str] = None
    address: Optional[str] = None

class UserLogin(BaseModel):
    username: str

class ShowUser(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    created_by: Optional[int] = None
    area: Optional[str] = None
    zone: Optional[str] = None
    meter_number: Optional[str] = None
    contact_number: Optional[str] = None
    address: Optional[str] = None
    total_amount: Optional[float] = 0
    last_reading_date: Optional[str] = None
    late_fees: float = 0
    bill_count: Optional[int] = 0

    class Config:
        orm_mode = True
