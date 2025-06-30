from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    zone: str = None
    meter_number: str = None
    contact_number: str = None
    address: str = None

class UserLogin(BaseModel):
    username: str

class ShowUser(BaseModel):
    id: int
    username: str
    email: EmailStr
    zone: str = None
    meter_number: str = None
    contact_number: str = None
    address: str = None

    class Config:
        orm_mode = True
