from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routes import user_routes  # User-related routes
import tensorflow as tf
import numpy as np
import cv2
import string
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models.user import User, Base
from pydantic import BaseModel
from sqlalchemy import func
import logging
from typing import Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# =================== #
# === App Config ==== #
# =================== #
app = FastAPI(
    title="OCR + User Auth API",
    description="FastAPI backend for OCR digit recognition with user registration and login",
    version="1.0.0"
)

# ============================= #
# === CORS Middleware Setup === #
# ============================= #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================= #
# === Route Registrations  ==== #
# ============================= #
app.include_router(user_routes.router, prefix="/users", tags=["Users"])

# ============================= #
# === QR Code Models ========= #
# ============================= #
class QRCodeData(BaseModel):
    username: str

class MeterReadingData(BaseModel):
    username: str
    reading: float
    staff_id: int

class UpdateUnitsData(BaseModel):
    username: str
    current_unit: float
    unit_consumed: float
    staff_id: int

# ============================= #
# === Database Dependency ===== #
# ============================= #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================= #
# === QR Code Endpoint ======= #
# ============================= #
@app.post("/scan-qr")
async def scan_qr(qr_data: QRCodeData, db: Session = Depends(get_db)):
    """
    Endpoint to handle QR code scanning and return user details
    """
    try:
        logger.info(f"Scanning QR code for username: {qr_data.username}")
        
        # Query the database for the user
        user = db.query(User).filter(User.username == qr_data.username).first()
        
        if not user:
            logger.warning(f"User not found: {qr_data.username}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with username '{qr_data.username}' not found"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Return user information
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active,
            "current_unit": user.current_unit or 0,
            "last_unit": user.last_unit or 0,
            "unit_consumed": user.unit_consumed or 0,
            "total_amount": user.total_amount or 0,
            "last_reading_date": user.last_reading_date.isoformat() if user.last_reading_date else None,
            # Add any other fields needed by frontend
            "zone": user.zone,
            "meter_number": user.meter_number,
            "contact_number": user.contact_number,
            "address": user.address,
            "late_fees": user.late_fees,
            "role": user.role,
            "area": user.area,
            "bill_count": user.bill_count,
            "message": "User found successfully",
            "status": "success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in scan_qr: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing QR code: {str(e)}"
        )

# ============================= #
# === OCR Model Constants   ==== #
# ============================= #
alphabet = string.digits + '.'
blank_index = len(alphabet)
MODEL_PATH = "model_float16.tflite"

# ============================= #
# === OCR Helper Functions  ==== #
# ============================= #

def prepare_input(image_bytes):
    """
    Converts input image bytes to normalized grayscale tensor with shape (1, 31, 200, 1)
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (200, 31))  # Resize for model
    image = image[np.newaxis, :, :, np.newaxis].astype('float32') / 255.0
    return image

def predict(image_bytes):
    """
    Runs inference on the given image bytes using TFLite model.
    """
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_data = prepare_input(image_bytes)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def decode_output(output):
    """
    Decodes TFLite model output into a readable string.
    """
    text = "".join(alphabet[index] for index in output[0] if index not in [blank_index, -1])
    try:
        if '.' in text:
            return str(float(text))  # Convert to float string if possible
    except ValueError:
        pass
    return text

# ============================= #
# === Prediction Endpoint   ==== #
# ============================= #

@app.post("/predict", summary="Predict digits from image", description="Uploads an image and returns predicted text")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be an image"}
            )

        image_bytes = await file.read()
        if len(image_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty file received"}
            )

        output = predict(image_bytes)
        result = decode_output(output)
        
        if not result or not any(c.isdigit() for c in result):
            return JSONResponse(
                status_code=400,
                content={"error": "No valid number detected in image"}
            )

        return {"result": result}
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")  # Add logging
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

# ============================= #
# === Update Meter Reading Endpoint === #
# ============================= #
@app.post("/update-meter-reading")
async def update_meter_reading(reading_data: MeterReadingData, db: Session = Depends(get_db)):
    """
    Endpoint to update meter readings and calculate billing
    """
    try:
        logger.info(f"Updating meter reading for user: {reading_data.username}")
        
        # Get user
        user = db.query(User).filter(User.username == reading_data.username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with username '{reading_data.username}' not found"
            )
        
        # Store the current unit as the last unit (this will become the previous reading)
        previous_current_unit = user.current_unit or 0
        
        # Log the values for debugging
        logger.info(f"Before update - last_unit: {user.last_unit}, current_unit: {user.current_unit}, new_reading: {reading_data.reading}")
        
        # Update readings with proper swap
        user.last_unit = previous_current_unit  # Move current to last
        user.current_unit = reading_data.reading  # Set new reading as current
        
        # Calculate consumption using the previous current unit (now stored as last_unit)
        user.unit_consumed = user.current_unit - user.last_unit
        
        # Validate the calculation
        expected_consumption = reading_data.reading - previous_current_unit
        logger.info(f"Validation - expected_consumption: {expected_consumption}, calculated_consumption: {user.unit_consumed}")
        
        # Ensure the calculation is correct
        if abs(user.unit_consumed - expected_consumption) > 0.01:
            logger.warning(f"Consumption calculation mismatch! Expected: {expected_consumption}, Got: {user.unit_consumed}")
            user.unit_consumed = expected_consumption
        
        # Log the values after update
        logger.info(f"After update - last_unit: {user.last_unit}, current_unit: {user.current_unit}, unit_consumed: {user.unit_consumed}")
        
        user.last_reading_date = datetime.utcnow()
        
        # Calculate billing (rate: ₹12.5 per unit + ₹75 fixed charge)
        RATE_PER_UNIT = 12.5
        FIXED_CHARGE = 75
        base_amount = (user.unit_consumed * RATE_PER_UNIT) + FIXED_CHARGE
        gst = base_amount * 0.18
        user.total_amount = base_amount + gst
        
        # Increment bill_count for staff
        staff = db.query(User).filter(User.id == reading_data.staff_id, User.role == 'staff').first()
        if staff:
            staff.bill_count = (staff.bill_count or 0) + 1
        
        # Commit changes to database
        db.commit()
        
        # Refresh user object to get the latest values from database
        db.refresh(user)
        
        # Log final values being returned
        logger.info(f"Final values being returned - last_unit: {user.last_unit}, current_unit: {user.current_unit}, unit_consumed: {user.unit_consumed}")
        
        # Return the updated values from database
        return {
            "username": user.username,
            "current_unit": user.current_unit,
            "last_unit": user.last_unit,
            "unit_consumed": user.unit_consumed,
            "total_amount": user.total_amount,
            "base_amount": base_amount,
            "gst_amount": gst,
            "rate_per_unit": RATE_PER_UNIT,
            "fixed_charge": FIXED_CHARGE,
            "last_reading_date": user.last_reading_date.isoformat(),
            "message": "Meter reading updated successfully",
            "status": "success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in update_meter_reading: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating meter reading: {str(e)}"
        )

# ============================= #
# === Update Units Endpoint === #
# ============================= #
@app.post("/update-units")
async def update_units(units_data: UpdateUnitsData, db: Session = Depends(get_db)):
    """
    Endpoint to update units without affecting invoice generation
    """
    try:
        logger.info(f"Updating units for user: {units_data.username}")
        
        # Get user
        user = db.query(User).filter(User.username == units_data.username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with username '{units_data.username}' not found"
            )
        
        # Log the values before update
        logger.info(f"Before update - last_unit: {user.last_unit}, current_unit: {user.current_unit}, unit_consumed: {user.unit_consumed}")
        
        # Update the units
        user.current_unit = units_data.current_unit
        user.unit_consumed = units_data.unit_consumed
        
        # Recalculate total amount based on updated consumption
        RATE_PER_UNIT = 12.5
        FIXED_CHARGE = 75
        base_amount = (user.unit_consumed * RATE_PER_UNIT) + FIXED_CHARGE
        gst = base_amount * 0.18
        user.total_amount = base_amount + gst
        
        # Update last reading date
        user.last_reading_date = datetime.utcnow()
        
        # Commit changes to database
        db.commit()
        
        # Refresh user object to get the latest values from database
        db.refresh(user)
        
        # Log final values being returned
        logger.info(f"After update - last_unit: {user.last_unit}, current_unit: {user.current_unit}, unit_consumed: {user.unit_consumed}")
        
        return {
            "username": user.username,
            "current_unit": user.current_unit,
            "last_unit": user.last_unit,
            "unit_consumed": user.unit_consumed,
            "total_amount": user.total_amount,
            "base_amount": base_amount,
            "gst_amount": gst,
            "rate_per_unit": RATE_PER_UNIT,
            "fixed_charge": FIXED_CHARGE,
            "last_reading_date": user.last_reading_date.isoformat(),
            "message": "Units updated successfully",
            "status": "success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in update_units: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating units: {str(e)}"
        )

# ============================= #
# === Run Command (terminal) === #
# ============================= #
# python -m uvicorn server:app --host 0.0.0.0 --port 8000
