from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routes import user_routes  # User-related routes
import tensorflow as tf
import numpy as np
import cv2
import string
from sqlalchemy.orm import Session
from database import SessionLocal
from models.user import User
from pydantic import BaseModel

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
    allow_origins=["*"],  # You can restrict to ["http://localhost:3000"] for frontend
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
@app.post("/scan-qr", response_model=dict)
async def scan_qr(qr_data: QRCodeData, db: Session = Depends(get_db)):
    """
    Endpoint to handle QR code scanning and return user details
    """
    try:
        # Query the database for the user
        user = db.query(User).filter(User.username == qr_data.username).first()
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User with username '{qr_data.username}' not found"
            )
        
        # Return user details
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "message": "User found successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
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
        image_bytes = await file.read()
        output = predict(image_bytes)
        result = decode_output(output)
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================= #
# === Run Command (terminal) === #
# ============================= #
# uvicorn server:app --host 0.0.0.0 --port 8000
