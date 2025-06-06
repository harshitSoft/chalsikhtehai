from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import string
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Allow cross-origin requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify specific origins like ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Constants
alphabet = string.digits + '.'  # Allowed characters for the OCR
blank_index = len(alphabet)  # Blank character index used in CTC loss
MODEL_PATH = "model_float16.tflite"  # Path to your trained TFLite model

# Function to prepare input for the model
def prepare_input(image_bytes):
    """
    Preprocesses the image to fit the model's input requirements:
    - Converts to grayscale.
    - Resizes to (200x31) pixels.
    - Normalizes pixel values to the range [0, 1].
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)  # Convert byte data to numpy array
    input_data = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Decode to grayscale image
    input_data = cv2.resize(input_data, (200, 31))  # Resize to match model input size
    input_data = input_data[np.newaxis, :, :, np.newaxis]  # Add batch and channel dimensions
    input_data = input_data.astype('float32') / 255.0  # Normalize pixel values
    return input_data

# Function to run the prediction using the TFLite model
def predict(image_bytes):
    """
    Makes a prediction using the provided image bytes.
    Loads the TFLite model, prepares the input data, and gets the output.
    """
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()  # Allocate tensors for the interpreter

    input_data = prepare_input(image_bytes)  # Prepare input data for prediction

    input_details = interpreter.get_input_details()  # Get model input details
    output_details = interpreter.get_output_details()  # Get model output details

    # Set input tensor and invoke model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor and return it
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Decode the output tensor into human-readable text
def decode_output(output):
    """
    Decodes the output tensor into a string using the allowed characters in the alphabet.
    Converts the string to a float if it contains a decimal point.
    """
    text = "".join(alphabet[index] for index in output[0] if index not in [blank_index, -1])  # Decode
    try:
        if '.' in text:
            text = str(float(text))  # Convert to float if it contains a decimal point
    except ValueError:
        pass  # If the conversion fails, keep the original text
    return text

# FastAPI endpoint to handle image upload and prediction
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    API endpoint to accept an image file upload from the frontend and return the extracted text.
    """
    try:
        image_bytes = await file.read()  # Read image data as bytes from the frontend
        output = predict(image_bytes)  # Run prediction on the image
        result = decode_output(output)  # Decode the predicted output
        return {"result": result}  # Return the extracted text to the frontend
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})  # Handle errors gracefully

# Run the server using uvicorn (this line can be used in a terminal)
# uvicorn server:app --host 0.0.0.0 --port 8000
