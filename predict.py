import cv2
import numpy as np
import string
import tensorflow as tf
from fastapi import UploadFile
from fastapi.responses import JSONResponse

# Constants
alphabet = string.digits + '.'
blank_index = len(alphabet)  # This is the index for the blank character (for CTC loss)
MODEL_PATH = "meeter_rec_float16.tflite"  # Path to your TFLite model

def prepare_input(image_bytes):
    """
    Prepares the input image for the model.
    - Converts to grayscale.
    - Resizes to the expected input size (200x31).
    - Normalizes the pixel values to [0, 1] range.
    """
    # Convert the image bytes to numpy array
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    input_data = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Decode image from bytes
    input_data = cv2.resize(input_data, (200, 31))  # Resize to match model input
    input_data = input_data[np.newaxis]  # Add batch dimension
    input_data = np.expand_dims(input_data, 3)  # Add channel dimension (1 channel for grayscale)
    input_data = input_data.astype('float32') / 255.0  # Normalize pixel values
    return input_data

def predict(image_bytes, model_path=MODEL_PATH):
    """
    Predicts the text from the image using the provided model.
    - Loads the model using TensorFlow Lite interpreter.
    - Prepares the input data and feeds it into the model.
    - Retrieves and returns the model output.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_data = prepare_input(image_bytes)

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor and invoke the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the output tensor, which contains the predicted class indices
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def decode_output(output):
    """
    Decodes the output tensor from the model into a human-readable string.
    - Maps the output indices to characters in the alphabet.
    - Removes blank and invalid indices.
    """
    # Decode the output into characters, filtering out blank indices
    text = "".join(alphabet[index] for index in output[0] if index not in [blank_index, -1])
    
    # Try to convert the text into a valid float if it contains a decimal point
    try:
        # If text contains a decimal point, convert it to a float
        if '.' in text:
            text = str(float(text))
    except ValueError:
        pass  # If conversion fails, just keep the text as is
    
    return text

async def predict_image(file: UploadFile):
    """
    Endpoint to handle image prediction from the uploaded image.
    - Uses FastAPI to accept an image from the frontend and process it.
    - Returns the extracted text from the image.
    """
    try:
        # Read the image bytes from the uploaded file
        image_bytes = await file.read()

        # Perform prediction
        output = predict(image_bytes)

        # Decode the model's output to get the text
        result = decode_output(output)

        # Return the extracted text
        return {"result": result}

    except Exception as e:
        # Handle errors (e.g., invalid image format, model issues)
        return JSONResponse(status_code=500, content={"error": str(e)})




