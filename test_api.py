import requests
import json

BASE_URL = "http://localhost:8000"

def test_user_registration():
    print("\n=== Testing User Registration ===")
    url = f"{BASE_URL}/users/user/register"
    data = {
        "username": "test_user",
        "email": "test.user@example.com",
        "password": "testpass123"
    }
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_user_login():
    print("\n=== Testing User Login ===")
    url = f"{BASE_URL}/users/user/login"
    # Try with correct credentials
    data = {
        "username": "john_doe",
        "password": "password123"
    }
    response = requests.post(url, json=data)
    print("Testing correct credentials:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Try with incorrect credentials
    data = {
        "username": "john_doe",
        "password": "wrongpassword"
    }
    response = requests.post(url, json=data)
    print("\nTesting incorrect credentials:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_ocr_prediction():
    print("\n=== Testing OCR Prediction ===")
    url = f"{BASE_URL}/predict"
    
    # You would need to provide an actual image file here
    # For demonstration, we'll just show the endpoint
    print("To test OCR prediction, send a POST request to /predict with an image file")
    print("Example using curl:")
    print('curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.png"')

if __name__ == "__main__":
    print("Starting API Tests...")
    test_user_registration()
    test_user_login()
    test_ocr_prediction() 