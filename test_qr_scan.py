import requests
import json

BASE_URL = "http://localhost:8000"

def test_qr_scan():
    print("\n=== Testing QR Code Scanning ===")
    url = f"{BASE_URL}/scan-qr"
    
    # Test with existing user
    print("\nTesting with existing user (john_doe):")
    data = {
        "username": "john_doe"
    }
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test with non-existing user
    print("\nTesting with non-existing user:")
    data = {
        "username": "nonexistent_user"
    }
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("Starting QR Code Scan Tests...")
    test_qr_scan() 