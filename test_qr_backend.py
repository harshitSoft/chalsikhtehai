from database import SessionLocal
from models.user import User

def simulate_qr_scan(dummy_username):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == dummy_username).first()
        if user:
            print("User found:")
            print(f"ID: {user.id}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
        else:
            print(f"User with username '{dummy_username}' not found.")
    finally:
        db.close()

if __name__ == "__main__":
    # Simulate QR scan with a dummy username
    dummy_username = "john_doe"  # Replace with any username you want to test
    simulate_qr_scan(dummy_username) 