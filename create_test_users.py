from sqlalchemy.orm import Session
from database import SessionLocal
from models.user import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_test_users():
    db = SessionLocal()
    try:
        # Create test users
        test_users = [
            {
                "username": "john_doe",
                "email": "john.doe@example.com",
                "password": pwd_context.hash("password123"),
                "is_active": True
            },
            {
                "username": "jane_smith",
                "email": "jane.smith@example.com",
                "password": pwd_context.hash("password123"),
                "is_active": True
            }
        ]

        for user_data in test_users:
            # Check if user already exists
            existing_user = db.query(User).filter(User.username == user_data["username"]).first()
            if not existing_user:
                user = User(**user_data)
                db.add(user)
                print(f"Created user: {user_data['username']}")
            else:
                print(f"User already exists: {user_data['username']}")

        db.commit()
        print("\nTest users created successfully!")
        
    except Exception as e:
        print(f"Error creating test users: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_users() 