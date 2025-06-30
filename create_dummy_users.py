from database import SessionLocal
from models.user import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dummy users data
dummy_users = [
    {
        "username": "john_doe",
        "email": "john.doe@example.com",
        "password": "password123"
    },
    {
        "username": "jane_smith",
        "email": "jane.smith@example.com",
        "password": "securepass456"
    },
    {
        "username": "admin_user",
        "email": "admin@example.com",
        "password": "adminpass789"
    }
]

def create_dummy_users():
    db = SessionLocal()
    try:
        for user_data in dummy_users:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.username == user_data["username"]) | 
                (User.email == user_data["email"])
            ).first()
            
            if not existing_user:
                hashed_password = pwd_context.hash(user_data["password"])
                new_user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    hashed_password=hashed_password
                )
                db.add(new_user)
                print(f"Created user: {user_data['username']}")
            else:
                print(f"User already exists: {user_data['username']}")
        
        db.commit()
        print("\nDummy users created successfully!")
    except Exception as e:
        print(f"Error creating dummy users: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_dummy_users() 