from sqlalchemy import create_engine, text
from database import Base, engine
from models.user import User

def update_database():
    try:
        # Drop existing tables
        Base.metadata.drop_all(bind=engine)
        
        # Create tables with new schema
        Base.metadata.create_all(bind=engine)
        
        print("Database schema updated successfully!")
        
    except Exception as e:
        print(f"Error updating database: {str(e)}")

if __name__ == "__main__":
    update_database() 