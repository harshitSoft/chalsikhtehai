from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL connection URL (update with your credentials)
DATABASE_URL = "mysql+pymysql://root:Ram1234@localhost:3306/meterusernew"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Helps to avoid stale connections
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for declarative models
Base = declarative_base()

# Dependency for FastAPI routes to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
