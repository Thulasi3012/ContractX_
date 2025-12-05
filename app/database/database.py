from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
from app.config import config

# Direct database credentials
# DB_USER = "Thulasi"
# DB_PASSWORD = "Thulasi@30125"
# DB_HOST = "localhost"
# DB_PORT = 5432
# DB_NAME = "Contract"

# Properly escape password
escaped_password = urllib.parse.quote_plus(config.Config.DB_PASSWORD)

# Build database URL
# DATABASE_URL = f"postgresql://{DB_USER}:{escaped_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(
    config.Config().DATABASE_URL,  # instantiate config
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False
)

# SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("[DB] Database initialized successfully")


def drop_db():
    """Drop all tables - use with caution!"""
    Base.metadata.drop_all(bind=engine)
    print("[DB] All     tables dropped successfully")
