from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Settings


settings = Settings()
SQLALCHEMY_DATABASE_URL = settings.database_url

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
