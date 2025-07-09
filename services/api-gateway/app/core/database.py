from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from typing import AsyncGenerator
from .config import get_settings

settings = get_settings()

# Handle different database URLs for testing
if settings.database_url.startswith("sqlite"):
    # For SQLite (testing)
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def get_database():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
else:
    # For PostgreSQL (production)
    async_database_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(async_database_url, echo=settings.debug)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async def get_database() -> AsyncGenerator[AsyncSession, None]:
        async with AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

# Create declarative base
Base = declarative_base()
