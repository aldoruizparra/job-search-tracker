"""Database setup and session management.

Kept separate from models.py so the engine configuration lives in one place
and tests can swap in a different database without touching the models.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///jobs.db")

# check_same_thread is a SQLite-only quirk: FastAPI serves requests on a
# threadpool, and SQLite refuses cross-thread connections by default.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


def init_db():
    """Create tables if they do not exist. Safe to call repeatedly."""
    Base.metadata.create_all(engine)


def get_session():
    """FastAPI dependency. Yields a session and always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
