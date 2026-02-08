"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging

from models import Base
from config import config

logger = logging.getLogger(__name__)


class Database:
    """Database connection manager"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        
    def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                config.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(session_factory)
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Provide a transactional scope for database operations
        
        Usage:
            with db.get_session() as session:
                user = session.query(User).first()
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connection"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")


# Global database instance
db = Database()


def init_db():
    """Initialize database - called at application startup"""
    db.initialize()


def get_db_session():
    """Get a database session (for dependency injection)"""
    return db.get_session()