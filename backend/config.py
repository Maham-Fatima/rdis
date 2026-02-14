"""
Configuration management for the Face Recognition Attendance System
"""
import os
from typing import Dict


class Config:
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        'DATABASE_URL', 
        'mysql+pymysql://root:admin@localhost:3306/attendance_db'
        # For SQLite use: 'sqlite:///attendance.db'
    )
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    REDIS_MODEL_KEY: str = 'face_model_cache'
    
    # RabbitMQ Configuration
    RABBITMQ_HOST: str = os.getenv('RABBITMQ_HOST', 'localhost')
    RABBITMQ_PORT: int = int(os.getenv('RABBITMQ_PORT', '5672'))
    RABBITMQ_USER: str = os.getenv('RABBITMQ_USER', 'guest')
    RABBITMQ_PASS: str = os.getenv('RABBITMQ_PASS', 'guest')
    
    # Queue Names
    ATTENDANCE_QUEUE: str = 'attendance'
    TRAINING_QUEUE: str = 'train'
    
    # Camera Configuration
    CAMERAS: Dict[str, str] = {
        "entrance": os.getenv('CAMERA_ENTRANCE', "http://192.168.100.52:8080/video"),
        "exit": os.getenv('CAMERA_EXIT', "http://192.168.100.52:8080/video")
    }
    
    # Face Detection Configuration
    FACE_CASCADE_PATH: str = 'haarcascade_frontalface_default.xml'
    DETECTION_SCALE_FACTOR: float = 1.3
    DETECTION_MIN_NEIGHBORS: int = 5
    
    # Model Configuration
    MODEL_PATH: str = 'trainer.yml'
    CONFIDENCE_THRESHOLD: float = 0.7  
    BATCH_SIZE: int = 50
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance
    PREFETCH_COUNT: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5  
    
    # Database Sync
    DB_SYNC_INTERVAL: int = int(os.getenv('DB_SYNC_INTERVAL', '10')) 
    REDIS_CLEANUP_DAYS: int = 7  # Keep Redis data for 7 days after sync

config = Config()