"""
Database models for Face Recognition Attendance System
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model for storing employee/student information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    department = Column(String(100), nullable=True)
    role = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    attendance_records = relationship('AttendanceRecord', back_populates='user', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"


class AttendanceRecord(Base):
    """Attendance records for tracking user presence"""
    __tablename__ = 'attendance_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    camera_type = Column(String(50), nullable=False)  # entrance, exit
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    confidence = Column(Float, nullable=True)  # Recognition confidence score
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='attendance_records')
    
    # Indexes for faster queries
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_camera_type', 'camera_type'),
    )
    
    def __repr__(self):
        return f"<AttendanceRecord(user_id={self.user_id}, camera='{self.camera_type}', time='{self.timestamp}')>"


class TrainingSession(Base):
    """Track training sessions for the face recognition model"""
    __tablename__ = 'training_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    frames_count = Column(Integer, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    
    def __repr__(self):
        return f"<TrainingSession(id={self.id}, user_id={self.user_id}, frames={self.frames_count})>"


class SystemLog(Base):
    """System logs for monitoring and debugging"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    component = Column(String(50), nullable=False)  # producer, attendance_worker, training_worker
    message = Column(String(500), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_timestamp_level', 'timestamp', 'level'),
    )
    
    def __repr__(self):
        return f"<SystemLog(level='{self.level}', component='{self.component}', time='{self.timestamp}')>"