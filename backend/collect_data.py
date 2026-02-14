"""
Training Module - Trains face recognition model from Redis data
"""
import redis
import cv2
import pickle
import numpy as np
from datetime import datetime
import os
import logging

from config import config
from database import db, init_db
from models import TrainingSession

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles face recognition model training"""
    
    def __init__(self):
        self.redis_client = None
        self.recognizer = None
        self._setup()
    
    def _setup(self):
        """Initialize Redis and recognizer"""
        try:
            # Initialize database
            init_db()
            
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Redis connected")
            
            # Initialize recognizer
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Load existing model if available
            self._load_existing_model()
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise
    
    def _load_existing_model(self):
        """Load existing model from cache or disk"""
        try:
            # Try Redis cache first
            model_bytes = self.redis_client.get(config.REDIS_MODEL_KEY)
            
            if model_bytes:
                tmp_path = "trainer_tmp.yml"
                with open(tmp_path, "wb") as f:
                    f.write(model_bytes)
                self.recognizer.read(tmp_path)
                os.remove(tmp_path)
                logger.info("Existing model loaded from Redis cache")
                return True
            
            # Try disk
            if os.path.exists(config.MODEL_PATH):
                self.recognizer.read(config.MODEL_PATH)
                logger.info("Existing model loaded from disk")
                return True
            
            logger.info("No existing model found - will train from scratch")
            return False
            
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            return False
    
    def _save_model(self):
        """Save trained model to disk and Redis cache"""
        try:
            # Save to disk
            self.recognizer.write(config.MODEL_PATH)
            logger.info(f"Model saved to {config.MODEL_PATH}")
            
            # Cache in Redis
            with open(config.MODEL_PATH, "rb") as f:
                self.redis_client.set(config.REDIS_MODEL_KEY, f.read())
            logger.info("Model cached in Redis")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def train_from_redis(self, batch_size: int = None, date: str = None, user_id: int = None):
        """
        Fetch frames from Redis and train the model
        
        Args:
            batch_size: Number of frames to process in each training iteration
            date: Date string (YYYY-MM-DD) to train from. If None, uses today
            user_id: Specific user ID to train. If None, trains all users
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Find training batches in Redis
            pattern = f"train_batch:{date}:*" if user_id is None else f"train_batch:{date}:{user_id}"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                logger.warning(f"No training data found for pattern: {pattern}")
                return
            
            logger.info(f"Found {len(keys)} training batch(es) to process")
            
            total_frames = 0
            
            for key in keys:
                logger.info(f"Processing batch: {key.decode()}")
                
                # Extract user_id from key
                key_parts = key.decode().split(':')
                batch_user_id = int(key_parts[2]) if len(key_parts) > 2 else 1
                
                # Update training session status
                session_id = self._update_training_session(batch_user_id, 'processing')
                
                faces = []
                labels = []
                
                # Fetch all frames from this batch
                while True:
                    data = self.redis_client.lpop(key)
                    if not data:
                        break
                    
                    try:
                        payload = pickle.loads(data)
                        face = payload.get('face')
                        face_user_id = payload.get('user_id', batch_user_id)
                        
                        if face is not None:
                            # Resize face to consistent size
                            if face.shape != (200, 200):
                                face = cv2.resize(face, (200, 200))
                            
                            faces.append(face)
                            labels.append(face_user_id)
                    
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        continue
                
                # Train in batches
                if faces:
                    logger.info(f"Training on {len(faces)} faces for user {batch_user_id}")
                    
                    for i in range(0, len(faces), batch_size):
                        batch_faces = faces[i:i+batch_size]
                        batch_labels = labels[i:i+batch_size]
                        
                        # Update model
                        self.recognizer.update(batch_faces, np.array(batch_labels))
                        logger.info(f"Trained batch {i//batch_size + 1}/{(len(faces)-1)//batch_size + 1}")
                    
                    total_frames += len(faces)
                    
                    # Save model after each user's data
                    self._save_model()
                    
                    # Update training session
                    self._update_training_session(batch_user_id, 'completed', len(faces))
                else:
                    logger.warning(f"No valid faces found in batch {key.decode()}")
                    self._update_training_session(batch_user_id, 'failed')
            
            logger.info(f"Training complete! Processed {total_frames} total frames")
            logger.info("Model updated and cached successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _update_training_session(self, user_id: int, status: str, frames_count: int = None):
        """Update training session status in database"""
        try:
            with db.get_session() as session:
                # Find the most recent pending/processing session for this user
                training_session = session.query(TrainingSession).filter_by(
                    user_id=user_id,
                    status =['pending', 'processing']
                ).order_by(TrainingSession.started_at.desc()).first()
                
                if training_session:
                    training_session.status = status
                    if status == 'completed':
                        training_session.completed_at = datetime.now()
                    if frames_count is not None:
                        training_session.frames_count = frames_count
                    
                    session.commit()
                    logger.debug(f"Updated training session for user {user_id}: {status}")
                    return training_session.id
                
        except Exception as e:
            logger.error(f"Failed to update training session: {e}")
        
        return None
    
    def train_all_pending(self):
        """Train all pending batches in Redis"""
        try:
            # Find all training batch keys
            all_keys = self.redis_client.keys("train_batch:*")
            
            if not all_keys:
                logger.info("No pending training batches found")
                return
            
            # Group by date
            dates = set()
            for key in all_keys:
                key_parts = key.decode().split(':')
                if len(key_parts) > 1:
                    dates.add(key_parts[1])
            
            logger.info(f"Found training data for {len(dates)} date(s)")
            
            # Train each date
            for date in sorted(dates):
                logger.info(f"Training data from {date}")
                self.train_from_redis(date=date)
            
        except Exception as e:
            logger.error(f"Failed to train all pending: {e}")


def main():
    """Main entry point for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Recognition Model Training')
    parser.add_argument('--date', type=str, help='Date to train (YYYY-MM-DD). Default: today')
    parser.add_argument('--user-id', type=int, help='Specific user ID to train')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Training batch size')
    parser.add_argument('--all', action='store_true', help='Train all pending batches')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.all:
        logger.info("Training all pending batches...")
        trainer.train_all_pending()
    else:
        logger.info(f"Training from Redis (date: {args.date or 'today'}, user_id: {args.user_id or 'all'})")
        trainer.train_from_redis(
            batch_size=args.batch_size,
            date=args.date,
            user_id=args.user_id
        )


if __name__ == "__main__":
    main()