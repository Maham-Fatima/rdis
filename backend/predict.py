"""
Face Recognition Prediction Module with Redis Caching
"""
import cv2
import redis
import logging
import os
from typing import Optional, Tuple

from config import config

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition with model caching"""
    
    def __init__(self):
        self.recognizer = None
        self.redis_client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize recognizer and Redis connection"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=False  # We're storing binary data
            )
            self.redis_client.ping()  # Test connection
            
            # Initialize recognizer
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Try to load model
            self._load_model()
            
            logger.info("Face recognizer initialized successfully")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize face recognizer: {e}")
            raise
    
    def _load_model(self) -> bool:
        """
        Load model from Redis cache or disk
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Try Redis cache first
            model_bytes = self.redis_client.get(config.REDIS_MODEL_KEY)
            
            if model_bytes:
                # Load from Redis
                tmp_path = "trainer_tmp.yml"
                with open(tmp_path, "wb") as f:
                    f.write(model_bytes)
                self.recognizer.read(tmp_path)
                os.remove(tmp_path)  # Clean up
                logger.info("Model loaded from Redis cache")
                return True
            
            # Try disk
            if os.path.exists(config.MODEL_PATH):
                self.recognizer.read(config.MODEL_PATH)
                logger.info("Model loaded from disk")
                
                # Cache in Redis for next time
                with open(config.MODEL_PATH, "rb") as f:
                    self.redis_client.set(config.REDIS_MODEL_KEY, f.read())
                logger.info("Model cached in Redis")
                return True
            
            logger.warning("No trained model found - predictions will fail until training is complete")
            return False
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def reload_model(self) -> bool:
        """
        Reload model from cache/disk
        Should be called after training completes
        """
        logger.info("Reloading model...")
        return self._load_model()
    
    def predict(self, face) -> Tuple[Optional[int], Optional[float]]:
        """
        Predict user ID from face image
        
        Args:
            face: Grayscale face image (numpy array)
            
        Returns:
            Tuple of (user_id, confidence) or (None, None) if prediction fails
        """
        try:
            if face is None or face.size == 0:
                logger.warning("Empty face image provided")
                return None, None
            
            # Resize face to consistent size if needed
            if face.shape != (200, 200):
                face = cv2.resize(face, (200, 200))
            
            # Predict
            user_id, confidence = self.recognizer.predict(face)
            
            # LBPH confidence: lower is better (0 = perfect match)
            # Typical range: 0-100+
            # We want confidence below threshold to accept
            if confidence < config.CONFIDENCE_THRESHOLD * 100:  # Convert threshold
                logger.debug(f"Recognized user {user_id} with confidence {confidence:.2f}")
                return user_id, confidence
            else:
                logger.debug(f"Recognition confidence too low: {confidence:.2f} (threshold: {config.CONFIDENCE_THRESHOLD * 100})")
                return None, None
        
        except cv2.error as e:
            logger.error(f"OpenCV error during prediction: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, None
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()


# Global recognizer instance (singleton pattern)
_recognizer_instance = None


def get_recognizer() -> FaceRecognizer:
    """Get or create global recognizer instance"""
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = FaceRecognizer()
    return _recognizer_instance


def predict(face) -> Optional[int]:
    """
    Convenience function for backward compatibility
    
    Args:
        face: Grayscale face image
        
    Returns:
        User ID if recognized, None otherwise
    """
    recognizer = get_recognizer()
    user_id, _ = recognizer.predict(face)
    return user_id


def predict_with_confidence(face) -> Tuple[Optional[int], Optional[float]]:
    """
    Predict with confidence score
    
    Args:
        face: Grayscale face image
        
    Returns:
        Tuple of (user_id, confidence)
    """
    recognizer = get_recognizer()
    return recognizer.predict(face)