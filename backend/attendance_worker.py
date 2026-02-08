"""
Attendance Worker - Processes attendance queue and stores records in Redis buffer
"""
import pickle
import pika
import redis
import logging
from datetime import datetime
from typing import Optional

from config import config
from predict import predict_with_confidence

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class AttendanceWorker:
    """Worker to process attendance recognition and store in Redis buffer (fast I/O)"""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.redis_client = None
        self.processed_count = 0
        self.recognized_count = 0
        self._setup()
    
    def _setup(self):
        """Initialize Redis and RabbitMQ connections"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Redis connected")
            
            # Setup RabbitMQ
            credentials = pika.PlainCredentials(config.RABBITMQ_USER, config.RABBITMQ_PASS)
            parameters = pika.ConnectionParameters(
                host=config.RABBITMQ_HOST,
                port=config.RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queue
            self.channel.queue_declare(queue=config.ATTENDANCE_QUEUE, durable=True)
            
            # Set QoS
            self.channel.basic_qos(prefetch_count=config.PREFETCH_COUNT)
            
            logger.info("Attendance worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize attendance worker: {e}")
            raise
    
    def _store_attendance_redis(self, user_id: int, camera_type: str, timestamp: datetime, confidence: float):
        """
        Store attendance record in Redis buffer (fast, non-blocking)
        
        Args:
            user_id: User ID
            camera_type: Camera location (e.g., 'entrance', 'exit')
            timestamp: Time of detection
            confidence: Recognition confidence score
        """
        try:
            # Create Redis key with date
            date_str = timestamp.strftime('%Y-%m-%d')
            key = f"attendance:{date_str}:{camera_type}"
            
            # Prepare record
            record = {
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "confidence": confidence,
                "camera_type": camera_type
            }
            
            # Push to Redis list (O(1) operation, very fast)
            self.redis_client.rpush(key, pickle.dumps(record))
            
            logger.debug(f"Attendance buffered in Redis: User {user_id} at {camera_type}")
            self.recognized_count += 1
                
        except Exception as e:
            logger.error(f"Failed to buffer attendance in Redis: {e}")
    
    def _process_message(self, ch, method, properties, body):
        """
        Process attendance message from queue
        
        Args:
            ch: Channel
            method: Delivery method
            properties: Message properties
            body: Message body (pickled data)
        """
        try:
            # Deserialize message
            data = pickle.loads(body)
            
            timestamp = data.get('timestamp')
            face = data.get('face')
            camera_type = data.get('camera_type', 'unknown')
            
            # Validate data
            if face is None:
                logger.warning("Received message with no face data")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Predict user ID
            user_id, confidence = predict_with_confidence(face)
            
            if user_id is not None:
                # Store in Redis buffer (fast write)
                self._store_attendance_redis(user_id, camera_type, timestamp, confidence)
            else:
                logger.debug(f"Face not recognized from {camera_type}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            self.processed_count += 1
            
            # Log progress
            if self.processed_count % 100 == 0:
                recognition_rate = (self.recognized_count / self.processed_count) * 100
                logger.info(f"Processed {self.processed_count} frames, recognized {self.recognized_count} ({recognition_rate:.1f}%)")
        
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to deserialize message: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge to remove from queue
        
        except Exception as e:
            logger.error(f"Error processing attendance message: {e}")
            # Don't acknowledge - message will be requeued
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start(self):
        """Start consuming messages from attendance queue"""
        try:
            logger.info("Starting attendance worker...")
            logger.info(f"Waiting for messages from '{config.ATTENDANCE_QUEUE}' queue. Press CTRL+C to exit.")
            
            # Start consuming
            self.channel.basic_consume(
                queue=config.ATTENDANCE_QUEUE,
                on_message_callback=self._process_message
            )
            
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in attendance worker: {e}")
            raise
    
    def stop(self):
        """Stop the worker gracefully"""
        try:
            if self.channel:
                self.channel.stop_consuming()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.redis_client:
                self.redis_client.close()
            
            logger.info(f"Attendance worker stopped. Total processed: {self.processed_count}, Recognized: {self.recognized_count}")
        except Exception as e:
            logger.error(f"Error stopping worker: {e}")


def main():
    """Main entry point"""
    worker = AttendanceWorker()
    worker.start()


if __name__ == "__main__":
    main()