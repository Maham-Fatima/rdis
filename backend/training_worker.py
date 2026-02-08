"""
Training Worker - Collects training data and triggers model training
"""
import pika
import redis
import pickle
import logging
from datetime import datetime

from config import config
from database import db, init_db
from models import TrainingSession

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TrainingWorker:
    """Worker to collect training frames and batch them for model training"""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.redis_client = None
        self.batch = []
        self.processed_count = 0
        self._setup()
    
    def _setup(self):
        """Initialize database, Redis, and RabbitMQ connections"""
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
            self.channel.queue_declare(queue=config.TRAINING_QUEUE, durable=True)
            
            # Set QoS
            self.channel.basic_qos(prefetch_count=config.PREFETCH_COUNT)
            
            logger.info("Training worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize training worker: {e}")
            raise
    
    def _push_batch_to_redis(self, user_id: int):
        """
        Push collected batch to Redis for training
        
        Args:
            user_id: User ID for this training batch
        """
        try:
            if not self.batch:
                return
            
            # Create Redis key with date and user_id
            date_str = datetime.now().strftime('%Y-%m-%d')
            key = f"train_batch:{date_str}:{user_id}"
            
            # Push each face in batch to Redis list
            for face_data in self.batch:
                self.redis_client.rpush(key, pickle.dumps(face_data))
            
            logger.info(f"Pushed batch of {len(self.batch)} frames to Redis for user {user_id}")
            
            # Create training session record
            with db.get_session() as session:
                training_session = TrainingSession(
                    user_id=user_id,
                    frames_count=len(self.batch),
                    started_at=datetime.now(),
                    status='pending'
                )
                session.add(training_session)
                session.commit()
            
            # Clear batch
            self.batch.clear()
            
        except Exception as e:
            logger.error(f"Failed to push batch to Redis: {e}")
    
    def _process_message(self, ch, method, properties, body):
        """
        Process training message from queue
        
        Args:
            ch: Channel
            method: Delivery method
            properties: Message properties
            body: Message body (pickled data)
        """
        try:
            # Deserialize message
            payload = pickle.loads(body)
            
            face = payload.get('face')
            user_id = payload.get('user_id', 1)  # Default to user 1 if not specified
            timestamp = payload.get('timestamp', datetime.now())
            
            # Validate data
            if face is None:
                logger.warning("Received training message with no face data")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Add to batch
            self.batch.append({
                'face': face,
                'user_id': user_id,
                'timestamp': timestamp
            })
            
            self.processed_count += 1
            
            # When batch is full, push to Redis
            if len(self.batch) >= config.BATCH_SIZE:
                logger.info(f"Batch full ({len(self.batch)} frames), pushing to Redis")
                self._push_batch_to_redis(user_id)
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            # Log progress
            if self.processed_count % 50 == 0:
                logger.info(f"Collected {self.processed_count} training frames (current batch: {len(self.batch)})")
        
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to deserialize message: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        
        except Exception as e:
            logger.error(f"Error processing training message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start(self):
        """Start consuming messages from training queue"""
        try:
            logger.info("Starting training worker...")
            logger.info(f"Waiting for messages from '{config.TRAINING_QUEUE}' queue. Press CTRL+C to exit.")
            
            # Start consuming
            self.channel.basic_consume(
                queue=config.TRAINING_QUEUE,
                on_message_callback=self._process_message
            )
            
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in training worker: {e}")
            raise
    
    def stop(self):
        """Stop the worker gracefully"""
        try:
            # Push any remaining batch to Redis
            if self.batch:
                logger.info(f"Pushing final batch of {len(self.batch)} frames")
                # We need to get user_id from the batch
                if self.batch:
                    user_id = self.batch[0].get('user_id', 1)
                    self._push_batch_to_redis(user_id)
            
            if self.channel:
                self.channel.stop_consuming()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.redis_client:
                self.redis_client.close()
            
            logger.info(f"Training worker stopped. Total processed: {self.processed_count}")
        except Exception as e:
            logger.error(f"Error stopping worker: {e}")


def main():
    """Main entry point"""
    worker = TrainingWorker()
    worker.start()


if __name__ == "__main__":
    main()