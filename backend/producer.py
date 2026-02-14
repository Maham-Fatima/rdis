"""
Producer - Captures video from cameras and publishes to RabbitMQ queues
"""
import cv2
import pickle
import pika
import time
import threading
import logging
from datetime import datetime
from typing import Optional

from config import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class CameraProducer:
    """Handles video capture and message publishing"""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.face_cascade = None
        self.running = False
        self._setup()
    
    def _setup(self):
        """Initialize RabbitMQ connection and face detector"""
        try:
            # Setup RabbitMQ connection
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
            
            # Declare queues
            self.channel.queue_declare(queue=config.ATTENDANCE_QUEUE, durable=True)
            self.channel.queue_declare(queue=config.TRAINING_QUEUE, durable=True)
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + config.FACE_CASCADE_PATH
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise ValueError("Failed to load face cascade classifier")
            
            logger.info("Producer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize producer: {e}")
            raise
    
    def _publish_message(self, queue: str, payload: dict):
        """Publish message to RabbitMQ queue"""
        try:
            body = pickle.dumps(payload)
            self.channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/pickle'
                )
            )
            logger.info("sending to queue successfully")
        except Exception as e:
            logger.error(f"Failed to publish message to {queue}: {e}")
            # Reconnect and retry
            self._setup()
            body = pickle.dumps(payload)
            self.channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=body,
                properties=pika.BasicProperties(delivery_mode=2)
            )
    
    def camera_worker(self, camera_type: str, camera_stream: str, user_id: Optional[int] = None):
        """
        Process video stream from a camera
        
        Args:
            camera_type: Name of camera (e.g., 'entrance', 'exit')
            camera_stream: Camera URL or device ID
            user_id: If provided, send to training queue; otherwise to attendance queue
        """
        logger.info(f"Starting camera worker for {camera_type} {camera_stream}")
        capture = None
        frame_count = 0
        detection_count = 0
        if user_id:
            logger.info(f"training for {user_id}")
        try:
            
            capture = cv2.VideoCapture(camera_stream)
            
            if not capture.isOpened():
                logger.error(f"Failed to open camera {camera_type}")
                return
            
            while self.running:
                ret, img = capture.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from {camera_type}")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=config.DETECTION_SCALE_FACTOR,
                    minNeighbors=config.DETECTION_MIN_NEIGHBORS
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Crop face region
                    cropped_face = gray[y:y+h, x:x+w]
                    
                    # Resize face for consistency (optional but recommended)
                    cropped_face = cv2.resize(cropped_face, (200, 200))
                    
                    detection_count += 1
                    current_time = datetime.now()
                    
                    if user_id is not None:
                        # Training mode - send to training queue
                        payload = {
                            "camera_type": camera_type,
                            "timestamp": current_time,
                            "face": cropped_face,
                            "user_id": user_id
                        }
                        self._publish_message(config.TRAINING_QUEUE, payload)
                        logger.debug(f"Training frame sent from {camera_type} for user {user_id}")
                    else:
                        # Attendance mode - send to attendance queue
                        payload = {
                            "camera_type": camera_type,
                            "timestamp": current_time,
                            "face": cropped_face
                        }
                        self._publish_message(config.ATTENDANCE_QUEUE, payload)
                        logger.debug(f"Attendance frame sent from {camera_type}")
                    
                    # Draw rectangle on frame for visualization (optional)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if frame_count == 1000:
                    break
                # Optional: Display the frame (comment out in production)
                # cv2.imshow(f'Camera {camera_type}', img)
                # Check for 'q' key to quit (only works if imshow is enabled)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
                # Log stats every 100 frames
                if frame_count % 100 == 0:
                    logger.info(f"{camera_type}: Processed {frame_count} frames, detected {detection_count} faces")
        
        except Exception as e:
            logger.error(f"Error in camera worker {camera_type}: {e}")
        
        finally:
            if capture:
                capture.release()
            cv2.destroyAllWindows()
            logger.info(f"Camera worker {camera_type} stopped. Total frames: {frame_count}, Detections: {detection_count}")
    
    def start_all_cameras(self, user_id: Optional[int] = None):
        """Start all configured cameras in separate threads"""
        self.running = True
        threads = []
        if user_id is not None:
            cameras_list = [("entrance", config.CAMERAS["entrance"])] 
        else:
            cameras_list = config.CAMERAS.items()
        for camera_type, camera_stream in cameras_list:
            thread = threading.Thread(
                target=self.camera_worker,
                args=(camera_type, camera_stream, user_id),
                daemon=True,
                name=f"Camera-{camera_type}"
            )
            thread.start()
            threads.append(thread)
            logger.info(f"Started thread for camera: {camera_type}")
        
        # Wait for all threads
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping cameras...")
            self.running = False
            for thread in threads:
                thread.join(timeout=5)
        
        self.close()
    
    def close(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("Producer connection closed")


def main():
    """Main entry point for producer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Recognition Camera Producer')
    parser.add_argument('--user-id', type=int, help='User ID for training mode (if not provided, runs in attendance mode)')
    args = parser.parse_args()
    
    producer = CameraProducer()
    
    if args.user_id:
        logger.info(f"Starting in TRAINING mode for user {args.user_id}")
    else:
        logger.info("Starting in ATTENDANCE mode")   
    
    producer.start_all_cameras(user_id=args.user_id)


if __name__ == "__main__":
    main()