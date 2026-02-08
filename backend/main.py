"""
Main Orchestrator for Face Recognition Attendance System
"""
import argparse
import logging
import subprocess
import sys
import signal
import time
from typing import List

from config import config
from database import init_db, db
from models import User

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages all system services"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = False
    
    def start_service(self, script: str, name: str, args: List[str] = None):
        """Start a service in a subprocess"""
        try:
            cmd = [sys.executable, script]
            if args:
                cmd.extend(args)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes.append(process)
            logger.info(f"Started {name} (PID: {process.pid})")
            return process
        
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return None
    
    def start_all(self):
        """Start all services"""
        logger.info("Starting Face Recognition Attendance System...")
        self.running = True
        
        # Start workers
        self.start_service('attendance_worker.py', 'Attendance Worker')
        self.start_service('training_worker.py', 'Training Worker')
        self.start_service('db_sync_worker.py', 'Database Sync Worker')
        
        # Start producer (attendance mode)
        self.start_service('producer.py', 'Camera Producer')
        
        logger.info("All services started successfully")
        logger.info("Press Ctrl+C to stop all services")
        
        # Monitor processes
        try:
            while self.running:
                time.sleep(1)
                # Check if any process died
                for i, proc in enumerate(self.processes):
                    if proc.poll() is not None:
                        logger.warning(f"Process {i} exited with code {proc.returncode}")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop_all()
    
    def stop_all(self):
        """Stop all services gracefully"""
        logger.info("Stopping all services...")
        self.running = False
        
        for proc in self.processes:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=10)
                logger.info(f"Stopped process {proc.pid}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {proc.pid} did not stop gracefully, killing...")
                proc.kill()
            except Exception as e:
                logger.error(f"Error stopping process {proc.pid}: {e}")
        
        logger.info("All services stopped")


def setup_database():
    """Initialize database and create tables"""
    logger.info("Setting up database...")
    init_db()
    logger.info("Database setup complete")


def create_user(name: str, email: str = None, department: str = None, role: str = None):
    """Create a new user"""
    try:
        init_db()
        with db.get_session() as session:
            user = User(
                name=name,
                email=email,
                department=department,
                role=role
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            
            logger.info(f"User created successfully: ID={user.id}, Name={user.name}")
            print(f"\n✓ User created successfully!")
            print(f"  User ID: {user.id}")
            print(f"  Name: {user.name}")
            print(f"  Email: {user.email or 'N/A'}")
            print(f"\nUse this ID ({user.id}) when collecting training data.")
            return user.id
    
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        print(f"\n✗ Failed to create user: {e}")
        return None


def list_users():
    """List all users"""
    try:
        init_db()
        with db.get_session() as session:
            users = session.query(User).all()
            
            if not users:
                print("\nNo users found.")
                return
            
            print(f"\n{'ID':<5} {'Name':<20} {'Email':<30} {'Department':<15} {'Active':<8}")
            print("-" * 85)
            
            for user in users:
                print(f"{user.id:<5} {user.name:<20} {user.email or 'N/A':<30} {user.department or 'N/A':<15} {'Yes' if user.is_active else 'No':<8}")
            
            print(f"\nTotal users: {len(users)}")
    
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        print(f"\n✗ Failed to list users: {e}")


def collect_training_data(user_id: int):
    """Start camera producer in training mode for a specific user"""
    logger.info(f"Starting training data collection for user {user_id}")
    print(f"\nStarting training data collection for user {user_id}...")
    print("The system will capture face images from all configured cameras.")
    print("Press 'q' in the camera window to stop.\n")
    
    try:
        subprocess.run([sys.executable, 'producer.py', '--user-id', str(user_id)])
    except KeyboardInterrupt:
        logger.info("Training data collection stopped by user")


def train_model(user_id: int = None, date: str = None):
    """Train the face recognition model"""
    cmd = [sys.executable, 'collect_data.py']
    
    if user_id:
        cmd.extend(['--user-id', str(user_id)])
    if date:
        cmd.extend(['--date', date])
    
    logger.info(f"Starting model training... (user_id: {user_id or 'all'}, date: {date or 'today'})")
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Training failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Face Recognition Attendance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup database
  python main.py --setup
  
  # Create a user
  python main.py --create-user "John Doe" --email "john@example.com"
  
  # List all users
  python main.py --list-users
  
  # Collect training data for user ID 1
  python main.py --collect-training 1
  
  # Train model
  python main.py --train
  
  # Start all services (attendance mode)
  python main.py --start-all
        """
    )
    
    # Actions
    parser.add_argument('--setup', action='store_true', help='Setup database')
    parser.add_argument('--create-user', metavar='NAME', help='Create a new user')
    parser.add_argument('--email', help='User email')
    parser.add_argument('--department', help='User department')
    parser.add_argument('--role', help='User role')
    parser.add_argument('--list-users', action='store_true', help='List all users')
    parser.add_argument('--collect-training', type=int, metavar='USER_ID', help='Collect training data for user')
    parser.add_argument('--train', action='store_true', help='Train face recognition model')
    parser.add_argument('--train-user', type=int, metavar='USER_ID', help='Train for specific user')
    parser.add_argument('--train-date', help='Train for specific date (YYYY-MM-DD)')
    parser.add_argument('--start-all', action='store_true', help='Start all services')
    
    args = parser.parse_args()
    
    # Execute actions
    if args.setup:
        setup_database()
    
    elif args.create_user:
        create_user(args.create_user, args.email, args.department, args.role)
    
    elif args.list_users:
        list_users()
    
    elif args.collect_training:
        collect_training_data(args.collect_training)
    
    elif args.train:
        train_model(args.train_user, args.train_date)
    
    elif args.start_all:
        manager = ServiceManager()
        manager.start_all()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()