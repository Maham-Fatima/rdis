"""
Database Sync Worker - Syncs attendance records from Redis to PostgreSQL in batches
"""
import redis
import pickle
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict

from config import config
from database import db, init_db
from models import AttendanceRecord, User

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DatabaseSyncWorker:
    """Worker to sync attendance records from Redis buffer to PostgreSQL in batches"""
    
    def __init__(self, sync_interval: int = 10):
        """
        Args:
            sync_interval: Seconds between sync operations (default: 10)
        """
        self.redis_client = None
        self.sync_interval = sync_interval
        self.total_synced = 0
        self.running = False
        self._setup()
    
    def _setup(self):
        """Initialize Redis and database connections"""
        try:
            # Initialize database
            init_db()
            logger.info("Database initialized")
            
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Redis connected")
            
            logger.info(f"Database sync worker initialized (sync interval: {self.sync_interval}s)")
            
        except Exception as e:
            logger.error(f"Failed to initialize database sync worker: {e}")
            raise
    
    def _get_attendance_keys(self) -> List[str]:
        """Get all Redis keys containing attendance data"""
        try:
            # Find all attendance keys
            keys = self.redis_client.keys("attendance:*")
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Failed to get attendance keys: {e}")
            return []
    
    def _sync_key_to_database(self, key: str) -> int:
        """
        Sync all records from a Redis key to database
        
        Args:
            key: Redis key to sync
            
        Returns:
            Number of records synced
        """
        synced_count = 0
        
        try:
            # Get all records from this key
            records_data = []
            
            # Pop records in batches (remove from Redis as we process)
            while True:
                data = self.redis_client.lpop(key)
                if not data:
                    break
                
                try:
                    record = pickle.loads(data)
                    records_data.append(record)
                except Exception as e:
                    logger.error(f"Failed to deserialize record: {e}")
                    continue
            
            if not records_data:
                return 0
            
            # Batch insert into database
            with db.get_session() as session:
                # Get all unique user IDs
                user_ids = set(r['user_id'] for r in records_data)
                
                # Verify users exist (batch query)
                existing_users = session.query(User.id).filter(
                    User.id.in_(user_ids),
                    User.is_active == True
                ).all()
                existing_user_ids = set(u.id for u in existing_users)
                
                # Create attendance records (only for existing users)
                attendance_records = []
                for record in records_data:
                    user_id = record['user_id']
                    
                    if user_id not in existing_user_ids:
                        logger.warning(f"User {user_id} not found or inactive - skipping")
                        continue
                    
                    # Parse timestamp
                    if isinstance(record['timestamp'], str):
                        timestamp = datetime.fromisoformat(record['timestamp'])
                    else:
                        timestamp = record['timestamp']
                    
                    attendance_records.append(
                        AttendanceRecord(
                            user_id=user_id,
                            camera_type=record.get('camera_type', 'unknown'),
                            timestamp=timestamp,
                            confidence=record.get('confidence')
                        )
                    )
                
                # Bulk insert (much faster than individual inserts)
                if attendance_records:
                    session.bulk_save_objects(attendance_records)
                    session.commit()
                    synced_count = len(attendance_records)
                    logger.info(f"Synced {synced_count} records from {key} to database")
                
        except Exception as e:
            logger.error(f"Failed to sync key {key}: {e}")
        
        return synced_count
    
    def _sync_all_keys(self):
        """Sync all pending attendance records from Redis to database"""
        try:
            keys = self._get_attendance_keys()
            
            if not keys:
                logger.debug("No attendance records to sync")
                return
            
            logger.info(f"Found {len(keys)} Redis keys to sync")
            
            total_synced = 0
            for key in keys:
                count = self._sync_key_to_database(key)
                total_synced += count
            
            if total_synced > 0:
                self.total_synced += total_synced
                logger.info(f"Sync complete: {total_synced} records written to database (total: {self.total_synced})")
        
        except Exception as e:
            logger.error(f"Error during sync: {e}")
    
    def _cleanup_old_keys(self, days: int = 7):
        """
        Clean up Redis keys older than specified days
        
        Args:
            days: Number of days to keep in Redis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            keys = self._get_attendance_keys()
            
            for key in keys:
                # Extract date from key (format: attendance:YYYY-MM-DD:camera)
                parts = key.split(':')
                if len(parts) >= 2:
                    try:
                        key_date = datetime.strptime(parts[1], '%Y-%m-%d')
                        if key_date < cutoff_date:
                            # Only delete if it's been synced (empty)
                            if self.redis_client.llen(key) == 0:
                                self.redis_client.delete(key)
                                logger.info(f"Cleaned up old key: {key}")
                    except ValueError:
                        continue
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def start(self):
        """Start the sync worker (runs continuously)"""
        try:
            self.running = True
            logger.info(f"Starting database sync worker (interval: {self.sync_interval}s)...")
            logger.info("Press CTRL+C to exit.")
            
            cleanup_counter = 0
            
            while self.running:
                # Sync records
                self._sync_all_keys()
                
                # Cleanup old keys every 100 sync cycles (~16 minutes at 10s interval)
                cleanup_counter += 1
                if cleanup_counter >= 100:
                    self._cleanup_old_keys()
                    cleanup_counter = 0
                
                # Sleep before next sync
                time.sleep(self.sync_interval)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in sync worker: {e}")
            raise
    
    def stop(self):
        """Stop the worker gracefully"""
        try:
            self.running = False
            
            # Do one final sync before stopping
            logger.info("Performing final sync...")
            self._sync_all_keys()
            
            if self.redis_client:
                self.redis_client.close()
            
            logger.info(f"Database sync worker stopped. Total synced: {self.total_synced}")
        except Exception as e:
            logger.error(f"Error stopping worker: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Sync Worker - Redis to PostgreSQL')
    parser.add_argument('--interval', type=int, default=10, help='Sync interval in seconds (default: 10)')
    parser.add_argument('--once', action='store_true', help='Sync once and exit (for cron jobs)')
    
    args = parser.parse_args()
    
    worker = DatabaseSyncWorker(sync_interval=args.interval)
    
    if args.once:
        logger.info("Running one-time sync...")
        worker._sync_all_keys()
        logger.info("One-time sync complete")
    else:
        worker.start()


if __name__ == "__main__":
    main()