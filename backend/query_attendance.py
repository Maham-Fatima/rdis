"""
Attendance Query Utility - View and export attendance records
"""
import argparse
from datetime import datetime, timedelta
import csv
import logging

from database import init_db, db
from models import AttendanceRecord, User
from sqlalchemy import func, and_

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def get_attendance_by_date(date: str = None, camera_type: str = None):
    """
    Get attendance records for a specific date
    
    Args:
        date: Date string (YYYY-MM-DD). If None, uses today
        camera_type: Filter by camera type (entrance/exit)
    """
    try:
        init_db()
        
        if date is None:
            target_date = datetime.now().date()
        else:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        
        with db.get_session() as session:
            query = session.query(
                AttendanceRecord.id,
                User.id.label('user_id'),
                User.name,
                User.department,
                AttendanceRecord.camera_type,
                AttendanceRecord.timestamp,
                AttendanceRecord.confidence
            ).join(User, AttendanceRecord.user_id == User.id)
            
            # Filter by date
            query = query.filter(
                func.date(AttendanceRecord.timestamp) == target_date
            )
            
            # Filter by camera type if specified
            if camera_type:
                query = query.filter(AttendanceRecord.camera_type == camera_type)
            
            # Order by timestamp
            query = query.order_by(AttendanceRecord.timestamp)
            
            records = query.all()
            
            # Display results
            print(f"\n{'='*100}")
            print(f"Attendance Report - {target_date}")
            if camera_type:
                print(f"Camera: {camera_type}")
            print(f"{'='*100}\n")
            
            if not records:
                print("No records found.")
                return []
            
            print(f"{'ID':<6} {'User ID':<8} {'Name':<20} {'Department':<15} {'Camera':<10} {'Time':<20} {'Confidence':<10}")
            print("-" * 100)
            
            for record in records:
                print(f"{record.id:<6} {record.user_id:<8} {record.name:<20} {record.department or 'N/A':<15} {record.camera_type:<10} {record.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {record.confidence:.2f if record.confidence else 'N/A':<10}")
            
            print(f"\nTotal records: {len(records)}")
            
            return records
    
    except Exception as e:
        logger.error(f"Failed to get attendance: {e}")
        return []


def get_user_attendance(user_id: int, start_date: str = None, end_date: str = None):
    """
    Get attendance records for a specific user
    
    Args:
        user_id: User ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    try:
        init_db()
        
        with db.get_session() as session:
            # Get user
            user = session.query(User).filter_by(id=user_id).first()
            
            if not user:
                print(f"\n✗ User {user_id} not found.")
                return []
            
            # Build query
            query = session.query(AttendanceRecord).filter(
                AttendanceRecord.user_id == user_id
            )
            
            # Date filters
            if start_date:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                query = query.filter(AttendanceRecord.timestamp >= start)
            
            if end_date:
                end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                query = query.filter(AttendanceRecord.timestamp < end)
            
            records = query.order_by(AttendanceRecord.timestamp).all()
            
            # Display results
            print(f"\n{'='*80}")
            print(f"Attendance Report for {user.name} (ID: {user_id})")
            if start_date or end_date:
                print(f"Period: {start_date or 'Beginning'} to {end_date or 'Now'}")
            print(f"{'='*80}\n")
            
            if not records:
                print("No records found.")
                return []
            
            print(f"{'Date':<12} {'Time':<10} {'Camera':<10} {'Confidence':<12}")
            print("-" * 80)
            
            for record in records:
                print(f"{record.timestamp.strftime('%Y-%m-%d'):<12} {record.timestamp.strftime('%H:%M:%S'):<10} {record.camera_type:<10} {record.confidence:.2f if record.confidence else 'N/A':<12}")
            
            print(f"\nTotal records: {len(records)}")
            
            return records
    
    except Exception as e:
        logger.error(f"Failed to get user attendance: {e}")
        return []


def get_attendance_summary(start_date: str = None, end_date: str = None):
    """
    Get attendance summary by user
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    try:
        init_db()
        
        with db.get_session() as session:
            query = session.query(
                User.id,
                User.name,
                User.department,
                func.count(AttendanceRecord.id).label('attendance_count'),
                func.count(func.distinct(func.date(AttendanceRecord.timestamp))).label('days_present')
            ).join(AttendanceRecord, User.id == AttendanceRecord.user_id, isouter=True)
            
            # Date filters
            if start_date:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                query = query.filter(AttendanceRecord.timestamp >= start)
            
            if end_date:
                end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                query = query.filter(AttendanceRecord.timestamp < end)
            
            query = query.group_by(User.id, User.name, User.department)
            query = query.order_by(User.name)
            
            results = query.all()
            
            # Display results
            print(f"\n{'='*80}")
            print("Attendance Summary")
            if start_date or end_date:
                print(f"Period: {start_date or 'Beginning'} to {end_date or 'Now'}")
            print(f"{'='*80}\n")
            
            print(f"{'User ID':<8} {'Name':<25} {'Department':<15} {'Records':<10} {'Days':<8}")
            print("-" * 80)
            
            for result in results:
                print(f"{result.id:<8} {result.name:<25} {result.department or 'N/A':<15} {result.attendance_count:<10} {result.days_present:<8}")
            
            print()
    
    except Exception as e:
        logger.error(f"Failed to get attendance summary: {e}")


def export_to_csv(records, filename: str):
    """Export attendance records to CSV"""
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['id', 'user_id', 'name', 'department', 'camera_type', 'timestamp', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in records:
                writer.writerow({
                    'id': record.id,
                    'user_id': record.user_id,
                    'name': record.name,
                    'department': record.department or 'N/A',
                    'camera_type': record.camera_type,
                    'timestamp': record.timestamp,
                    'confidence': record.confidence if record.confidence else 'N/A'
                })
        
        print(f"\n✓ Exported {len(records)} records to {filename}")
    
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Attendance Query Utility')
    
    # Query type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--by-date', action='store_true', help='Get attendance by date')
    group.add_argument('--by-user', type=int, metavar='USER_ID', help='Get attendance for specific user')
    group.add_argument('--summary', action='store_true', help='Get attendance summary')
    
    # Filters
    parser.add_argument('--date', help='Date (YYYY-MM-DD). Default: today')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--camera', choices=['entrance', 'exit'], help='Filter by camera type')
    parser.add_argument('--export', metavar='FILENAME', help='Export to CSV file')
    
    args = parser.parse_args()
    
    records = []
    
    if args.by_date:
        records = get_attendance_by_date(args.date, args.camera)
    
    elif args.by_user:
        records = get_user_attendance(args.by_user, args.start_date, args.end_date)
    
    elif args.summary:
        get_attendance_summary(args.start_date, args.end_date)
    
    # Export if requested
    if args.export and records:
        export_to_csv(records, args.export)


if __name__ == "__main__":
    main()