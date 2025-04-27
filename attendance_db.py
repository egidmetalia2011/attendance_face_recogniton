import sqlite3
import os
from datetime import datetime

class AttendanceDB:
    def __init__(self):
        self.db_file = "attendance.db"
        self.initialize_db()

    def initialize_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Create Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        ''')

        # Create Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date DATE NOT NULL,
                time TIME NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')

        # Initialize students from dataset directory
        dataset_dir = "dataset"
        if os.path.exists(dataset_dir):
            for student_name in os.listdir(dataset_dir):
                if not student_name.startswith('.'):  # Skip hidden files
                    cursor.execute('''
                        INSERT OR IGNORE INTO students (name)
                        VALUES (?)
                    ''', (student_name,))

        conn.commit()
        conn.close()

    def mark_attendance(self, name):
        """Mark attendance for a student"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        # Get student ID
        cursor.execute("SELECT id FROM students WHERE name = ?", (name,))
        result = cursor.fetchone()
        
        if result:
            student_id = result[0]
            
            # Check if attendance already marked for today
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ?
            """, (student_id, date))
            
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO attendance (student_id, date, time, status)
                    VALUES (?, ?, ?, ?)
                """, (student_id, date, time, "Present"))
                conn.commit()

        conn.close()

    def get_daily_attendance(self, date=None):
        """Get attendance records for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT s.name, a.time, a.status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
            ORDER BY s.name
        """, (date,))

        results = cursor.fetchall()
        conn.close()
        return results

    def get_all_students(self):
        """Get list of all registered students"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM students ORDER BY name")
        students = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return students

    def reset_attendance(self):
        """Reset the attendance database by dropping and recreating tables"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS attendance")
        cursor.execute("DROP TABLE IF EXISTS students")

        # Reinitialize the database
        self.initialize_db()

        conn.close()
