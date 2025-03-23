import sqlite3
import os
from datetime import datetime

# Database file
DB_FILE = 'hangout_sessions.db'

def reset_database():
    """Drop and recreate all tables in the database"""
    print("Resetting database...")
    
    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS poll_responses")
    cursor.execute("DROP TABLE IF EXISTS polls")
    cursor.execute("DROP TABLE IF EXISTS presentations")
    
    print("Tables dropped successfully")
    
    # Create presentations table
    cursor.execute('''
    CREATE TABLE presentations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        presenter TEXT NOT NULL,
        scheduled_date TEXT,
        created_at TEXT NOT NULL,
        status TEXT DEFAULT 'pending'
    )
    ''')
    
    # Create polls table
    cursor.execute('''
    CREATE TABLE polls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        active INTEGER DEFAULT 1
    )
    ''')
    
    # Create poll_responses table with user_id to track unique votes
    cursor.execute('''
    CREATE TABLE poll_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        poll_id INTEGER,
        user_id TEXT NOT NULL,
        response INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (poll_id) REFERENCES polls (id),
        UNIQUE(poll_id, user_id)
    )
    ''')
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    print("Database has been reset with fresh tables")
    print("All data has been wiped clean")

if __name__ == "__main__":
    # Run without confirmation
    reset_database() 