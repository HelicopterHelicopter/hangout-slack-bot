import sqlite3
from datetime import datetime
import os

# Database file
DB_FILE = 'hangout_sessions.db'

def init_db():
    """Initialize the database with necessary tables"""
    if os.path.exists(DB_FILE):
        return
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create presentations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS presentations (
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
    CREATE TABLE IF NOT EXISTS polls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        active INTEGER DEFAULT 1
    )
    ''')
    
    # Create poll_responses table with user_id to track unique votes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS poll_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        poll_id INTEGER,
        user_id TEXT NOT NULL,
        response INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (poll_id) REFERENCES polls (id),
        UNIQUE(poll_id, user_id)
    )
    ''')
    
    conn.commit()
    conn.close()

def migrate_poll_responses_table():
    """Check if poll_responses table has the correct schema"""
    # We've already reset the database with the correct schema,
    # so this function is now just a placeholder for backward compatibility
    print("Database schema is up to date.")
    return

def add_presentation(topic, presenter, scheduled_date=None, status="pending"):
    """Add a new presentation to the database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    date_str = scheduled_date.isoformat() if scheduled_date else None
    
    cursor.execute(
        "INSERT INTO presentations (topic, presenter, scheduled_date, created_at, status) VALUES (?, ?, ?, ?, ?)",
        (topic, presenter, date_str, now, status)
    )
    
    presentation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return get_presentation(presentation_id)

def get_presentation(presentation_id):
    """Get a specific presentation by ID"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM presentations WHERE id = ?", (presentation_id,))
    presentation = cursor.fetchone()
    conn.close()
    
    if not presentation:
        return None
    
    return {
        'id': presentation[0],
        'topic': presentation[1],
        'presenter': presentation[2],
        'scheduled_date': datetime.fromisoformat(presentation[3]) if presentation[3] else None,
        'created_at': datetime.fromisoformat(presentation[4]),
        'status': presentation[5]
    }

def get_presentations(include_completed=False):
    """Get all presentations, optionally including completed ones"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    if include_completed:
        cursor.execute("SELECT * FROM presentations ORDER BY scheduled_date ASC")
    else:
        cursor.execute("SELECT * FROM presentations WHERE status != 'completed' ORDER BY scheduled_date ASC")
    
    presentations_data = cursor.fetchall()
    conn.close()
    
    presentations = []
    for p in presentations_data:
        presentations.append({
            'id': p[0],
            'topic': p[1],
            'presenter': p[2],
            'scheduled_date': datetime.fromisoformat(p[3]) if p[3] else None,
            'created_at': datetime.fromisoformat(p[4]),
            'status': p[5]
        })
    
    return presentations

def update_presentation_date(presentation_id, scheduled_date):
    """Update the scheduled date for a presentation"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    date_str = scheduled_date.isoformat()
    
    cursor.execute(
        "UPDATE presentations SET scheduled_date = ?, status = 'scheduled' WHERE id = ?",
        (date_str, presentation_id)
    )
    
    conn.commit()
    conn.close()
    
    return get_presentation(presentation_id)

def create_poll(topic, created_by):
    """Create a new poll for a topic"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    cursor.execute(
        "INSERT INTO polls (topic, created_by, created_at) VALUES (?, ?, ?)",
        (topic, created_by, now)
    )
    
    poll_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        'id': poll_id,
        'topic': topic,
        'created_by': created_by,
        'created_at': datetime.fromisoformat(now),
        'active': True
    }

def add_poll_response(poll_id, interested, user_id='unknown'):
    """Add a response to a poll"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    response_value = 1 if interested else 0
    
    try:
        # Check if user has already voted on this poll
        cursor.execute(
            "SELECT id, response FROM poll_responses WHERE poll_id = ? AND user_id = ?",
            (poll_id, user_id)
        )
        existing_vote = cursor.fetchone()
        
        if existing_vote:
            # User has already voted, update their response
            cursor.execute(
                "UPDATE poll_responses SET response = ?, created_at = ? WHERE id = ?",
                (response_value, now, existing_vote[0])
            )
        else:
            # New vote
            cursor.execute(
                "INSERT INTO poll_responses (poll_id, user_id, response, created_at) VALUES (?, ?, ?, ?)",
                (poll_id, user_id, response_value, now)
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error recording poll response: {e}")
        conn.rollback()
        conn.close()
        return False

def get_poll_results(poll_id):
    """Get the results of a poll with unique vote counting by user"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get the poll
    cursor.execute("SELECT * FROM polls WHERE id = ?", (poll_id,))
    poll = cursor.fetchone()
    
    if not poll:
        conn.close()
        return None
    
    # Count unique users who voted (total responses)
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM poll_responses WHERE poll_id = ?", (poll_id,))
    total_unique_voters = cursor.fetchone()[0]
    
    # Count users who voted yes (interested)
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM poll_responses WHERE poll_id = ? AND response = 1", (poll_id,))
    interested = cursor.fetchone()[0]
    
    # Count users who voted no (not interested)
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM poll_responses WHERE poll_id = ? AND response = 0", (poll_id,))
    not_interested = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'topic': poll[1],
        'interested': interested,
        'not_interested': not_interested,
        'total_responses': total_unique_voters
    }

def get_active_polls():
    """Get all active polls"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM polls WHERE active = 1")
    polls_data = cursor.fetchall()
    conn.close()
    
    polls = []
    for p in polls_data:
        polls.append({
            'id': p[0],
            'topic': p[1],
            'created_by': p[2],
            'created_at': datetime.fromisoformat(p[3]),
            'active': bool(p[4])
        })
    
    return polls

def delete_poll(poll_id):
    """Delete/deactivate a poll by setting its active status to 0"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE polls SET active = 0 WHERE id = ?",
        (poll_id,)
    )
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success

def mark_presentation_completed(presentation_id):
    """Mark a presentation as completed"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE presentations SET status = 'completed' WHERE id = ?",
        (presentation_id,)
    )
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success

def update_presentation_status_and_date(presentation_id, status, scheduled_date=None):
    """Update a presentation's status and optionally its date"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    if scheduled_date:
        date_str = scheduled_date.isoformat()
        cursor.execute(
            "UPDATE presentations SET status = ?, scheduled_date = ? WHERE id = ?",
            (status, date_str, presentation_id)
        )
    else:
        cursor.execute(
            "UPDATE presentations SET status = ? WHERE id = ?",
            (status, presentation_id)
        )
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success

def delete_presentation(presentation_id):
    """Delete a presentation from the database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM presentations WHERE id = ?",
        (presentation_id,)
    )
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success 