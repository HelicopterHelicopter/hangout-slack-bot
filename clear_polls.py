import database as db
import sqlite3

# Initialize the database
db.init_db()

# Function to clear all polls
def clear_all_polls():
    try:
        # Connect to the database directly
        conn = sqlite3.connect(db.DB_FILE)
        cursor = conn.cursor()
        
        # Delete all poll responses
        cursor.execute("DELETE FROM poll_responses")
        print("Deleted all poll responses")
        
        # Delete all polls or set them to inactive
        cursor.execute("DELETE FROM polls")
        # Alternative: cursor.execute("UPDATE polls SET active = 0")
        print("Deleted all polls")
        
        # Commit the changes
        conn.commit()
        
        # Print confirmation
        print("All polls and their responses have been deleted successfully!")
        
    except Exception as e:
        print(f"Error clearing polls: {e}")
    finally:
        # Close connection
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_all_polls() 