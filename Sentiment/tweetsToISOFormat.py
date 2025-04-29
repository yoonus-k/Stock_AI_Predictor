import sqlite3
from datetime import datetime
import sys

def check_iso_format(created_at):
    """
    Check if a datetime string is in ISO format.

    Args:
        created_at (str): The datetime string to check.

    Returns:
        bool: True if the string is in ISO format, False otherwise.
    """
    try:
        # Attempt to parse the string as ISO format
        datetime.fromisoformat(created_at)
        return True
    except ValueError:
        return False

def convert_twitter_dates_in_db(db_path):
    """
    Convert all Twitter-formatted dates in the 'created_at' column to ISO format

    Args:
        db_path (str): Path to the SQLite database file
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all rows that need conversion
    cursor.execute("SELECT id, created_at FROM tweets")
    rows = cursor.fetchall()

    print(f"Found {len(rows)} tweets to process")
    # Printing the data
    for row in rows:
        print(row)
    # Counter for converted dates
    converted = 0
    failed = 0

    # Process each row
    for row_id, created_at in rows:
        try:
            # Check if created_at is already in ISO format
            if check_iso_format(created_at):
                print(f"Row {row_id} already in ISO format, skipping.")
                continue
            # Parse the Twitter datetime format
            dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")

            # Convert to ISO format
            iso_datetime = dt.isoformat()

            # Update the database
            cursor.execute(
                "UPDATE tweets SET created_at = ? WHERE id = ?",
                (iso_datetime, row_id)
            )

            converted += 1

            # Print progress every 100 rows
            if converted % 100 == 0:
                print(f"Converted {converted} tweets so far...")

        except Exception as e:
            print(f"Error converting datetime for row {row_id}: {e}")
            failed += 1

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print(f"Conversion complete: {converted} tweets converted, {failed} failed")


# Example usage
if __name__ == "__main__":
    # Replace with your actual database path
    database_path = "../Data/data.db"
    convert_twitter_dates_in_db(database_path)

# If your database is not SQLite or has a different schema,
# you may need to modify the code accordingly.
# For MySQL example:
"""
import mysql.connector

def convert_twitter_dates_in_mysql():
    # Connect to MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="yourusername",
        password="yourpassword",
        database="yourdatabase"
    )
    cursor = conn.cursor()

    # Rest of the code is similar, but with MySQL syntax
    # ...

    cursor.execute("SELECT id, created_at FROM tweets")
    rows = cursor.fetchall()

    for row_id, created_at in rows:
        # Same conversion logic
        # ...

        cursor.execute(
            "UPDATE tweets SET created_at = %s WHERE id = %s", 
            (iso_datetime, row_id)
        )

    conn.commit()
    cursor.close()
    conn.close()
"""