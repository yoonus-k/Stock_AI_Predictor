import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
current_dir = Path(os.getcwd())
project_root = current_dir
if "Stock_AI_Predictor" in str(current_dir):
    while not (project_root / "README.md").exists() and str(project_root) != project_root.root:
        project_root = project_root.parent

# Database path
db_path = project_root / 'Data' / 'data.db'

# Connect to the database
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check if the articles table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles';")
if cursor.fetchone():
    # Get the table schema
    print("Articles table schema:")
    schema = cursor.execute("PRAGMA table_info(articles)").fetchall()
    for col in schema:
        print(f"- {col[1]} ({col[2]})")
    
    # Check if the table has an event_type column
    has_event_type = any(col[1] == 'event_type' for col in schema)
    if not has_event_type:
        print("\nThe 'event_type' column is missing in the articles table.")
        print("Adding the column...")
        try:
            cursor.execute("ALTER TABLE articles ADD COLUMN event_type TEXT;")
            conn.commit()
            print("Column 'event_type' added successfully.")
        except sqlite3.Error as e:
            print(f"Error adding column: {e}")
    else:
        print("\nThe 'event_type' column already exists in the articles table.")

    # Check the date format in the table
    print("\nSample dates from articles table:")
    try:
        cursor.execute("SELECT date FROM articles LIMIT 5;")
        dates = cursor.fetchall()
        for date in dates:
            print(f"- {date[0]}")
    except sqlite3.Error as e:
        print(f"Error fetching dates: {e}")
else:
    print("The 'articles' table doesn't exist in the database.")

# Check for articles_new table
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles_new';")
if cursor.fetchone():
    print("\nThe 'articles_new' table exists in the database.")
else:
    print("\nThe 'articles_new' table doesn't exist in the database.")

# Close the connection
conn.close()
