import json
import sqlite3
from datetime import datetime
import sys
from Data.Database.db import Database
db = Database()


def process_json_file(file_path, cursor, conn):
    try:
        # Read the JSON data from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Check if the data is already in a list format or if it's a single object
        if not isinstance(data, list):
            data = [data]

        for article in data:
            # Extract the author - handle both string and list formats
            author = article.get('author', '')
            if isinstance(author, list):
                author = ', '.join(author)

            # Prepare the data for insertion
            article_data = (
                article.get('date', ''),  # Date
                author,  # Author
                '',  # Source_ID - left empty
                article.get('publisher', ''),  # Source_Name
                article.get('title', ''),  # Title
                article.get('description', ''),  # Description
                article.get('url', ''),  # Url
                article.get('text', ''),  # Content
                '',  # Event_Type - left empty
                '',  # Sentiment_Score - left empty
                None,  # Sentiment_Score - left empty
                ''  # Fetch_Timestamp - left empty
            )

            # Insert the data into the database
            cursor.execute('''
            INSERT INTO Articles 
            (Date, Author, Source_ID, Source_Name, Title, Description, Url, Content, Event_Type, Sentiment_Label, Sentiment_Score, Fetch_Timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', article_data)

        # Commit the changes
        conn.commit()
        print(f"Successfully imported {len(data)} article(s) to the database.")

    except json.JSONDecodeError:
        print(f"Error: The file {file_path} does not contain valid JSON.")
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function to run the script."""
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <json_file_path>")
    #     return

    file_path = "dataset_fox-news-scraper_2025-04-10_05-31-57-464.json"
    cursor = db.cursor
    conn = db.connection

    try:
        process_json_file(file_path, cursor, conn)
    finally:
        db.close()


if __name__ == "__main__":
    main()