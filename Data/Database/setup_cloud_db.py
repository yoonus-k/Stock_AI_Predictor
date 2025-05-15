#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQLite Cloud Database Setup Script

This script sets up a SQLiteCloud database instance and copies your local database structure
and data to the cloud, enabling:
1. Remote database access for distributed computing (like Google Colab)
2. Concurrent access from multiple machines/environments
3. Automatic backup and disaster recovery

Uses sqlitecloud.com services which offers free tiers for smaller databases.
"""

import os
import sys
import sqlite3
import sqlitecloud
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Navigate up to project root
sys.path.append(str(project_root))

# Create argument parser
parser = argparse.ArgumentParser(description="Set up SQLite Cloud database for Stock AI Predictor")
parser.add_argument("--local-db", default="Data/Storage/data.db", help="Path to local database")
parser.add_argument("--create-env", action="store_true", help="Create .env file with connection string")
parser.add_argument("--backup", action="store_true", help="Backup cloud database to local file")
parser.add_argument("--sync", action="store_true", help="Sync local database to cloud")

def create_env_file(connection_string):
    """Create or update .env file with SQLiteCloud connection string."""
    env_path = project_root / '.env'
    
    # Load existing .env if it exists
    env_vars = {}
    if env_path.exists():
        load_dotenv(env_path)
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    # Update with new connection string
    env_vars['SQLITECLOUD_URL'] = f'"{connection_string}"'
    
    # Write back to file
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"Created/updated .env file with SQLiteCloud connection string")

def sync_to_cloud(local_db_path, cloud_connection_string):
    """Sync local database to SQLiteCloud."""
    print(f"Syncing local database to SQLiteCloud...")
    
    # Connect to local database
    local_conn = sqlite3.connect(local_db_path)
    local_cursor = local_conn.cursor()
    
    # Connect to cloud database
    cloud_conn = sqlitecloud.connect(cloud_connection_string)
    cloud_cursor = cloud_conn.cursor()
    
    # Get all tables from local database
    local_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = local_cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        if table_name.startswith('sqlite_'):
            continue  # Skip internal SQLite tables
            
        print(f"Syncing table: {table_name}")
        
        # Get table schema
        local_cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}';")
        schema = local_cursor.fetchone()[0]
        
        # Create table in cloud if it doesn't exist
        try:
            cloud_cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            cloud_cursor.execute(schema)
            print(f"  Created table {table_name} in cloud")
        except Exception as e:
            print(f"  Error creating table {table_name}: {e}")
            continue
        
        # Copy data
        try:
            # Get all data from local table
            local_cursor.execute(f"SELECT * FROM {table_name};")
            rows = local_cursor.fetchall()
            
            if not rows:
                print(f"  No data in table {table_name}")
                continue
                
            # Get column names
            local_cursor.execute(f"PRAGMA table_info({table_name});")
            columns = local_cursor.fetchall()
            column_count = len(columns)
            
            # Create placeholders for INSERT
            placeholders = ", ".join(["?" for _ in range(column_count)])
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                cloud_cursor.executemany(
                    f"INSERT INTO {table_name} VALUES ({placeholders})", 
                    batch
                )
                cloud_conn.commit()
                print(f"  Inserted {len(batch)} rows into {table_name} ({i+len(batch)}/{len(rows)})")
        except Exception as e:
            print(f"  Error copying data for table {table_name}: {e}")
    
    # Close connections
    local_conn.close()
    cloud_conn.close()
    
    print("Database sync completed!")

def backup_from_cloud(cloud_connection_string, backup_path):
    """Backup cloud database to a local file."""
    print(f"Backing up cloud database to {backup_path}...")
    
    # Connect to cloud database
    cloud_conn = sqlitecloud.connect(cloud_connection_string)
    cloud_cursor = cloud_conn.cursor()
    
    # Create local backup file
    backup_conn = sqlite3.connect(backup_path)
    backup_cursor = backup_conn.cursor()
    
    # Get all tables from cloud database
    cloud_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cloud_cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        if table_name.startswith('sqlite_'):
            continue  # Skip internal SQLite tables
            
        print(f"Backing up table: {table_name}")
        
        # Get table schema
        cloud_cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}';")
        schema = cloud_cursor.fetchone()[0]
        
        # Create table in backup
        try:
            backup_cursor.execute(schema)
            print(f"  Created table {table_name} in backup")
        except Exception as e:
            print(f"  Error creating table {table_name}: {e}")
            continue
        
        # Copy data
        try:
            # Get all data from cloud table
            cloud_cursor.execute(f"SELECT * FROM {table_name};")
            rows = cloud_cursor.fetchall()
            
            if not rows:
                print(f"  No data in table {table_name}")
                continue
                
            # Get column names
            cloud_cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cloud_cursor.fetchall()
            column_count = len(columns)
            
            # Create placeholders for INSERT
            placeholders = ", ".join(["?" for _ in range(column_count)])
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                backup_cursor.executemany(
                    f"INSERT INTO {table_name} VALUES ({placeholders})", 
                    batch
                )
                backup_conn.commit()
                print(f"  Inserted {len(batch)} rows into backup {table_name} ({i+len(batch)}/{len(rows)})")
        except Exception as e:
            print(f"  Error copying data for table {table_name}: {e}")
    
    # Close connections
    cloud_conn.close()
    backup_conn.close()
    
    print(f"Backup completed! Saved to {backup_path}")

def setup_sqlitecloud():
    """
    Main function to set up SQLiteCloud.
    
    1. Sign up for SQLiteCloud account and get connection string
    2. Create or update .env file with connection string
    3. Sync local database to cloud
    """
    # Load environment variables
    load_dotenv()
    connection_string = os.getenv('SQLITECLOUD_URL')
    
    if not connection_string:
        print("SQLiteCloud connection string not found.")
        print("Please sign up at https://sqlitecloud.io/ and get a connection string.")
        print("Format: sqlitecloud://username:password@hostname:port/database")
        connection_string = input("Enter your SQLiteCloud connection string: ")
    
    return connection_string

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Resolve local database path
    local_db_path = project_root / args.local_db
    if not local_db_path.exists():
        print(f"Error: Local database not found at {local_db_path}")
        sys.exit(1)
    
    # Set up SQLiteCloud
    connection_string = setup_sqlitecloud()
    
    if args.create_env:
        create_env_file(connection_string)
    
    if args.sync:
        sync_to_cloud(local_db_path, connection_string)
    
    if args.backup:
        backup_path = project_root / "Data" / "Storage" / f"cloud_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_from_cloud(connection_string, backup_path)
    
    if not (args.create_env or args.sync or args.backup):
        print("No action specified. Use --create-env, --sync, or --backup flags.")
        parser.print_help()
