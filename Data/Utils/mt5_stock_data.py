"""
MetaTrader 5 Stock Data Fetcher

This module connects to MetaTrader 5 to fetch stock price data across multiple timeframes 
and stores it in the database.

Features:
- Fetches data for 7 different timeframes (1min, 5min, 15min, 30min, 1h, 4h, daily)
- Checks database for existing data to avoid duplicates
- Handles proper timeframe mapping between MT5 and database
- Gets stock ID from stocks table when fetching data
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import time
import sys
import os


# Define mapping between MT5 timeframes and database timeframe IDs
def get_timeframe_mapping():
    """Map MT5 timeframe constants to our database timeframe IDs"""
    timeframe_map = {
        mt5.TIMEFRAME_M1: 1,    # 1 minute
        mt5.TIMEFRAME_M5: 2,    # 5 minutes
        mt5.TIMEFRAME_M15: 3,   # 15 minutes
        mt5.TIMEFRAME_M30: 4,   # 30 minutes
        mt5.TIMEFRAME_H1: 5,    # 1 hour
        mt5.TIMEFRAME_H4: 6,    # 4 hours
        mt5.TIMEFRAME_D1: 7,    # 1 day
    }
    return timeframe_map


# Define MT5 timeframes to fetch
def get_mt5_timeframes():
    """Return a list of all MT5 timeframes we want to fetch"""
    return [
        mt5.TIMEFRAME_M1,   # 1 minute
        mt5.TIMEFRAME_M5,   # 5 minutes
        mt5.TIMEFRAME_M15,  # 15 minutes
        mt5.TIMEFRAME_M30,  # 30 minutes
        mt5.TIMEFRAME_H1,   # 1 hour
        mt5.TIMEFRAME_H4,   # 4 hours
        mt5.TIMEFRAME_D1,   # 1 day
    ]


# Get human readable timeframe name
def get_timeframe_name(timeframe):
    """Convert MT5 timeframe constant to human-readable name"""
    timeframe_names = {
        mt5.TIMEFRAME_M1: "1 Minute",
        mt5.TIMEFRAME_M5: "5 Minutes",
        mt5.TIMEFRAME_M15: "15 Minutes",
        mt5.TIMEFRAME_M30: "30 Minutes",
        mt5.TIMEFRAME_H1: "1 Hour",
        mt5.TIMEFRAME_H4: "4 Hours",
        mt5.TIMEFRAME_D1: "Daily",
    }
    
    return timeframe_names.get(timeframe, "Unknown Timeframe")
# get the timeframes in minutes from the timeframe id from the database
def get_timeframes_in_minutes():
    """Return a dictionary mapping timeframe IDs to their duration in minutes"""
    return {
        1: 1,    # 1 minute
        2: 5,    # 5 minutes
        3: 15,   # 15 minutes
        4: 30,   # 30 minutes
        5: 60,   # 1 hour
        6: 240,  # 4 hours
        7: 1440, # Daily
    }

class StockDataFetcher:
    """Class to handle fetching stock data from MT5 and storing in database"""
    def __init__(self, db_path=None):
        """
        Initialize the StockDataFetcher
        
        Args:
            db_path (str, optional): Path to the SQLite database. If None, uses default path.
        """
        # Set default database path if not provided
        if db_path is None:
            # Try to detect the proper path relative to the script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(os.path.dirname(script_dir))
            db_path = os.path.join(project_dir, 'Data', 'Storage', 'data.db')
            
            # If that doesn't exist, try the alternate path
            if not os.path.exists(db_path):
                db_path = os.path.join(project_dir, 'Data', 'data.db')
        
        # Connect to the database with timeout and enable connection sharing
        self.conn = sqlite3.connect(db_path, timeout=60.0)  # 60 seconds timeout for locked database
        # Enable WAL mode for better concurrency
        self.conn.execute('PRAGMA journal_mode=WAL')
        # Enable foreign keys
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.cursor = self.conn.cursor()
        
        # Ensure we have the necessary indexes for efficient duplicate checking
        self._ensure_indexes()
        
        # Initialize MT5 connection status
        self.mt5_initialized = False
    
    def _ensure_indexes(self):
        """Ensure that we have the necessary indexes for efficient operation"""
        try:
            # Check if the lookup index exists
            self.cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='index' AND tbl_name='stock_data' 
                AND name='idx_stock_data_lookup'
            """)
            
            if self.cursor.fetchone()[0] == 0:
                print("Creating composite index for faster lookups...")
                self.cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_data_lookup 
                    ON stock_data (stock_id, timeframe_id, timestamp)
                """)
                self.conn.commit()
            
            # Check if unique constraint exists
            self.cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='index' AND tbl_name='stock_data' 
                AND sql LIKE '%UNIQUE%stock_id%timeframe_id%timestamp%'
            """)
            
            if self.cursor.fetchone()[0] == 0:
                print("Adding unique constraint to prevent duplicates...")
                self.cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_data_unique 
                    ON stock_data (stock_id, timeframe_id, timestamp)
                """)
                self.conn.commit()
                
        except sqlite3.Error as e:
            print(f"Warning: Could not create indexes: {e}")
            # Continue even if we can't create indexes
    
    def check_db_connection(self):
        """Check if database connection is valid"""
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.cursor.fetchall()
            if tables:
                print(f"Connected to database. Found {len(tables)} tables.")
                return True
            return False
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
    
    def get_stock_id(self, symbol):
        """
        Get stock ID from database by symbol
        
        Args:
            symbol (str): The stock symbol to look up
            
        Returns:
            int or None: The stock_id if found, None otherwise
        """
        try:
            self.cursor.execute("SELECT stock_id FROM stocks WHERE symbol = ?", (symbol,))
            result = self.cursor.fetchone()
            if result:
                return result[0]
            else:
                print(f"Stock symbol '{symbol}' not found in database")
                return None
        except sqlite3.Error as e:
            print(f"Error getting stock ID: {e}")
            return None
    
    def get_last_timestamp(self, stock_id, timeframe_id):
        """
        Get the latest timestamp for a stock and timeframe in the database
        
        Args:
            stock_id (int): The stock ID
            timeframe_id (int): The timeframe ID
            
        Returns:
            datetime: The latest timestamp, or 2015-01-01 if no data exists
        """
        try:
            self.cursor.execute(
                "SELECT MAX(timestamp) FROM stock_data WHERE stock_id = ? AND timeframe_id = ?", 
                (stock_id, timeframe_id)
            )
            result = self.cursor.fetchone()
            if result and result[0]:
                # Convert string timestamp to datetime object
                return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            else:
                # If no data exists, return default start date
                return datetime(2015, 1, 1)
        except sqlite3.Error as e:
            print(f"Error getting last timestamp: {e}")
            return datetime(2015, 1, 1)
    
    def fetch_mt5_data(self, symbol, timeframe, from_date, to_date, max_retries=3, retry_delay=5):
        """
        Fetch data from MT5 with retry mechanism
        
        Args:
            symbol (str): The symbol to fetch data for
            timeframe (int): The MT5 timeframe constant
            from_date (datetime): Start date for data
            to_date (datetime): End date for data
            max_retries (int, optional): Number of retry attempts. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds. Defaults to 5.
            
        Returns:
            DataFrame: DataFrame with the fetched data, or empty DataFrame if failed
        """
        for attempt in range(max_retries):
            try:
                # Fetch data from MT5
                data = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
                
                if data is not None and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # if timeframe is 4h, return the data frame as is
                    if timeframe == mt5.TIMEFRAME_H4:
                        return df
                    
                    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
                    return df
                
                # If no data returned but no error, just return empty DataFrame
                if data is not None:
                    print(f"No data returned for {symbol}, {get_timeframe_name(timeframe)} between {from_date} and {to_date}")
                    return pd.DataFrame()
                
                print(f"Attempt {attempt+1}/{max_retries}: Failed to fetch data. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries}: Error fetching data: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        print(f"Failed to fetch data after {max_retries} attempts. Returning empty DataFrame.")
        return pd.DataFrame()
    def insert_stock_data(self, data_df, stock_id, timeframe_id, max_retries=5, retry_delay=1.0):
        """
        Insert data into the database with retry mechanism for database lock errors
        
        Args:
            data_df (DataFrame): DataFrame with the data to insert
            stock_id (int): The stock ID
            timeframe_id (int): The timeframe ID
            max_retries (int, optional): Maximum number of retries on lock errors. Defaults to 5.
            retry_delay (float, optional): Initial delay between retries in seconds. Defaults to 1.
            
        Returns:
            int: Number of records inserted
        """
        if data_df.empty:
            return 0
        
        # Prepare batches for insertion (for better performance)
        batch_size = 1000
        records = []
        inserted_count = 0
        
        for idx, row in data_df.iterrows():
            record = (
                stock_id,
                timeframe_id,
                idx.strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
                row['open'],  # open_price
                row['high'],  # high_price
                row['low'],   # low_price
                row['close'], # close_price
                row['tick_volume']  # volume
            )
            records.append(record)
            
            # Insert in batches for better performance
            if len(records) >= batch_size:
                inserted = self._insert_batch(records, max_retries, retry_delay)
                inserted_count += inserted
                records = []
        
        # Insert remaining records
        if records:
            inserted = self._insert_batch(records, max_retries, retry_delay)
            inserted_count += inserted
        
        return inserted_count    
    
    def _insert_batch(self, records, max_retries=5, retry_delay=1.0):
        """
        Insert a batch of records with retry logic for database locks
        
        Args:
            records (list): List of record tuples to insert
            max_retries (int): Maximum number of retries
            retry_delay (float): Initial delay between retries in seconds
            
        Returns:
            int: Number of records inserted
        """
        # If no records to insert, return early
        if not records:
            return 0
            
        for attempt in range(max_retries):
            try:
                # First check which records already exist in the database
                # Extract stock_id, timeframe_id, timestamp from records for checking
                check_data = [(r[0], r[1], r[2]) for r in records]
                
                # Use placeholders for the IN clause
                placeholders = ','.join(['(?, ?, ?)'] * len(check_data))
                check_params = [param for record in check_data for param in record]  # Flatten the list
                
                # Query to find existing records
                query = f"""
                    SELECT stock_id, timeframe_id, timestamp 
                    FROM stock_data 
                    WHERE (stock_id, timeframe_id, timestamp) IN ({placeholders})
                """
                
                self.cursor.execute(query, check_params)
                existing_records = set((r[0], r[1], r[2]) for r in self.cursor.fetchall())
                
                # Filter out records that already exist
                new_records = [r for r in records if (r[0], r[1], r[2]) not in existing_records]
                
                # If all records already exist, return early
                if not new_records:
                    print(f"All {len(records)} records already exist in the database. Skipping insertion.")
                    return 0
                
                # Insert only new records
                self.cursor.executemany(
                    "INSERT INTO stock_data (stock_id, timeframe_id, timestamp, open_price, high_price, low_price, close_price, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    new_records
                )
                self.conn.commit()
                
                # Report how many were inserted vs skipped
                skipped = len(records) - len(new_records)
                if skipped > 0:
                    print(f"Inserted {len(new_records)} new records, skipped {skipped} existing records")
                
                return len(new_records)
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    current_delay = retry_delay * (2 ** attempt) * (0.8 + 0.4 * np.random.random())
                    print(f"Database lock detected, retrying in {current_delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(current_delay)
                else:
                    print(f"Error inserting batch: {e}")
                    self.conn.rollback()
                    return 0
            except sqlite3.Error as e:
                print(f"Error inserting batch: {e}")
                self.conn.rollback()
                return 0
        
        print(f"Failed to insert batch after {max_retries} attempts")
        return 0
    
    def get_stock_data(self, symbol, start_date=None, end_date=None):
        """
        Fetch stock data from MetaTrader 5 for all timeframes and store in database.
        
        Args:
            symbol (str): The stock symbol to fetch data for (must exist in stocks table)
            start_date (datetime, optional): Start date for data fetching. Defaults to 2015-01-01.
            end_date (datetime, optional): End date for data fetching. Defaults to current date/time.
        
        Returns:
            dict: Summary of data fetched and stored
        """
        # Check database connection
        if not self.check_db_connection():
            return {"error": "Database connection failed"}
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime(2015, 1, 1)
        if end_date is None:
            end_date = datetime.now()
            print("Using current date/time as end date:", end_date)
            # convert to riyadh time zone
            #end_date = end_date.replace(tzinfo=None) - timedelta(hours=3)
        
        # Initialize MT5 if not already initialized
        if not self.mt5_initialized:
            if not mt5.initialize():
                return {"error": f"MT5 initialization failed: {mt5.last_error()}"}
            self.mt5_initialized = True
        
        # Get stock ID from database
        stock_id = self.get_stock_id(symbol)
        if stock_id is None:
            return {"error": f"Symbol {symbol} not found in stocks table"}
        
        # Prepare to collect results
        results = {"symbol": symbol, "stock_id": stock_id, "timeframes": {}}
        
        # Get timeframe mapping
        timeframe_map = get_timeframe_mapping()
        
        # Process each timeframe
        for mt5_timeframe in get_mt5_timeframes():
            # Map MT5 timeframe to our database ID
            timeframe_id = timeframe_map[mt5_timeframe]
            timeframe_name = get_timeframe_name(mt5_timeframe)
            
            print(f"\nProcessing {timeframe_name} data for {symbol}...")
            
            # Get the latest timestamp in our database for this stock and timeframe
            last_timestamp = self.get_last_timestamp(stock_id, timeframe_id)
            
            # Add a small buffer to avoid duplicates (1 minute)
            if last_timestamp > datetime(2015, 1, 1):
                #last_timestamp += timedelta(minutes=1)
                print(f"Last record in database: {last_timestamp}")
            else:
                print("No existing data found. Starting from", last_timestamp)
            
            # Use the later of our default start_date or the last timestamp in database
            # make last timestamp to last_timestamp + current timeframe duration
            timeframe_duration = get_timeframes_in_minutes().get(timeframe_id, 1)
            last_timestamp = last_timestamp + timedelta(minutes=timeframe_duration)
            
            from_date = max(start_date, last_timestamp)
            
            # Skip if we already have data up to the end date
            if from_date >= end_date:
                print(f"Already have {timeframe_name} data up to {end_date}. Skipping.")
                results["timeframes"][timeframe_name] = {"new_records": 0, "status": "up-to-date"}
                continue
            
            print(f"Fetching {timeframe_name} data from {from_date} to {end_date}")
            
            # Fetch the data
            df = self.fetch_mt5_data(symbol, mt5_timeframe, from_date, end_date)
            
            if df.empty:
                print(f"No new {timeframe_name} data available")
                results["timeframes"][timeframe_name] = {"new_records": 0, "status": "no new data"}
                continue
            
            print(f"Got {len(df)} new {timeframe_name} records")
            
            # Insert into database
            inserted = self.insert_stock_data(df, stock_id, timeframe_id)
            
            # Store results
            results["timeframes"][timeframe_name] = {
                "new_records": inserted,
                "status": "success" if inserted > 0 else "no new data"
            }
            
            print(f"Inserted {inserted} new {timeframe_name} records")
            
            # Sleep briefly to avoid overwhelming the MT5 API
            time.sleep(0.5)
        
        return results
    
    def update_recent_candles(self, symbol, num_candles=10):
        """
        Update the most recent candles for all timeframes to ensure data consistency with MT5.
        This is useful to fix any data discrepancies caused by broker adjustments,
        internet connectivity issues, or other factors.
        
        Args:
            symbol (str): The stock symbol to update data for
            num_candles (int, optional): Number of recent candles to update. Defaults to 10.
        
        Returns:
            dict: Summary of updates performed
        """
        # Check database connection
        if not self.check_db_connection():
            return {"error": "Database connection failed"}
        
        # Initialize MT5 if not already initialized
        if not self.mt5_initialized:
            if not mt5.initialize():
                return {"error": f"MT5 initialization failed: {mt5.last_error()}"}
            self.mt5_initialized = True
        
        # Get stock ID from database
        stock_id = self.get_stock_id(symbol)
        if stock_id is None:
            return {"error": f"Symbol {symbol} not found in stocks table"}
        
        # Prepare to collect results
        results = {"symbol": symbol, "stock_id": stock_id, "timeframes": {}}
        
        # Get timeframe mapping
        timeframe_map = get_timeframe_mapping()
        
        # Current time for the queries
        now = datetime.now()
        
        # Process each timeframe
        for mt5_timeframe in get_mt5_timeframes():
            # Map MT5 timeframe to our database ID
            timeframe_id = timeframe_map[mt5_timeframe]
            timeframe_name = get_timeframe_name(mt5_timeframe)
            
            print(f"\nUpdating recent {timeframe_name} data for {symbol}...")
            
            try:
                # Get the latest data from MT5
                # Using copy_rates_from_pos to get the most recent candles
                mt5_data = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
                
                if mt5_data is None or len(mt5_data) == 0:
                    print(f"No recent {timeframe_name} data available from MT5")
                    results["timeframes"][timeframe_name] = {
                        "status": "no data from MT5",
                        "updated": 0,
                        "inserted": 0
                    }
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(mt5_data)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                if 'dt' in df.columns:
                    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
                
                # Count updates and inserts
                updates = 0
                inserts = 0
                
                # Process each candle
                for idx, row in df.iterrows():
                    timestamp = row['time'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Check if this record exists in the database
                    self.cursor.execute(
                        "SELECT entry_id FROM stock_data WHERE stock_id = ? AND timeframe_id = ? AND timestamp = ?",
                        (stock_id, timeframe_id, timestamp)
                    )
                    existing = self.cursor.fetchone()
                    
                    record = (
                        row['open'],      # open_price
                        row['high'],      # high_price
                        row['low'],       # low_price
                        row['close'],     # close_price
                        row['tick_volume'], # volume
                        stock_id,
                        timeframe_id,
                        timestamp
                    )
                    
                    if existing:
                        # Update existing record
                        try:
                            self.cursor.execute(
                                """
                                UPDATE stock_data 
                                SET open_price = ?, high_price = ?, low_price = ?, close_price = ?, volume = ?
                                WHERE stock_id = ? AND timeframe_id = ? AND timestamp = ?
                                """,
                                record
                            )
                            updates += 1
                        except sqlite3.Error as e:
                            print(f"Error updating record: {e}")
                    else:
                        # Insert new record
                        try:
                            self.cursor.execute(
                                """
                                INSERT INTO stock_data 
                                (open_price, high_price, low_price, close_price, volume, stock_id, timeframe_id, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                record
                            )
                            inserts += 1
                        except sqlite3.Error as e:
                            print(f"Error inserting record: {e}")
                
                # Commit changes
                self.conn.commit()
                
                # Store results
                results["timeframes"][timeframe_name] = {
                    "status": "success",
                    "updated": updates,
                    "inserted": inserts
                }
                
                print(f"Updated {updates} and inserted {inserts} {timeframe_name} records")
                
            except Exception as e:
                print(f"Error processing {timeframe_name}: {e}")
                results["timeframes"][timeframe_name] = {
                    "status": "error",
                    "message": str(e)
                }
            
            # Sleep briefly to avoid overwhelming the MT5 API
            time.sleep(0.2)
        
        return results
    
    def close(self):
        """Close database and MT5 connections"""
        # Close database connection
        if self.conn:
            self.conn.close()
            print("Database connection closed")
        
        # Shutdown MT5 if initialized
        if self.mt5_initialized:
            mt5.shutdown()
            self.mt5_initialized = False
            print("MT5 connection closed")


def get_stock_data(symbol, start_date=None, end_date=None, db_path=None):
    """
    Convenience function to fetch stock data without managing the fetcher instance.
    
    Args:
        symbol (str): The stock symbol to fetch data for
        start_date (datetime, optional): Start date for data fetching. Defaults to 2015-01-01.
        end_date (datetime, optional): End date for data fetching. Defaults to current date/time.
        db_path (str, optional): Path to the database. Defaults to None (auto-detect).
    
    Returns:
        dict: Summary of data fetched and stored
    """
    fetcher = StockDataFetcher(db_path)
    try:
        results = fetcher.get_stock_data(symbol, start_date, end_date)
        return results
    finally:
        fetcher.close()


def update_recent_candles(symbol, num_candles=10, db_path=None):
    """
    Convenience function to update the most recent candles for all timeframes.
    
    Args:
        symbol (str): The stock symbol to update data for
        num_candles (int, optional): Number of recent candles to update. Defaults to 10.
        db_path (str, optional): Path to the database. Defaults to None (auto-detect).
    
    Returns:
        dict: Summary of updates performed
    """
    fetcher = StockDataFetcher(db_path)
    try:
        results = fetcher.update_recent_candles(symbol, num_candles)
        return results
    finally:
        fetcher.close()


if __name__ == "__main__":
    # Check if command line arguments are provided
    results = get_stock_data("XAUUSD", datetime(2015, 1, 1))
  
    # Print summary
    if "error" in results:
        print(f"Error: {results['error']}")
    
    print(f"\nSummary for {results['symbol']}:")
    for timeframe, data in results.get("timeframes", {}).items():
        if "new_records" in data:
            print(f"  {timeframe}: {data['new_records']} new records - {data['status']}")
        elif "updated" in data:
            print(f"  {timeframe}: {data['updated']} updated, {data['inserted']} inserted - {data['status']}")
        else:
            print(f"  {timeframe}: {data['status']}")
            
            
    results = update_recent_candles("XAUUSD", 10)