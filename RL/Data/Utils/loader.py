import sys
from pathlib import Path


import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from Data.Database.db import Database


def load_data_from_db(db_path="RL/Data/Storage/samples.db", table_name="dataset", timeframe_id=5):
    """
    Loads data from a SQLite database into a pandas DataFrame and applies preprocessing.
    
    Parameters:
        db_path (str): Path to the SQLite database file. If None, will search in common locations.
        table_name (str): Name of the table to query (default is 'dataset').
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded and processed data.
    """
    try:
        # Connect to database and load data
        conn = sqlite3.connect(db_path)
        
        # Read the data
        df = pd.read_sql_query(f"SELECT * FROM {table_name} where timeframe_id = {timeframe_id}", conn)
        conn.close()
        
        # Ensure datetime column exists and is proper datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

if __name__ == "__main__":
    # Example usage
    training_data = load_data_from_db()
    data_point = training_data.iloc[0]
    print(training_data.head(1))

