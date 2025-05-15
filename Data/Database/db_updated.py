# Database Handler Module
# 
# This module is used to connect to the database, create tables, and store/retrieve data.
# It provides functionality for managing stock data, patterns, clusters, news articles, 
# tweets, and user authentication.

#################################################################################
# IMPORTS
#################################################################################
import sys
import os
from pathlib import Path

# Third-party imports
import bcrypt
import json
import numpy as np
import pandas as pd
import sqlite3 as db
from datetime import datetime, timedelta

# Setup path
# Get the current working directory (where the notebook/script is running)
current_dir = Path(os.getcwd())
# Navigate to the 'main' folder (adjust if needed)
main_dir = str(current_dir.parent)  # If notebook is inside 'main'
# OR if notebook is outside 'main':
# main_dir = str(current_dir / "main")  # Assumes 'main' is a subfolder
sys.path.append(main_dir)

#################################################################################
# CONSTANTS
#################################################################################
companies = {
    1: "GOLD (XAUUSD)",
    2: "BTC (BTCUSD)",
    3: "APPL (AAPL)",
    4: "Amazon (AMZN)",
    5: "NVIDIA (NVDA)",
}

#################################################################################
# DATABASE CLASS
#################################################################################
class Database:
    """
    Database class to handle all database operations including connection,
    data storage, retrieval, and analysis.
    """
    def __init__(self, db_name='Data/Storage/data.db'):
        """
        Initialize the database connection.
        
        Args:
            db_name (str): Path to the SQLite database file
        """
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.connect()
     
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.connection = db.connect(self.db_name, check_same_thread=False)
            self.cursor = self.connection.cursor()
            print(f"Connected to SQLite database: {self.db_name}")
        except db.Error as e:
            print(f"Database connection error: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print(f"Closed connection to database")
    
    #############################################################################
    # DATABASE UTILITY FUNCTIONS
    #############################################################################
    def execute_query(self, query, params=None, commit=False):
        """
        Execute a SQL query and optionally commit the changes.
        
        Args:
            query (str): SQL query to execute
            params (tuple/list/dict, optional): Parameters for the query
            commit (bool): Whether to commit the changes
            
        Returns:
            cursor: Database cursor after executing the query
        """
        try:
            if params:
                result = self.cursor.execute(query, params)
            else:
                result = self.cursor.execute(query)
            
            if commit:
                self.connection.commit()
            
            return result
        except db.Error as e:
            print(f"Query execution error: {e}")
            print(f"Query: {query}")
            if params:
                print(f"Parameters: {params}")
            raise
    
    def fetch_all(self, query, params=None):
        """
        Execute a query and fetch all results.
        
        Args:
            query (str): SQL query to execute
            params (tuple/list/dict, optional): Parameters for the query
            
        Returns:
            list: List of query results
        """
        cursor = self.execute_query(query, params)
        return cursor.fetchall()
    
    def fetch_one(self, query, params=None):
        """
        Execute a query and fetch one result.
        
        Args:
            query (str): SQL query to execute
            params (tuple/list/dict, optional): Parameters for the query
            
        Returns:
            tuple: Query result
        """
        cursor = self.execute_query(query, params)
        return cursor.fetchone()
    
    def list_tables(self):
        """
        List all tables in the database.
        
        Returns:
            list: Table names
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        tables = self.fetch_all(query)
        return [table[0] for table in tables]
    
    def describe_table(self, table_name):
        """
        Get the schema for a table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            list: Column information
        """
        query = f"PRAGMA table_info({table_name})"
        return self.fetch_all(query)
    
    #############################################################################
    # AUTHENTICATION FUNCTIONS
    #############################################################################        
    def login(self, username, password):
        """
        Authenticate a user with the provided credentials.
        
        Args:
            username (str): The username to authenticate
            password (str): The password to verify
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        user = self.fetch_one(
            "SELECT password FROM users WHERE username = ?", 
            (username,)
        )
        
        if user:
            stored_password_hash = user[0]
            return bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8'))
        
        return False
    
    def get_user_email(self, username):
        """
        Get the email address for a given username.
        
        Args:
            username (str): The username to look up
            
        Returns:
            str: The user's email address
        """
        user = self.fetch_one(
            "SELECT email FROM users WHERE username = ?", 
            (username,)
        )
        
        return user[0] if user else None
    
    def get_user_id(self, username):
        """
        Get the user ID for a given username.
        
        Args:
            username (str): The username to look up
            
        Returns:
            int: The user's ID
        """
        user = self.fetch_one(
            "SELECT user_id FROM users WHERE username = ?", 
            (username,)
        )
        
        return user[0] if user else None
    
    def create_user(self, username, password, email, preferences=None):
        """
        Create a new user account.
        
        Args:
            username (str): Username for the new account
            password (str): Password for the new account
            email (str): Email address for the new account
            preferences (str, optional): User preferences as JSON string
            
        Returns:
            int: New user ID
        """
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        cursor = self.execute_query(
            "INSERT INTO users (username, password, email, preferences) VALUES (?, ?, ?, ?)",
            (username, hashed_password, email, preferences),
            commit=True
        )
        
        return cursor.lastrowid
    
    #############################################################################
    # STOCK DATA FUNCTIONS
    #############################################################################
    def get_stock_list(self):
        """
        Get list of all stocks in the database.
        
        Returns:
            DataFrame: All stocks
        """
        query = "SELECT * FROM stocks"
        stocks = self.fetch_all(query)
        columns = ['stock_id', 'symbol', 'name', 'description', 'sector', 'created_at']
        
        return pd.DataFrame(stocks, columns=columns)
    
    def get_timeframes(self):
        """
        Get all available timeframes.
        
        Returns:
            DataFrame: All timeframes
        """
        query = "SELECT * FROM timeframes"
        timeframes = self.fetch_all(query)
        columns = ['timeframe_id', 'minutes', 'name', 'description']
        
        return pd.DataFrame(timeframes, columns=columns)
    
    def get_stock_data(self, stock_id, timeframe_id, limit=None):
        """
        Get stock data for a specific stock and timeframe.
        
        Args:
            stock_id (int): ID of the stock
            timeframe_id (int): ID of the timeframe
            limit (int, optional): Maximum number of records to return
            
        Returns:
            DataFrame: Stock data
        """
        query = """
            SELECT * FROM stock_data 
            WHERE stock_id = ? AND timeframe_id = ?
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        stock_data = self.fetch_all(query, (stock_id, timeframe_id))
        columns = ['entry_id', 'stock_id', 'timeframe_id', 'timestamp', 'open_price', 
                   'high_price', 'low_price', 'close_price', 'volume']
        
        df = pd.DataFrame(stock_data, columns=columns)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def get_stock_data_range(self, stock_id, timeframe_id, start_date, end_date):
        """
        Get stock data for a specific date range.
        
        Args:
            stock_id (int): ID of the stock
            timeframe_id (int): ID of the timeframe
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            DataFrame: Stock data within the specified range
        """
        query = """
            SELECT * FROM stock_data 
            WHERE stock_id = ? AND timeframe_id = ?
                AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        
        stock_data = self.fetch_all(query, (stock_id, timeframe_id, start_date, end_date))
        columns = ['entry_id', 'stock_id', 'timeframe_id', 'timestamp', 'open_price', 
                   'high_price', 'low_price', 'close_price', 'volume']
        
        df = pd.DataFrame(stock_data, columns=columns)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def store_stock_data(self, stock_data, stock_id, timeframe_id):
        """
        Store stock data in the database.
        
        Args:
            stock_data (DataFrame): Stock data to store
            stock_id (int): ID of the stock
            timeframe_id (int): ID of the timeframe
            
        Returns:
            int: Number of records inserted
        """
        count = 0
        for index, row in stock_data.iterrows():
            timestamp = index.strftime('%Y-%m-%d %H:%M:%S')
            
            self.execute_query(
                """
                INSERT INTO stock_data (stock_id, timeframe_id, timestamp, open_price, 
                                       high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stock_id, timeframe_id, timestamp, row['Open'], row['High'], 
                 row['Low'], row['Close'], row.get('Volume', 0)),
                commit=False
            )
            count += 1
        
        self.connection.commit()
        print(f"Stored {count} records for stock_id: {stock_id}, timeframe_id: {timeframe_id}")
        return count
    
    #############################################################################
    # PATTERN FUNCTIONS
    #############################################################################
    def get_patterns(self, stock_id=None, timeframe_id=None, limit=None):
        """
        Get patterns from the database with optional filtering.
        
        Args:
            stock_id (int, optional): Filter by stock ID
            timeframe_id (int, optional): Filter by timeframe ID
            limit (int, optional): Maximum number of patterns to return
            
        Returns:
            DataFrame: Patterns matching the criteria
        """
        query = "SELECT * FROM patterns"
        params = []
        conditions = []
        
        if stock_id is not None:
            conditions.append("stock_id = ?")
            params.append(stock_id)
        
        if timeframe_id is not None:
            conditions.append("timeframe_id = ?")
            params.append(timeframe_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        patterns = self.fetch_all(query, params if params else None)
        columns = ['pattern_id', 'stock_id', 'timeframe_id', 'config_id', 'cluster_id',
                   'price_points_json', 'volume_data', 'market_condition', 'outcome',
                   'max_gain', 'max_drawdown', 'reward_risk_ratio']
        
        df = pd.DataFrame(patterns, columns=columns)
        
        # Convert JSON strings to Python objects
        if not df.empty and 'price_points_json' in df.columns:
            df['price_points'] = df['price_points_json'].apply(lambda x: json.loads(x) if x else None)
        
        return df
    
    def store_pattern(self, pattern_data):
        """
        Store a pattern in the database.
        
        Args:
            pattern_data (dict): Pattern data including:
                - stock_id: ID of the stock
                - timeframe_id: ID of the timeframe
                - config_id: ID of the experiment configuration
                - price_points: List of price points
                - market_condition: Market condition (Bullish, Bearish, Neutral)
                - outcome: Pattern outcome
                - max_gain: Maximum gain
                - max_drawdown: Maximum drawdown
                
        Returns:
            int: New pattern ID
        """
        # Convert price points to JSON if needed
        if 'price_points' in pattern_data and not isinstance(pattern_data['price_points'], str):
            pattern_data['price_points_json'] = json.dumps(pattern_data['price_points'])
        
        # Prepare query and parameters
        columns = []
        placeholders = []
        values = []
        
        for key, value in pattern_data.items():
            if key != 'price_points':  # Skip the Python object version
                columns.append(key)
                placeholders.append('?')
                values.append(value)
        
        query = f"""
            INSERT INTO patterns ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor = self.execute_query(query, values, commit=True)
        return cursor.lastrowid
    
    def store_patterns_batch(self, patterns, stock_id, timeframe_id, config_id, pip_pattern_miner):
        """
        Store multiple patterns in a batch.
        
        Args:
            patterns (list): List of pattern data
            stock_id (int): ID of the stock
            timeframe_id (int): ID of the timeframe
            config_id (int): ID of the configuration
            pip_pattern_miner: Pattern miner object with additional data
            
        Returns:
            int: Number of patterns stored
        """
        count = 0
        
        for i, pattern in enumerate(pip_pattern_miner._unique_pip_patterns):
            # Convert the pattern to JSON
            price_points_json = json.dumps(pattern)
            
            # Determine market condition
            first_point = pattern[0]
            last_point = pattern[-1]
            
            if first_point < last_point:
                market_condition = 'Bullish'
            elif first_point > last_point:
                market_condition = 'Bearish'
            else:
                market_condition = 'Neutral'
            
            # Get pattern return and determine outcome metrics
            outcome = pip_pattern_miner._returns_fixed_hold[i]
            
            # Calculate max gain and drawdown
            max_gain = pip_pattern_miner._returns_mfe[i]
            max_drawdown = pip_pattern_miner._returns_mae[i]
            
            # Calculate reward/risk ratio
            reward_risk_ratio = abs(max_gain) / (abs(max_drawdown) + 1e-10)  # Avoid division by zero
            
            # Insert the pattern
            self.execute_query(
                """
                INSERT INTO patterns (
                    pattern_id, stock_id, timeframe_id, config_id, 
                    price_points_json, market_condition, outcome,
                    max_gain, max_drawdown, reward_risk_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (i, stock_id, timeframe_id, config_id, price_points_json, 
                 market_condition, outcome, max_gain, max_drawdown, reward_risk_ratio),
                commit=False
            )
            count += 1
        
        self.connection.commit()
        print(f"Stored {count} patterns for stock_id: {stock_id}, timeframe_id: {timeframe_id}")
        return count
    
    #############################################################################
    # CLUSTER FUNCTIONS
    #############################################################################
    def get_clusters(self, stock_id=None, timeframe_id=None, limit=None):
        """
        Get clusters from the database with optional filtering.
        
        Args:
            stock_id (int, optional): Filter by stock ID
            timeframe_id (int, optional): Filter by timeframe ID
            limit (int, optional): Maximum number of clusters to return
            
        Returns:
            DataFrame: Clusters matching the criteria
        """
        query = "SELECT * FROM clusters"
        params = []
        conditions = []
        
        if stock_id is not None:
            conditions.append("stock_id = ?")
            params.append(stock_id)
        
        if timeframe_id is not None:
            conditions.append("timeframe_id = ?")
            params.append(timeframe_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        clusters = self.fetch_all(query, params if params else None)
        columns = ['cluster_id', 'stock_id', 'timeframe_id', 'config_id', 'description', 
                   'avg_price_points_json', 'avg_volume_data', 'market_condition', 'outcome',
                   'label', 'probability_score', 'pattern_count', 'max_gain', 'max_drawdown',
                   'reward_risk_ratio', 'profit_factor', 'created_at']
        
        df = pd.DataFrame(clusters, columns=columns)
        
        # Convert JSON strings to Python objects
        if not df.empty and 'avg_price_points_json' in df.columns:
            df['avg_price_points'] = df['avg_price_points_json'].apply(lambda x: json.loads(x) if x else None)
        
        # Calculate additional metrics if not present
        if not df.empty:
            # Ensure these columns exist (for backward compatibility)
            if 'reward_risk_ratio' not in df.columns:
                df['reward_risk_ratio'] = abs(df['max_gain']) / (abs(df['max_drawdown']) + 1e-10)
            
            if 'profit_factor' not in df.columns:
                df['profit_factor'] = (df['probability_score'] * df['reward_risk_ratio']) / (1 - df['probability_score'] + 1e-10)
        
        return df
    
    def store_cluster(self, cluster_data):
        """
        Store a cluster in the database.
        
        Args:
            cluster_data (dict): Cluster data including:
                - stock_id: ID of the stock
                - timeframe_id: ID of the timeframe
                - config_id: ID of the experiment configuration
                - avg_price_points: List of average price points
                - market_condition: Market condition (Bullish, Bearish, Neutral)
                - outcome: Cluster outcome
                - label: Cluster label (Buy, Sell, Neutral)
                - probability_score: Probability score
                - pattern_count: Number of patterns in the cluster
                - max_gain: Maximum gain
                - max_drawdown: Maximum drawdown
                
        Returns:
            int: New cluster ID
        """
        # Convert price points to JSON if needed
        if 'avg_price_points' in cluster_data and not isinstance(cluster_data['avg_price_points'], str):
            cluster_data['avg_price_points_json'] = json.dumps(cluster_data['avg_price_points'])
        
        # Prepare query and parameters
        columns = []
        placeholders = []
        values = []
        
        for key, value in cluster_data.items():
            if key != 'avg_price_points':  # Skip the Python object version
                columns.append(key)
                placeholders.append('?')
                values.append(value)
        
        query = f"""
            INSERT INTO clusters ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor = self.execute_query(query, values, commit=True)
        return cursor.lastrowid
    
    def store_clusters_batch(self, clusters, stock_id, timeframe_id, config_id, pip_pattern_miner):
        """
        Store multiple clusters in a batch.
        
        Args:
            clusters (list): List of cluster data
            stock_id (int): ID of the stock
            timeframe_id (int): ID of the timeframe
            config_id (int): ID of the configuration
            pip_pattern_miner: Pattern miner object with additional data
            
        Returns:
            int: Number of clusters stored
        """
        count = 0
        
        for i, cluster in enumerate(pip_pattern_miner._cluster_centers):
            # Convert the cluster to JSON
            avg_price_points_json = json.dumps(cluster)
            
            # Determine market condition
            first_point = cluster[0]
            last_point = cluster[-1]
            
            if first_point < last_point:
                market_condition = 'Bullish'
            elif first_point > last_point:
                market_condition = 'Bearish'
            else:
                market_condition = 'Neutral'
            
            # Get cluster return and label
            outcome = pip_pattern_miner._cluster_returns[i]
            
            if outcome > 0:
                label = 'Buy'
            elif outcome < 0:
                label = 'Sell'
            else:
                label = 'Neutral'
            
            # Pattern count
            pattern_count = len(pip_pattern_miner._pip_clusters[i])
            
            # Calculate metrics
            max_gain = pip_pattern_miner._cluster_mfe[i]
            max_drawdown = pip_pattern_miner._cluster_mae[i]
            
            # Calculate probability score, reward/risk ratio, and profit factor
            probability_score = self.calculate_cluster_probability_score(
                stock_id, i, timeframe_id, config_id, label, pip_pattern_miner
            )
            
            reward_risk_ratio = abs(max_gain) / (abs(max_drawdown) + 1e-10)
            profit_factor = (probability_score * reward_risk_ratio) / (1 - probability_score + 1e-10)
            
            # Insert the cluster
            self.execute_query(
                """
                INSERT INTO clusters (
                    cluster_id, stock_id, timeframe_id, config_id,
                    avg_price_points_json, market_condition, outcome, label,
                    probability_score, pattern_count, max_gain, max_drawdown,
                    reward_risk_ratio, profit_factor, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (i, stock_id, timeframe_id, config_id, avg_price_points_json, 
                 market_condition, outcome, label, probability_score, pattern_count,
                 max_gain, max_drawdown, reward_risk_ratio, profit_factor),
                commit=False
            )
            count += 1
        
        self.connection.commit()
        print(f"Stored {count} clusters for stock_id: {stock_id}, timeframe_id: {timeframe_id}")
        return count
    
    def bind_patterns_to_clusters(self, stock_id, timeframe_id, config_id, pip_pattern_miner):
        """
        Update pattern records with their cluster IDs.
        
        Args:
            stock_id (int): ID of the stock
            timeframe_id (int): ID of the timeframe
            config_id (int): ID of the configuration
            pip_pattern_miner: Pattern miner object with cluster assignments
        """
        for cluster_id, pattern_ids in enumerate(pip_pattern_miner._pip_clusters):
            for pattern_id in pattern_ids:
                self.execute_query(
                    """
                    UPDATE patterns 
                    SET cluster_id = ? 
                    WHERE pattern_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
                    """,
                    (cluster_id, pattern_id, stock_id, timeframe_id, config_id),
                    commit=False
                )
            
            # Update pattern count in clusters table
            self.execute_query(
                """
                UPDATE clusters 
                SET pattern_count = ? 
                WHERE cluster_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
                """,
                (len(pattern_ids), cluster_id, stock_id, timeframe_id, config_id),
                commit=False
            )
        
        self.connection.commit()
        print(f"Updated cluster assignments for patterns")
    
    def calculate_cluster_probability_score(self, stock_id, cluster_id, timeframe_id, config_id, label, pip_pattern_miner=None):
        """
        Calculate probability score for a cluster based on its patterns.
        
        Args:
            stock_id (int): ID of the stock
            cluster_id (int): ID of the cluster
            timeframe_id (int): ID of the timeframe
            config_id (int): ID of the configuration
            label (str): Cluster label (Buy, Sell, Neutral)
            pip_pattern_miner: Pattern miner object (optional)
            
        Returns:
            float: Probability score (0-1)
        """
        if pip_pattern_miner:
            # Calculate directly from pattern miner
            pattern_indices = pip_pattern_miner._pip_clusters[cluster_id]
            outcomes = [pip_pattern_miner._returns_fixed_hold[i] for i in pattern_indices]
            
            total_patterns = len(outcomes)
            if total_patterns == 0:
                return 0.5
            
            if label == 'Buy':
                positive_outcomes = sum(1 for outcome in outcomes if outcome > 0)
                return positive_outcomes / total_patterns
            elif label == 'Sell':
                negative_outcomes = sum(1 for outcome in outcomes if outcome < 0)
                return negative_outcomes / total_patterns
            else:
                return 0.5
        else:
            # Retrieve from database
            query = """
                SELECT outcome FROM patterns 
                WHERE cluster_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
            """
            pattern_outcomes = self.fetch_all(query, (cluster_id, stock_id, timeframe_id, config_id))
            
            total_patterns = len(pattern_outcomes)
            if total_patterns == 0:
                return 0.5
            
            if label == 'Buy':
                positive_outcomes = sum(1 for outcome in pattern_outcomes if outcome[0] > 0)
                return positive_outcomes / total_patterns
            elif label == 'Sell':
                negative_outcomes = sum(1 for outcome in pattern_outcomes if outcome[0] < 0)
                return negative_outcomes / total_patterns
            else:
                return 0.5
    
    def update_cluster_probability_score(self, stock_id, cluster_id, timeframe_id, config_id):
        """
        Update probability score for a cluster in the database.
        
        Args:
            stock_id (int): ID of the stock
            cluster_id (int): ID of the cluster
            timeframe_id (int): ID of the timeframe
            config_id (int): ID of the configuration
            
        Returns:
            float: Updated probability score
        """
        # First get the cluster label
        cluster = self.fetch_one(
            """
            SELECT label FROM clusters 
            WHERE cluster_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
            """, 
            (cluster_id, stock_id, timeframe_id, config_id)
        )
        
        if not cluster:
            return 0.5
        
        label = cluster[0]
        
        # Calculate probability score
        probability_score = self.calculate_cluster_probability_score(
            stock_id, cluster_id, timeframe_id, config_id, label
        )
        
        # Update the database
        self.execute_query(
            """
            UPDATE clusters SET probability_score = ? 
            WHERE cluster_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
            """,
            (probability_score, cluster_id, stock_id, timeframe_id, config_id),
            commit=True
        )
        
        return probability_score
    
    #############################################################################
    # PREDICTION FUNCTIONS
    #############################################################################
    def get_predictions(self, stock_id=None, limit=None):
        """
        Get predictions from the database.
        
        Args:
            stock_id (int, optional): Filter by stock ID
            limit (int, optional): Maximum number of predictions to return
            
        Returns:
            DataFrame: Predictions matching the criteria
        """
        query = "SELECT * FROM predictions"
        params = []
        
        if stock_id is not None:
            query += " WHERE stock_id = ?"
            params.append(stock_id)
        
        query += " ORDER BY prediction_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        predictions = self.fetch_all(query, params if params else None)
        columns = ['prediction_id', 'stock_id', 'pattern_id', 'timeframe_id', 'config_id', 
                   'prediction_date', 'predicted_outcome', 'confidence_level', 
                   'sentiment_data_id', 'prediction_metrics']
        
        df = pd.DataFrame(predictions, columns=columns)
        
        # Parse JSON columns
        if not df.empty:
            for col in ['predicted_outcome', 'prediction_metrics']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if x else None)
        
        return df
    
    def store_prediction(self, prediction_data):
        """
        Store a prediction in the database.
        
        Args:
            prediction_data (dict): Prediction data including:
                - stock_id: ID of the stock
                - pattern_id: ID of the pattern
                - timeframe_id: ID of the timeframe
                - config_id: ID of the configuration
                - prediction_date: Date of prediction
                - predicted_outcome: Detailed prediction results (JSON)
                - confidence_level: Confidence level (0-1)
                - sentiment_data_id: ID of related sentiment data
                - prediction_metrics: Additional metrics (JSON)
                
        Returns:
            int: New prediction ID
        """
        # Convert JSON objects to strings
        for key in ['predicted_outcome', 'prediction_metrics']:
            if key in prediction_data and isinstance(prediction_data[key], (dict, list)):
                prediction_data[key] = json.dumps(prediction_data[key])
        
        # Prepare query and parameters
        columns = []
        placeholders = []
        values = []
        
        for key, value in prediction_data.items():
            columns.append(key)
            placeholders.append('?')
            values.append(value)
        
        query = f"""
            INSERT INTO predictions ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor = self.execute_query(query, values, commit=True)
        return cursor.lastrowid
    
    #############################################################################
    # NOTIFICATION FUNCTIONS
    #############################################################################
    def get_notifications(self, user_id=None, limit=None):
        """
        Get notifications from the database.
        
        Args:
            user_id (int, optional): Filter by user ID
            limit (int, optional): Maximum number of notifications to return
            
        Returns:
            DataFrame: Notifications matching the criteria
        """
        query = "SELECT * FROM notifications"
        params = []
        
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY sent_time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        notifications = self.fetch_all(query, params if params else None)
        columns = ['notification_id', 'user_id', 'prediction_id', 'sent_time', 
                   'notification_type', 'status']
        
        return pd.DataFrame(notifications, columns=columns)
    
    def store_notification(self, notification_data):
        """
        Store a notification in the database.
        
        Args:
            notification_data (dict): Notification data including:
                - user_id: ID of the user
                - prediction_id: ID of the related prediction
                - sent_time: Time the notification was sent
                - notification_type: Type of notification
                - status: Status of the notification
                
        Returns:
            int: New notification ID
        """
        # Prepare query and parameters
        columns = []
        placeholders = []
        values = []
        
        for key, value in notification_data.items():
            columns.append(key)
            placeholders.append('?')
            values.append(value)
        
        query = f"""
            INSERT INTO notifications ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor = self.execute_query(query, values, commit=True)
        return cursor.lastrowid
    
    #############################################################################
    # SENTIMENT DATA FUNCTIONS
    #############################################################################    
    def get_articles(self, stock_id=None, start_date=None, end_date=None, limit=None):
        """
        Get articles from the database.
        
        Args:
            stock_id (int, optional): Filter by stock ID
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            limit (int, optional): Maximum number of articles to return
            
        Returns:
            DataFrame: Articles matching the criteria
        """
        query = "SELECT * FROM articles"
        params = []
        conditions = []
        
        if stock_id is not None:
            conditions.append("most_relevant_stock_id = ?")
            params.append(stock_id)
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        articles = self.fetch_all(query, params if params else None)
        columns = ['article_id', 'date', 'authors', 'source_domain', 'source_name', 
                   'title', 'summary', 'url', 'topics', 'overall_sentiment_label', 
                   'overall_sentiment_score', 'event_type', 'fetch_timestamp',
                   'most_relevant_stock_id', 'most_relevant_stock_sentiment_score',
                   'most_relevant_stock_sentiment_label']
        
        df = pd.DataFrame(articles, columns=columns)
        
        # Convert timestamps
        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'fetch_timestamp' in df.columns:
                df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])
            
            # Parse JSON columns
            if 'topics' in df.columns:
                df['topics'] = df['topics'].apply(lambda x: json.loads(x) if x and isinstance(x, str) else x)
        
        return df
    
    def store_article(self, article_data):
        """
        Store an article in the database.
        
        Args:
            article_data (dict): Article data including relevant fields
                
        Returns:
            int: New article ID
        """
        # Convert JSON objects to strings
        for key in ['ticker_sentiment']:
            if key in article_data and isinstance(article_data[key], (dict, list)):
                article_data[key] = json.dumps(article_data[key])
        
        # Prepare query and parameters
        columns = []
        placeholders = []
        values = []
        
        for key, value in article_data.items():
            columns.append(key)
            placeholders.append('?')
            values.append(value)
        
        query = f"""
            INSERT INTO articles ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        cursor = self.execute_query(query, values, commit=True)
        return cursor.lastrowid
        
    def get_tweets(self, stock_id=None, start_date=None, end_date=None, limit=None):
        """
        Get tweets from the database.
        
        Args:
            stock_id (int, optional): Filter by stock ID
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            limit (int, optional): Maximum number of tweets to return
            
        Returns:
            DataFrame: Tweets matching the criteria
        """
        query = "SELECT * FROM tweets"
        params = []
        conditions = []
        
        if stock_id is not None:
            conditions.append("stock_id = ?")
            params.append(stock_id)
        
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        tweets = self.fetch_all(query, params if params else None)
        columns = ['tweet_id', 'stock_id', 'author_username', 'author_name', 'author_followers',
                  'author_following', 'author_verified', 'author_blue_verified', 'tweet_text',
                  'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count',
                  'bookmark_count', 'lang', 'is_reply', 'is_quote', 'is_retweet', 'url',
                  'search_term', 'sentiment_label', 'sentiment_score', 'sentiment_magnitude',
                  'weighted_sentiment', 'collected_at']
        
        df = pd.DataFrame(tweets, columns=columns)
        
        # Convert timestamps
        if not df.empty:
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
            if 'collected_at' in df.columns:
                df['collected_at'] = pd.to_datetime(df['collected_at'])
                
        return df
        
       
    def store_tweet(self, tweet_data):
        """
        Store a tweet in the database.
        
        Args:
            tweet_data (dict): Tweet data including all relevant fields such as:
                - tweet_id: Unique identifier for the tweet
                - stock_id: ID of the related stock
                - author_username: Username of the tweet author
                - author_name: Display name of the tweet author
                - author_followers: Number of followers
                - author_following: Number of accounts following
                - author_verified: Whether the author is verified
                - author_blue_verified: Whether the author has Twitter Blue verification
                - tweet_text: Content of the tweet
                - created_at: When the tweet was created
                - retweet_count, reply_count, like_count, quote_count, bookmark_count: Engagement metrics
                - lang: Language of the tweet
                - is_reply, is_quote, is_retweet: Tweet type flags
                - url: URL to the tweet
                - search_term: Term used to find this tweet
                - sentiment_label, sentiment_score, sentiment_magnitude, weighted_sentiment: Sentiment analysis
                - collected_at: When the tweet was collected
                
        Returns:
            str: Tweet ID of the stored tweet
        """
        # Check if tweet already exists
        existing = self.fetch_one(
            "SELECT tweet_id FROM tweets WHERE tweet_id = ?", 
            (tweet_data.get('tweet_id'),)
        )
        
        if existing:
            return existing[0]
        
        # Prepare query and parameters
        columns = []
        placeholders = []
        values = []
        
        for key, value in tweet_data.items():
            columns.append(key)
            placeholders.append('?')
            values.append(value)
        
        query = f"""
            INSERT INTO tweets ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor = self.execute_query(query, values, commit=True)
        return tweet_data.get('tweet_id')
    def get_stock_sentiment(self, stock_id, start_date=None, end_date=None):
        """
        Get stock sentiment data aggregated by date.
        
        Args:
            stock_id (int): Stock ID to retrieve sentiment for
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame: Sentiment data by date
        """
        query = """
            SELECT sentiment_id, stock_id, date, twitter_sentiment_score, news_sentiment_score, 
                   combined_sentiment_score, sentiment_label, tweet_count, article_count
            FROM stock_sentiment
            WHERE stock_id = ?
        """
        
        params = [stock_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        sentiment_data = self.fetch_all(query, params)
        columns = ['sentiment_id', 'stock_id', 'date', 'twitter_sentiment_score', 'news_sentiment_score', 
                   'combined_sentiment_score', 'sentiment_label', 'tweet_count', 'article_count']
        
        df = pd.DataFrame(sentiment_data, columns=columns)
        
        # Convert date to datetime
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    #############################################################################
    # STATISTICS FUNCTIONS
    #############################################################################
    def get_statistics(self, stock_id=None, timeframe_id=None):
        """
        Get statistics about clusters and patterns.
        
        Args:
            stock_id (int, optional): Filter by stock ID
            timeframe_id (int, optional): Filter by timeframe ID
            
        Returns:
            dict: Statistics including total clusters, patterns, win rates, etc.
        """
        # Get clusters
        clusters = self.get_clusters(stock_id, timeframe_id)
        
        # Get patterns
        patterns = self.get_patterns(stock_id, timeframe_id)
        
        # Calculate statistics
        total_clusters = len(clusters)
        total_patterns = len(patterns)
        
        # Avoid division by zero
        avg_patterns_per_cluster = total_patterns / total_clusters if total_clusters > 0 else 0
        
        # Cluster statistics
        if not clusters.empty:
            avg_win_rate = clusters['probability_score'].mean() * 100 if 'probability_score' in clusters else 0
            avg_max_gain = clusters['max_gain'].mean() * 100 if 'max_gain' in clusters else 0
            avg_max_drawdown = clusters['max_drawdown'].mean() * 100 if 'max_drawdown' in clusters else 0
            avg_reward_risk_ratio = clusters['reward_risk_ratio'].mean() if 'reward_risk_ratio' in clusters else 0
            avg_profit_factor = clusters['profit_factor'].mean() if 'profit_factor' in clusters else 0
            
            # Best cluster
            if 'max_gain' in clusters and not clusters.empty:
                best_cluster_idx = clusters['max_gain'].idxmax()
                best_cluster_return = clusters.loc[best_cluster_idx, 'max_gain'] * 100
                best_cluster_reward_risk_ratio = clusters.loc[best_cluster_idx, 'reward_risk_ratio']
                best_cluster_profit_factor = clusters.loc[best_cluster_idx, 'profit_factor']
            else:
                best_cluster_return = 0
                best_cluster_reward_risk_ratio = 0
                best_cluster_profit_factor = 0
        else:
            avg_win_rate = 0
            avg_max_gain = 0
            avg_max_drawdown = 0
            avg_reward_risk_ratio = 0
            avg_profit_factor = 0
            best_cluster_return = 0
            best_cluster_reward_risk_ratio = 0
            best_cluster_profit_factor = 0
        
        # Format the results
        results = {
            "Total Clusters": total_clusters,
            "Total Patterns": total_patterns,
            "Avg Patterns/Cluster": round(avg_patterns_per_cluster, 1),
            "Avg Win Rate": f"{round(avg_win_rate, 2)}%",
            "Avg Max Gain": f"{round(avg_max_gain, 2)}%",
            "Avg Max Drawdown": f"{round(avg_max_drawdown, 2)}%",
            "Avg Reward/Risk Ratio": round(avg_reward_risk_ratio, 2),
            "Avg Profit Factor": round(avg_profit_factor, 2),
            "Best Cluster Return": f"+{round(best_cluster_return, 1)}%",
            "Best Cluster Reward/Risk Ratio": round(best_cluster_reward_risk_ratio, 2),
            "Best Cluster Profit Factor": round(best_cluster_profit_factor, 2),
        }
        
        return results

# Main function to test the database connection
if __name__ == '__main__':
    db = Database()
    tables = db.get_tweets(stock_id=1, limit=5)
    print(f"Database tables: {tables}")
    db.close()
