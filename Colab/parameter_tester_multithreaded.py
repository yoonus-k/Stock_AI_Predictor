#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Parameter Optimization Framework with Multi-Threading Support and GPU Acceleration

This script systematically tests different combinations of parameters
for pattern mining to determine the optimal settings for Gold across
different timeframes. It implements multi-threading with GPU acceleration
for improved performance.

The process:
1. Connects to the normalized database
2. For each timeframe, runs pattern mining with different parameter combinations in parallel
3. Evaluates cluster performance metrics using GPU acceleration when available
4. Collects results in memory rather than writing to the database during parallel execution
5. Stores all results in the database in a single batch at the end
6. Generates performance reports and visualizations

This implementation avoids database lock issues by collecting all data in memory during
multithreaded processing and only writing to the database after all computation is complete.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from datetime import datetime, timedelta
import time
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import threading
import sqlite3
import warnings
import functools

# GPU Support
import tensorflow as tf

# Configure GPU memory growth to avoid taking all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {physical_devices}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found, using CPU mode")

# Add a warnings attribute to numpy if it doesn't exist
if not hasattr(np, 'warnings'):
    np.warnings = warnings

print("NumPy warnings patch applied successfully")
# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent # Navigate up to project root
sys.path.append(str(project_root))

# Import custom modules
from pip_pattern_miner import Pattern_Miner
from parameter_tester import ParameterTester

# Parameter ranges to test
PARAM_RANGES = {
    'n_pips': [3, 4, 5, 6, 7, 8],
    'lookback': [12, 24, 36, 48],
    'hold_period': [3, 6, 12, 24],
}

# Fixed parameter for test
DISTANCE_MEASURE = 2  # Perpendicular distance selected as standard

# GPU-accelerated distance calculation
def calculate_distances_gpu(data_points, centroids):
    """Calculate distances between data points and centroids using GPU acceleration.
    
    Args:
        data_points: NumPy array of data points [n_samples, n_features]
        centroids: NumPy array of centroids [n_clusters, n_features]
        
    Returns:
        Distance matrix as NumPy array [n_samples, n_clusters]
    """
    # Convert to TensorFlow tensors
    data_points_tf = tf.convert_to_tensor(data_points, dtype=tf.float32)
    centroids_tf = tf.convert_to_tensor(centroids, dtype=tf.float32)
    
    # Compute distances (squared Euclidean distance)
    expanded_data = tf.expand_dims(data_points_tf, 1)  # Shape [n_samples, 1, n_features]
    expanded_centroids = tf.expand_dims(centroids_tf, 0)  # Shape [1, n_clusters, n_features]
    
    # Calculate squared distances
    distances = tf.reduce_sum(tf.square(expanded_data - expanded_centroids), axis=2)
    
    # Convert back to NumPy
    return distances.numpy()

def retry_on_db_lock(max_retries=5, initial_delay=0.1):
    """Decorator to retry a function on database lock error."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                        # Log the retry attempt
                        if hasattr(args[0], 'use_gpu') and args[0].use_gpu:
                            print(f"Database locked, retrying in {delay:.2f}s (GPU mode, attempt {attempt+1}/{max_retries})")
                        else:
                            print(f"Database locked, retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        # Exponential backoff
                        delay *= 2
                    else:
                        raise
            return None
        return wrapper
    return decorator

class MultiThreadedParameterTester(ParameterTester):
    """Class for testing different parameter combinations for pattern mining with multi-threading and GPU support.
    
    This implementation uses in-memory storage for thread results and performs a single database write at the end
    to avoid database locking issues that occur with concurrent writes.
    """
    def __init__(self, db_path=None, num_workers=None):
        """Initialize the parameter tester with multi-threading and GPU support.
        
        Args:
            db_path: Path to the SQLite database. If None, uses the default connection.
            num_workers: Number of worker threads to use. If None, uses CPU count - 1.
        """
        # Initialize the parent class
        super().__init__(db_path)
        
        # Store the db_path for thread-specific connections
        self.db_path = db_path
        
        # Thread-local storage for database connections (used only for reading)
        self.thread_local = threading.local()
        
        # Database lock for synchronizing access - set as reentrant lock
        self.db_lock = threading.RLock()
        
        # Prepare the database for concurrent access
        self._optimize_database_for_concurrency()
        
        # Check for GPU availability and adjust number of workers
        if len(physical_devices) > 0:
            # If GPU is available, use fewer threads since GPU will handle parallelization
            self.num_workers = 2 if num_workers is None else min(4, num_workers)
            print(f"GPU detected: Using {self.num_workers} worker threads with GPU acceleration")
        else:
            # If no GPU, use more threads for CPU parallelization
            self.num_workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
            print(f"Initialized with {self.num_workers} worker threads (CPU only)")
        
        # Progress tracking
        self.total_combinations = 0
        self.completed_combinations = 0
        
        # GPU flag
        self.use_gpu = len(physical_devices) > 0
          # In-memory storage for thread results
        self.results_lock = threading.RLock()
        self.cluster_results = {}  # {(stock_id, timeframe_id, config_id): [cluster_metrics]}
        self.pattern_results = {}  # {(stock_id, timeframe_id, config_id): [(cluster_id, pattern_data)]}
        self.config_results = {}   # {(stock_id, timeframe_id, n_pips, lookback, hold_period): config_id}

    def _optimize_database_for_concurrency(self):
        """Configure the SQLite database for better concurrency."""
        try:
            # Use a temporary connection to set global database parameters
            conn = sqlite3.connect(self.db_path)
            
            # Set the database to WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Set synchronous mode to NORMAL for better performance
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Increase cache size for better performance
            conn.execute("PRAGMA cache_size=-20000")  # ~20MB
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Set a long timeout for busy connections
            conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            
            # Optimize the database
            conn.execute("PRAGMA optimize")
            
            conn.commit()
            conn.close()
            print("Database optimized for concurrent access")
            
        except Exception as e:
            print(f"Error optimizing database: {e}")
            # Continue anyway
    def get_thread_db_connection(self):
        """Get a thread-specific database connection."""
        if not hasattr(self.thread_local, "connection"):
            # Create a new connection
            self.thread_local.connection = sqlite3.connect(self.db_path)
            
            # Enable WAL mode for better concurrency
            self.thread_local.connection.execute("PRAGMA journal_mode=WAL")
            
            # Set a higher timeout for busy connections (30 seconds)
            self.thread_local.connection.execute("PRAGMA busy_timeout=30000")
            
            # Setting synchronous mode to NORMAL for better performance with WAL
            self.thread_local.connection.execute("PRAGMA synchronous=NORMAL")
            
            # Set the cache size to improve performance
            self.thread_local.connection.execute("PRAGMA cache_size=-20000")  # Approx. 20MB cache
            
            # Enable extended result codes for better error messages
            self.thread_local.connection.execute("PRAGMA extended_result_codes=ON")
            
            # Use deferred transactions for better concurrency
            self.thread_local.connection.isolation_level = 'DEFERRED'
            
            # Allow multiple threads to use the connection
            self.thread_local.connection.execute("PRAGMA threads=4")
        
        return self.thread_local.connection
     

    def get_stock_data_thread_safe(self, conn, stock_id, timeframe_id, start_date=None, end_date=None):
        """Thread-safe version of get_stock_data using a dedicated connection.
        
        Args:
            conn: Thread-specific database connection
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            
        Returns:
            DataFrame: Stock data
        """
        # Base query
        query = """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM stock_data
            WHERE stock_id = ? AND timeframe_id = ?
        """
        params = [stock_id, timeframe_id]
        
        # Add date filters if provided
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        # Execute query and fetch data
        cursor = conn.cursor()
        cursor.execute(query, params)
        data = cursor.fetchall()
        
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    def register_experiment_config_thread_safe(self, conn, stock_id, timeframe_id, n_pips, lookback, hold_period, strategy_type=None):
        """Thread-safe version of register_experiment_config that stores config in memory rather than database.
        
        Args:
            conn: Thread-specific database connection (used only for checking existence)
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            n_pips: Number of PIPs
            lookback: Lookback window size
            hold_period: Hold period
            strategy_type: Type of hold period strategy ('timeframe', 'formula', None)
            
        Returns:
            config_id: ID of the registered configuration (temporary for in-memory storage)
        """
        description = (f"Pattern mining with {n_pips} pips, {lookback} bar lookback, "
                      f"{hold_period} bar hold, perpendicular distance")
        
        name = f"Config_P{n_pips}_L{lookback}_H{hold_period}_D{DISTANCE_MEASURE}"
        if strategy_type:
            name = f"{name}_{strategy_type}"
        
        # Check if configuration already exists in the database
        cursor = conn.cursor()
        result = cursor.execute(
            """SELECT config_id FROM experiment_configs 
               WHERE stock_id = ? AND timeframe_id = ? AND n_pips = ? AND lookback = ? AND 
               hold_period = ? AND distance_measure = ?""",
            (stock_id, timeframe_id, n_pips, lookback, hold_period, DISTANCE_MEASURE)
        ).fetchone()
        
        if result:
            return result[0]
        
        # Generate a unique config key
        config_key = (stock_id, timeframe_id, n_pips, lookback, hold_period)
        
        # Thread-safe update of the in-memory store
        with self.results_lock:
            # Check if this config key is already in our in-memory store
            if config_key in self.config_results:
                return self.config_results[config_key]['temp_id']
            
            # Generate a temporary config ID (negative to avoid conflicts with DB IDs)
            # We'll replace this with a real ID when we write to the database
            temp_config_id = -1000000 - len(self.config_results)
            
            # Store the configuration in memory
            self.config_results[config_key] = {
                'temp_id': temp_config_id,
                'name': name,
                'stock_id': stock_id,
                'timeframe_id': timeframe_id,
                'n_pips': n_pips,
                'lookback': lookback,
                'hold_period': hold_period,
                'returns_hold_period': hold_period,
                'distance_measure': DISTANCE_MEASURE,
                'description': description
            }
            
            return temp_config_id
    
    def force_close_db_connections(self):
        """Force close all database connections to release locks."""
        try:
            # Close all existing connections
            self.close()
            
            # Create a new connection with immediate mode to force other connections to close
            temp_conn = sqlite3.connect(self.db_path, timeout=1, isolation_level="IMMEDIATE")
            # Run optimize and clean the database
            temp_conn.execute("PRAGMA optimize")
            temp_conn.execute("PRAGMA wal_checkpoint(FULL)")
            temp_conn.close()
            
            print("Database connections forcibly closed and database optimized")
            return True
        except Exception as e:
            print(f"Error forcing database connections to close: {e}")
            return False  
          
    def store_cluster_metrics_thread_safe(self, conn, stock_id, timeframe_id, config_id, pip_miner):
        """Thread-safe version of store_cluster_metrics that stores metrics in memory rather than database.
        
        Args:
            conn: Thread-specific database connection (used only for retrieving metadata)
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration (can be a temporary ID for in-memory storage)
            pip_miner: Trained Pattern_Miner instance
            
        Returns:
            List of cluster metrics dictionaries
        """
        try:
            # Get stock symbol and timeframe name for descriptions
            cursor = conn.cursor()
            
            stock_symbol = cursor.execute(
                "SELECT symbol FROM stocks WHERE stock_id = ?", (stock_id,)
            ).fetchone()[0]
            
            timeframe_name = cursor.execute(
                "SELECT name FROM timeframes WHERE timeframe_id = ?", (timeframe_id,)
            ).fetchone()[0]
            
            # Collect cluster metrics
            cluster_metrics = []
            
            # Determine if we can use GPU for calculations
            use_gpu = self.use_gpu and hasattr(self, 'calculate_metrics_gpu')
        except Exception as e:
            print(f"Error initializing database connection or retrieving metadata: {e}")
            return None
    
    # Continue with the rest of the method...
        
        for i, cluster_center in enumerate(pip_miner._cluster_centers):
            # Get patterns in this cluster
            pattern_indices = pip_miner._pip_clusters[i]
            pattern_count = len(pattern_indices)
            
            if pattern_count < 4:
                continue
            
            # Use GPU-accelerated metrics calculation if available
            if use_gpu:
                try:
                    metrics = self.calculate_metrics_gpu(
                        pip_miner._returns_fixed_hold,
                        pattern_indices,
                        pip_miner._cluster_centers[i],
                        pip_miner._returns_mfe,
                        pip_miner._returns_mae
                    )
                    
                    if metrics is not None:
                        # Use GPU-calculated metrics
                        avg_outcome = metrics['avg_outcome']
                        label = metrics['label']
                        max_gain = metrics['max_gain']
                        max_drawdown = metrics['max_drawdown']
                        reward_risk_ratio = metrics['reward_risk_ratio']
                        probability_score_dir = metrics['probability_score_dir']
                        probability_score_stat = metrics['probability_score_stat']
                        profit_factor = metrics['profit_factor']
                    else:
                        # Fall back to CPU calculation
                        use_gpu = False
                        print("GPU calculation failed, falling back to CPU")
                except Exception as e:
                    use_gpu = False
                    print(f"Error using GPU acceleration: {e}, falling back to CPU")
            
            # CPU calculations if GPU is not available or failed
            if not use_gpu:
                # Calculate metrics using CPU
                outcomes = pip_miner._returns_fixed_hold[pattern_indices]
                avg_outcome = pip_miner._cluster_returns[i]
                
                if avg_outcome > 0:
                    label = 'Buy'
                elif avg_outcome < 0:
                    label = 'Sell'
                else:
                    label = 'Neutral'
                    
                # Calculate max gain and drawdown based on cluster label
                if label == 'Buy':
                    max_gain = pip_miner._cluster_mfe[i]
                    max_drawdown = pip_miner._cluster_mae[i]
                elif label == 'Sell':
                    max_gain = pip_miner._cluster_mae[i]
                    max_drawdown = pip_miner._cluster_mfe[i]            
                else:
                    max_gain = 0
                    max_drawdown = 0
                    
                # Calculate reward/risk ratio with capping to prevent unrealistically high values
                if abs(max_drawdown) < 0.0001:  # If drawdown is essentially zero
                    reward_risk_ratio = min(100.0, abs(max_gain) * 100) if max_gain != 0 else 1.0
                else:
                    raw_ratio = abs(max_gain) / abs(max_drawdown)
                    # Cap the reward/risk ratio at 100 to prevent unrealistic values
                    reward_risk_ratio = min(100.0, raw_ratio)
                              
                # Calculate probability score (confidence) based on consistency and directional clarity
                if pattern_count > 1:
                    # Calculate statistics
                    avg_outcome = np.mean(outcomes)
                    outcome_std = np.std(outcomes)
                    
                    # Get directional consistency (skew toward positive or negative)
                    pos_ratio = sum(1 for o in outcomes if o > 0) / len(outcomes)
                    neg_ratio = sum(1 for o in outcomes if o < 0) / len(outcomes)
                    direction_consistency = max(pos_ratio, neg_ratio)  # Higher value indicates stronger directional bias
                    
                    # Calculate variance relative to mean absolute outcome (more stable than just using avg_outcome)
                    mean_abs_outcome = np.mean(np.abs(outcomes))
                    relative_variance = min(1.0, outcome_std / (mean_abs_outcome + 0.001))
                    
                    # Combine both factors - higher score for strong direction and low relative variance
                    consistency = direction_consistency * (1 - relative_variance)
                    
                    # Ensure range is within 0.1 to 0.9 to avoid extreme scores
                    consistency = 0.1 + (consistency * 0.8)
                    
                    # Weight by pattern count (with cap based on average pattern count)
                    max_patterns_for_full_weight = pip_miner._avg_patterns_count
                    pattern_weight = min(1.0, pattern_count / max_patterns_for_full_weight)
                    
                    # Final probability score - balanced approach
                    probability_score_stat = consistency * pattern_weight
                else:
                    probability_score_stat = 0.5  # Default for single pattern clusters
                
                # calculate probability score directionally
                total_patterns = len(outcomes)
                if pattern_count == 0:
                   probability_score_dir = 0.5
                
                if label == 'Buy':
                    positive_outcomes = sum(1 for outcome in outcomes if outcome > 0)
                    probability_score_dir = positive_outcomes / total_patterns
                elif label == 'Sell':
                    negative_outcomes = sum(1 for outcome in outcomes if outcome < 0)               
                    probability_score_dir = negative_outcomes / total_patterns
                else:
                    probability_score_dir = 0.5
                
                # Calculate profit factor with reasonable limits
                if probability_score_dir >= 0.99:  # Nearly perfect win rate
                    profit_factor = reward_risk_ratio * 10  # Still reward good performance but cap it
                elif probability_score_dir <= 0.01:  # Nearly perfect loss rate
                    profit_factor = 0.01  # Very poor profit factor
                else:
                    # Standard calculation with reasonable upper limit
                    raw_profit_factor = (probability_score_dir * reward_risk_ratio) / (1 - probability_score_dir)
                    profit_factor = min(1000.0, raw_profit_factor)  # Cap at 1000 for database consistency
            
            # Create cluster description
            win_rate = probability_score_dir * 100 
            pattern_type = "Bullish Pattern" if label == 'Buy' else "Bearish Pattern" if label == 'Sell' else "Neutral Pattern"
            
            avg_gain_str = f"{avg_outcome:.2f}%" if avg_outcome != 0 else "N/A"
            max_gain_str = f"{max_gain:.2f}%" if max_gain != 0 else "0.0%"
            max_dd_str = f"{max_drawdown:.2f}%" if max_drawdown != 0 else "0.0%"
            rr_ratio_str = f"{reward_risk_ratio:.2f}" if reward_risk_ratio > 0 else "N/A"
            
            
            cluster_description = (
                f"Cluster #{i} (Stock ID: {stock_id} - {stock_symbol}) | {pattern_type} | Win Rate: {win_rate:.1f}%\n"
                f"Performance: Avg. Gain {avg_gain_str} | Max Gain {max_gain_str} | Max DD {max_dd_str} | R/R Ratio: {rr_ratio_str}\n"
                f"Contains {pattern_count} patterns from {timeframe_name} data"
            )
              # Store cluster in memory instead of database
            price_points_json = json.dumps(cluster_center.tolist() if isinstance(cluster_center, np.ndarray) else cluster_center)
            
            avg_volume = None
            
            # Use cluster index as temporary cluster ID
            cluster_id = i + 1  # Use 1-based indexing for clarity
            
            # Create cluster data dictionary
            cluster_data = {
                'cluster_id': cluster_id,
                'stock_id': stock_id,
                'timeframe_id': timeframe_id,
                'config_id': config_id,
                'avg_price_points_json': price_points_json,
                'avg_volume': avg_volume,
                'outcome': avg_outcome,
                'label': label,
                'probability_score_dir': probability_score_dir,
                'probability_score_stat': probability_score_stat,
                'pattern_count': pattern_count,
                'max_gain': max_gain,
                'max_drawdown': max_drawdown,
                'reward_risk_ratio': reward_risk_ratio,
                'profit_factor': profit_factor,
                'description': cluster_description,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Store in the in-memory collection (thread-safe)
            with self.results_lock:
                if (stock_id, timeframe_id, config_id) not in self.cluster_results:
                    self.cluster_results[(stock_id, timeframe_id, config_id)] = []
                self.cluster_results[(stock_id, timeframe_id, config_id)].append(cluster_data)              # Store metrics for return
            cluster_metrics.append({
                'cluster_id': cluster_id,
                'pattern_count': pattern_count,
                'outcome': avg_outcome,
                'max_gain': max_gain,
                'max_drawdown': max_drawdown,
                'reward_risk_ratio': reward_risk_ratio,
                'profit_factor': profit_factor,
                'probability_score': probability_score_dir,
                'label': label,
                'description': cluster_description
            })
        
        # No need to commit to database as we're storing in memory
        return cluster_metrics
    
    def calculate_metrics_gpu(self, outcomes, pattern_indices, cluster_centers, max_gain_values, max_drawdown_values):
        """Calculate cluster metrics using GPU acceleration.
        
        Args:
            outcomes: Array of pattern outcome values
            pattern_indices: List of pattern indices in this cluster
            cluster_centers: Array of cluster centers
            max_gain_values: Array of maximum gain values for patterns
            max_drawdown_values: Array of maximum drawdown values for patterns
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Convert to TensorFlow tensors
            outcomes_tf = tf.convert_to_tensor([outcomes[i] for i in pattern_indices], dtype=tf.float32)
            max_gain_tf = tf.convert_to_tensor([max_gain_values[i] for i in pattern_indices], dtype=tf.float32)
            max_dd_tf = tf.convert_to_tensor([max_drawdown_values[i] for i in pattern_indices], dtype=tf.float32)
            
            # Calculate statistics
            pattern_count = len(pattern_indices)
            avg_outcome = tf.reduce_mean(outcomes_tf).numpy()
            
            # Determine label based on average outcome
            if avg_outcome > 0:
                label = 'Buy'
                max_gain = tf.reduce_mean(max_gain_tf).numpy()
                max_drawdown = tf.reduce_mean(max_dd_tf).numpy()
            elif avg_outcome < 0:
                label = 'Sell'
                max_gain = tf.reduce_mean(max_dd_tf).numpy()  # For sell, max gain is from max drawdown
                max_drawdown = tf.reduce_mean(max_gain_tf).numpy()  # For sell, max drawdown is from max gain
            else:
                label = 'Neutral'
                max_gain = 0.0
                max_drawdown = 0.0
            
            # Calculate reward/risk ratio with GPU
            if abs(max_drawdown) < 0.0001:
                reward_risk_ratio = tf.constant(min(100.0, abs(max_gain) * 100) if max_gain != 0 else 1.0).numpy()
            else:
                raw_ratio = abs(max_gain) / abs(max_drawdown)
                reward_risk_ratio = tf.constant(min(100.0, raw_ratio)).numpy()
            
            # Calculate probability scores
            if label == 'Buy':
                positive_outcomes = tf.reduce_sum(tf.cast(outcomes_tf > 0, tf.float32)).numpy()
                probability_score_dir = positive_outcomes / pattern_count
            elif label == 'Sell':
                negative_outcomes = tf.reduce_sum(tf.cast(outcomes_tf < 0, tf.float32)).numpy()
                probability_score_dir = negative_outcomes / pattern_count
            else:
                probability_score_dir = 0.5
            
            # Calculate profit factor
            if probability_score_dir >= 0.99:
                profit_factor = reward_risk_ratio * 10
            elif probability_score_dir <= 0.01:
                profit_factor = 0.01
            else:
                raw_profit_factor = (probability_score_dir * reward_risk_ratio) / (1 - probability_score_dir)
                profit_factor = min(1000.0, raw_profit_factor)
            
            # Return metrics dictionary
            return {
                'pattern_count': pattern_count,
                'avg_outcome': avg_outcome,
                'label': label,
                'max_gain': max_gain,
                'max_drawdown': max_drawdown,
                'reward_risk_ratio': reward_risk_ratio,
                'probability_score_dir': probability_score_dir,
                'probability_score_stat': probability_score_dir,  # Using same value for simplicity
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            print(f"GPU calculation error: {e}")
            # Fall back to CPU calculation
            return None
    def store_patterns_thread_safe(self, conn, stock_id, timeframe_id, config_id, pip_miner):
        """Thread-safe version of store_patterns that stores patterns in memory rather than database.
        
        Args:
            conn: Thread-specific database connection (used only for format checking)
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration (can be a temporary ID for in-memory storage)
            pip_miner: Trained Pattern_Miner instance
        """
        # Initialize pattern list for this configuration if it doesn't exist
        with self.results_lock:
            if (stock_id, timeframe_id, config_id) not in self.pattern_results:
                self.pattern_results[(stock_id, timeframe_id, config_id)] = []
        
        # Loop through the clusters
        for j, cluster in enumerate(pip_miner._pip_clusters):
            # Use 1-based cluster IDs for consistency with other methods
            cluster_id = j + 1
            
            # loop through the patterns in the cluster
            for i, pattern_id in enumerate(cluster):
                pattern = pip_miner._unique_pip_patterns[pattern_id]
                
                # Convert pattern to JSON
                pattern_json = json.dumps(pattern)
                outcome = pip_miner._returns_fixed_hold[pattern_id]
                
                if outcome > 0:
                    pattern_label = 'Buy'
                elif outcome < 0:
                    pattern_label = 'Sell'
                else:
                    pattern_label = 'Neutral'
                
                # Calculate max gain and drawdown based on pattern label
                if pattern_label == 'Buy':
                    max_gain = pip_miner._returns_mfe[pattern_id] if pattern_id < len(pip_miner._returns_mfe) else 0
                    max_drawdown = pip_miner._returns_mae[pattern_id] if pattern_id < len(pip_miner._returns_mae) else 0
                elif pattern_label == 'Sell':
                    max_gain = pip_miner._returns_mae[pattern_id] if pattern_id < len(pip_miner._returns_mae) else 0
                    max_drawdown = pip_miner._returns_mfe[pattern_id] if pattern_id < len(pip_miner._returns_mfe) else 0
                else:
                    max_gain = 0
                    max_drawdown = 0
                
                # Store volume data if available
                volume_data = None
                if hasattr(pip_miner, '_volume_data') and pip_miner._volume_data is not None:
                    volume_data = json.dumps(pip_miner._volume_data[i].tolist())
                
                # Create pattern data dictionary
                pattern_data = {
                    'pattern_id': pattern_id,
                    'stock_id': stock_id,
                    'timeframe_id': timeframe_id,
                    'config_id': config_id,
                    'cluster_id': cluster_id,
                    'price_points_json': pattern_json,
                    'volume': volume_data,
                    'outcome': outcome,
                    'max_gain': max_gain,
                    'max_drawdown': max_drawdown,
                    'label': pattern_label,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Store in the in-memory collection (thread-safe)
                with self.results_lock:
                    self.pattern_results[(stock_id, timeframe_id, config_id)].append(pattern_data)
          # No database commit needed as we're storing in memory
  
    def close(self):
        """Close all database connections."""
        # Close parent connection
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'connection') and self.db.connection:
                self.db.connection.close()
                print("Closed parent database connection")
        except Exception as e:
            print(f"Error closing parent connection: {e}")
        
        # Close our main connection if it exists
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
                print("Closed main database connection")
        except Exception as e:
            print(f"Error closing main connection: {e}")
        
        # Close thread-specific connections if they exist
        closed_count = 0
        if hasattr(self, 'thread_local'):
            # Try to close thread_local connection directly
            try:
                if hasattr(self.thread_local, 'connection'):
                    self.thread_local.connection.close()
                    closed_count += 1
            except Exception as e:
                print(f"Error closing thread_local connection: {e}")
            
            # Also try to check all active threads for connections
            for thread_id, thread in threading._active.items():
                if hasattr(thread, '_thread_local') and hasattr(thread._thread_local, 'connection'):
                    try:
                        thread._thread_local.connection.close()
                        closed_count += 1
                    except Exception as e:
                        print(f"Error closing thread connection: {e}")
        
        print(f"Closed {closed_count} thread database connections")
        
    def write_all_results_to_database(self):
        """Write all collected in-memory results to the database in a single transaction.
        
        This method should be called after all threads have completed their processing
        to avoid database locks during parallel execution.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Writing all results to database...")
        
        try:
            # Create a direct connection to the database with immediate mode
            # to ensure we have exclusive access during the write operation
            conn = sqlite3.connect(self.db_path, isolation_level="IMMEDIATE")
            cursor = conn.cursor()
            
            # Get current timestamp for all insertions
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Dictionary to map temporary config IDs to real database IDs
            temp_to_real_config_ids = {}
            
            print(f"Writing {len(self.config_results)} configurations to database...")
            
            # 1. First, write all configs to the database and get real IDs
            for config_key, config_data in self.config_results.items():
                stock_id, timeframe_id, n_pips, lookback, hold_period = config_key
                temp_config_id = config_data['temp_id']
                
                # Check if this configuration already exists
                result = cursor.execute(
                    """SELECT config_id FROM experiment_configs 
                       WHERE stock_id = ? AND timeframe_id = ? AND n_pips = ? AND 
                       lookback = ? AND hold_period = ? AND distance_measure = ?""",
                    (stock_id, timeframe_id, n_pips, lookback, hold_period, DISTANCE_MEASURE)
                ).fetchone()
                
                if result:
                    # Use existing config ID
                    real_config_id = result[0]
                    
                    # Update the record to ensure consistency
                    cursor.execute(
                        """UPDATE experiment_configs SET
                           name = ?, description = ?, strategy_type = ?
                           WHERE config_id = ?""",
                        (config_data.get('name', ''), config_data.get('description', ''), 
                         config_data.get('strategy_type', 'timeframe'), real_config_id)
                    )
                else:
                    # Insert new config
                    cursor.execute(
                        """INSERT INTO experiment_configs 
                           (stock_id, timeframe_id, n_pips, lookback, hold_period, 
                            returns_hold_period, distance_measure, name, description, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (stock_id, timeframe_id, n_pips, lookback, hold_period,
                         hold_period, DISTANCE_MEASURE, config_data.get('name', ''),
                         config_data.get('description', ''), timestamp)
                    )
                    
                    # Get the new ID
                    real_config_id = cursor.lastrowid
                
                # Map temporary ID to real ID
                temp_to_real_config_ids[temp_config_id] = real_config_id
            
            print(f"Writing cluster results for {len(self.cluster_results)} configurations...")
            
            # 2. Write all clusters to database
            for key, clusters in self.cluster_results.items():
                stock_id, timeframe_id, temp_config_id = key
                
                # Get the real config ID
                config_id = temp_to_real_config_ids.get(temp_config_id)
                if config_id is None:
                    # Use the ID directly if it's not a temporary ID
                    config_id = temp_config_id
                
                # Clear existing clusters for this config to avoid conflicts
                cursor.execute(
                    "DELETE FROM clusters WHERE stock_id = ? AND timeframe_id = ? AND config_id = ?",
                    (stock_id, timeframe_id, config_id)
                )
                
                # Insert all clusters
                for cluster in clusters:
                    # Prepare cluster data with real config ID
                    cluster_data = cluster.copy()
                    cluster_data['config_id'] = config_id
                    
                    # Insert cluster
                    cursor.execute(
                        """INSERT INTO clusters 
                           (cluster_id,stock_id, timeframe_id, config_id, avg_price_points_json, avg_volume,
                            outcome, label, probability_score_dir, probability_score_stat, pattern_count,
                            max_gain, max_drawdown, reward_risk_ratio, profit_factor, description, created_at)
                           VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (cluster_data['cluster_id'],cluster_data['stock_id'], cluster_data['timeframe_id'], cluster_data['config_id'],
                         cluster_data['avg_price_points_json'], cluster_data.get('avg_volume'),
                         cluster_data['outcome'], cluster_data['label'],
                         cluster_data['probability_score_dir'], cluster_data['probability_score_stat'],
                         cluster_data['pattern_count'], cluster_data['max_gain'], 
                         cluster_data['max_drawdown'], cluster_data['reward_risk_ratio'],
                         cluster_data['profit_factor'], cluster_data['description'],
                         cluster_data.get('created_at', timestamp))
                    )
                    
                    # Store the real cluster ID for patterns
                    cluster_data['real_cluster_id'] = cursor.lastrowid
            
            print(f"Writing pattern results for {len(self.pattern_results)} configurations...")
            
            # 3. Write patterns to database
            for key, patterns in self.pattern_results.items():
                stock_id, timeframe_id, temp_config_id = key
                
                # Get the real config ID
                config_id = temp_to_real_config_ids.get(temp_config_id)
                if config_id is None:
                    # Use the ID directly if it's not a temporary ID
                    config_id = temp_config_id
                
                # Clear existing patterns for this config to avoid conflicts
                cursor.execute(
                    "DELETE FROM patterns WHERE stock_id = ? AND timeframe_id = ? AND config_id = ?",
                    (stock_id, timeframe_id, config_id)
                )
                
                # Insert all patterns
                for pattern in patterns:
                    # Prepare pattern data with real config ID
                    pattern_data = pattern.copy()
                    pattern_data['config_id'] = config_id
                    
             
                    # Insert pattern
                    cursor.execute(
                        """INSERT INTO patterns 
                           (pattern_id, stock_id, timeframe_id, config_id, cluster_id,
                            price_points_json, volume, outcome, max_gain, max_drawdown)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (pattern_data['pattern_id'], pattern_data['stock_id'], 
                         pattern_data['timeframe_id'], pattern_data['config_id'], 
                         pattern_data["cluster_id"], pattern_data['price_points_json'],
                         pattern_data.get('volume'), pattern_data['outcome'],                         pattern_data['max_gain'], pattern_data['max_drawdown'])
                    )
            
            # Commit the transaction
            conn.commit()
            
            # Clear in-memory data after successful write
            self.cluster_results.clear()
            self.pattern_results.clear()
            self.config_results.clear()
            
            print("Successfully wrote all results to database")
            
            # Optimize database after bulk insert
            conn.execute("PRAGMA optimize")
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.close()
            
            return True
            
        except sqlite3.Error as e:
            print(f"Database error writing results: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
        except Exception as e:
            print(f"Unexpected error writing results: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False

    def calculate_weighted_score(self, metrics):
        """
        Calculate weighted score based on cluster metrics.
        
        Args:
            metrics: List of cluster metrics
            
        Returns:
            float: Weighted score
        """
        if not metrics:
            return 0
        
        # Weights for different metrics
        weights = {
            'profit_factor': 0.35,
            'reward_risk_ratio': 0.25,
            'pattern_count': 0.20,
            'probability_score': 0.20
        }
        
        # Normalize pattern count (more is better, up to a point)
        max_pattern_count = max([m['pattern_count'] for m in metrics]) if metrics else 1
        normalized_pattern_counts = [min(1.0, m['pattern_count'] / 20) for m in metrics]
        
        # Calculate weighted score for each cluster
        cluster_scores = []
        for i, metric in enumerate(metrics):
            # Cap values to prevent excessive influence of outliers
            capped_profit_factor = min(5.0, metric['profit_factor'])
            capped_reward_risk_ratio = min(5.0, metric['reward_risk_ratio'])
            
            # Calculate weighted score using capped values
            score = (
                weights['profit_factor'] * capped_profit_factor / 5.0 +
                weights['reward_risk_ratio'] * capped_reward_risk_ratio / 5.0 +
                weights['pattern_count'] * normalized_pattern_counts[i] +
                weights['probability_score'] * metric['probability_score']
            )
            cluster_scores.append(score)
        
        # Return average of top 3 cluster scores, weighted by pattern count
        if len(cluster_scores) >= 3:
            top_indices = np.argsort(cluster_scores)[-3:]
            top_metrics = [metrics[i] for i in top_indices]
            weights = [m['pattern_count'] for m in top_metrics]
            return np.average([cluster_scores[i] for i in top_indices], weights=weights)
        else:
            return np.mean(cluster_scores) if cluster_scores else 0

    def get_timeframe_category(self, timeframe_id):
        """
        Determine the category of a timeframe (lower, medium, higher).
        
        Args:
            timeframe_id: ID of the timeframe
            
        Returns:
            str: Category of the timeframe ('lower', 'medium', 'higher')
        """
        conn = self.get_thread_db_connection()
        timeframe_info = conn.execute(
            "SELECT minutes, name FROM timeframes WHERE timeframe_id = ?", 
            (timeframe_id,)
        ).fetchone()
        
        if timeframe_info:
            minutes, name = timeframe_info
            
            if "1min" in name or "5min" in name or minutes <= 5:
                return "lower"
            elif "15min" in name or "30min" in name or "1h" in name or 5 < minutes <= 60:
                return "medium"
            else:  # 4h, Daily or higher
                return "higher"
        
        return "medium"  # Default if timeframe not found

    def get_hold_periods_for_timeframe(self, timeframe_id):
        """
        Get appropriate hold period values for a specific timeframe.
        
        Args:
            timeframe_id: ID of the timeframe
            
        Returns:
            list: Appropriate hold period values for this timeframe
        """
        category = self.get_timeframe_category(timeframe_id)
        
        if category == "lower":
            return [3, 6]
        elif category == "medium":
            return [6, 12]
        else:  # higher
            return [12, 24]

    def get_formula_based_hold_period(self, lookback):
        """
        Calculate hold period based on lookback period.
        
        Args:
            lookback: Lookback window size
            
        Returns:
            int: Calculated hold period
        """
        return max(3, int(lookback / 4))

    def run_parameter_test(self, stock_id, timeframe_id, start_date=None, end_date=None, 
                          hold_period_strategy="timeframe", test_all=False, single_test=False):
        """Run parameter testing for a specific stock and timeframe with multi-threading support.
        
        Args:
            stock_id: ID of the stock to test
            timeframe_id: ID of the timeframe to test
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            hold_period_strategy: Strategy for hold period determination ('timeframe', 'formula')
            test_all: If True, tests all parameter combinations. If False, tests a subset.
            single_test: If True, tests only one combination for testing functionality.
            
        Returns:
            results: DataFrame containing test results
        """
        # Get thread-specific database connection for reading data
        conn = self.get_thread_db_connection()
        
        # Get stock data
        df = self.get_stock_data_thread_safe(conn, stock_id, timeframe_id, start_date, end_date)
        if df is None or len(df) < 100:  # Skip if not enough data
            print(f"Skipping stock_id={stock_id}, timeframe_id={timeframe_id}: Insufficient data")
            return None
        
        # Extract close prices for pattern mining
        close_prices = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Split data into train and test (80/20)
        train_size = int(len(close_prices) * 0.8)
        train_data = close_prices[:train_size]
        test_data = close_prices[train_size:]
        
        if len(train_data) < 100 or len(test_data) < 50:  # Skip if not enough data after split
            print(f"Skipping stock_id={stock_id}, timeframe_id={timeframe_id}: Insufficient data after split")
            return None
        
        # Determine parameter combinations to test
        if single_test:
            # Just test one combination for quick functionality check
            param_combinations = [(4, 24, 6)]  # Default settings
        elif test_all:
            # Test all combinations with appropriate hold periods
            param_combinations = []
            for n_pips in PARAM_RANGES['n_pips']:
                for lookback in PARAM_RANGES['lookback']:
                    if hold_period_strategy == "timeframe":
                        # Use timeframe-specific hold periods
                        for hold_period in self.get_hold_periods_for_timeframe(timeframe_id):
                            if lookback > hold_period and lookback > n_pips:  # Skip invalid combinations
                                param_combinations.append((n_pips, lookback, hold_period))
                    
                    elif hold_period_strategy == "formula":
                        # Use formula-based hold period
                        hold_period = self.get_formula_based_hold_period(lookback)
                        if lookback > hold_period and lookback > n_pips:  # Skip invalid combinations
                            param_combinations.append((n_pips, lookback, hold_period))
                    
                    else:  # fixed strategy - test all hold periods
                        for hold_period in PARAM_RANGES['hold_period']:
                            if lookback > hold_period and lookback > n_pips:  # Skip invalid combinations
                                param_combinations.append((n_pips, lookback, hold_period))
        else:
            # Test a subset of combinations to save time
            param_combinations = [
                (5, 24, 6),   # Default
                (3, 24, 6),   # Fewer pips
                (7, 24, 6),   # More pips
                (5, 12, 3),   # Shorter lookback
                (5, 36, 9),   # Longer lookback
                (5, 48, 12),  # Even longer lookback
                (3, 12, 3),   # Combo: short patterns
                (7, 48, 12)   # Combo: long patterns
            ]
        
        # Set up for parallel execution
        self.total_combinations = len(param_combinations)
        self.completed_combinations = 0
        
        # Results collection
        all_results = []
        
        # Define the worker function for each parameter combination
        def process_params(params):
            n_pips, lookback, hold_period = params
            thread_conn = self.get_thread_db_connection()
            
             # Get thread identification information
            thread_id = threading.get_ident()
            thread_name = threading.current_thread().name
            worker_id = thread_name.split('-')[-1] if '-' in thread_name else '?'
            
            
            try:
                print(f"Testing: n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}, "
                      f"distance_measure={DISTANCE_MEASURE}")
                print(f"[Worker {worker_id} | Thread {thread_id}] Testing: n_pips={n_pips}, "
                      f"lookback={lookback}, hold_period={hold_period}, "
                      f"distance_measure={DISTANCE_MEASURE}")
                
                
                # Register this configuration (thread-safe, stores in memory)
                config_id = self.register_experiment_config_thread_safe(
                    thread_conn, stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period_strategy
                )
                
                # Initialize and train pattern miner
                pip_miner = Pattern_Miner(
                    n_pips=n_pips,
                    lookback=lookback,
                    hold_period=hold_period,
                    returns_hold_period=hold_period  # Set equal for consistency
                )
                
                start_time = time.time()
                pip_miner.train(train_data)
                training_time = time.time() - start_time
                
                # Store patterns and clusters in memory
                cluster_metrics = self.store_cluster_metrics_thread_safe(
                    thread_conn, stock_id, timeframe_id, config_id, pip_miner
                )
                
                self.store_patterns_thread_safe(
                    thread_conn, stock_id, timeframe_id, config_id, pip_miner
                )
                
                if not cluster_metrics:
                    print(f"  No clusters generated for n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}")
                    return None
                
                # Calculate weighted score
                weighted_score = self.calculate_weighted_score(cluster_metrics)
                
                # Calculate cluster aggregate stats
                bullish_clusters = [m for m in cluster_metrics if m['label'] == 'Buy']
                bearish_clusters = [m for m in cluster_metrics if m['label'] == 'Sell']
                neutral_clusters = [m for m in cluster_metrics if m['label'] == 'Neutral']
                total_patterns = sum(m['pattern_count'] for m in cluster_metrics)
                avg_probability = np.mean([m['probability_score'] for m in cluster_metrics]) if cluster_metrics else 0
                avg_profit_factor = np.mean([m['profit_factor'] for m in cluster_metrics]) if cluster_metrics else 0
                avg_reward_risk = np.mean([m['reward_risk_ratio'] for m in cluster_metrics]) if cluster_metrics else 0
                
                # Return results for aggregation
                return {
                    'stock_id': stock_id,
                    'timeframe_id': timeframe_id,
                    'config_id': config_id,
                    'n_pips': n_pips,
                    'lookback': lookback,
                    'hold_period': hold_period,
                    'distance_measure': DISTANCE_MEASURE,
                    'num_clusters': len(cluster_metrics),
                    'num_patterns': total_patterns,
                    'bullish_clusters': len(bullish_clusters),
                    'bearish_clusters': len(bearish_clusters),
                    'neutral_clusters': len(neutral_clusters),
                    'training_time': training_time,
                    'avg_probability': avg_probability,
                    'avg_profit_factor': avg_profit_factor,
                    'avg_reward_risk': avg_reward_risk,
                    'weighted_score': weighted_score
                }
            
            except Exception as e:
                print(f"  Error testing parameters: {e}")
                import traceback
                traceback.print_exc()
                return None
            finally:
                # Update progress
                with self.results_lock:
                    self.completed_combinations += 1
                    completion = (self.completed_combinations / self.total_combinations) * 100
                    print(f"[Worker {worker_id}] Progress: {self.completed_combinations}/{self.total_combinations} "
                          f"({completion:.1f}%) combinations processed")
        
        # Execute parameter combinations in parallel
        print(f"Testing {len(param_combinations)} parameter combinations using {self.num_workers} worker threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all parameter combinations
            futures = [executor.submit(process_params, params) for params in param_combinations]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
        
        # Write all results to database in a single transaction
        self.write_all_results_to_database()
        
        if all_results:
            return pd.DataFrame(all_results)
        else:
            return None

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Threaded Gold Parameter Optimization Framework with GPU Acceleration')
    parser.add_argument('--stock', help='Stock ID or symbol to test (default: tests all)')
    parser.add_argument('--timeframe', help='Timeframe ID or name to test (default: tests all)')
    parser.add_argument('--start-date', help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data (format: YYYY-MM-DD)')
    parser.add_argument('--hold-strategy', choices=['timeframe', 'formula'], default='timeframe',
                       help='Hold period strategy (default: timeframe)')
    parser.add_argument('--test-all', action='store_true', help='Test all parameter combinations')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with default parameters')
    parser.add_argument('--compare', action='store_true', help='Compare hold period strategies')
    parser.add_argument('--threads', type=int, help='Number of worker threads to use (default: auto-detect based on CPU/GPU)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration even if available')
    
    args = parser.parse_args()
    
    try:
        # Create multi-threaded parameter tester
        tester = MultiThreadedParameterTester(num_workers=args.threads)
        
        # Print hardware information
        if tester.use_gpu and not args.no_gpu:
            print(f"GPU acceleration enabled: {physical_devices}")
            print(f"Using {tester.num_workers} worker threads with GPU acceleration")
        else:
            print(f"System has {multiprocessing.cpu_count()} CPU cores, using {tester.num_workers} worker threads (CPU only)")
        
        if args.compare:
            # Compare hold period strategies
            tester.compare_hold_period_strategies(args.stock)
        
        elif args.quick_test:
            # Run quick test with default parameters
            if not args.stock or not args.timeframe:
                print("Error: --stock and --timeframe are required for --quick-test")
                return 1
            
            tester.run_quick_test(args.stock, args.timeframe, args.start_date, args.end_date)
        
        else:
            # Run full tests
            if args.stock and args.timeframe:
                # Test specific stock and timeframe
                stock_id, stock_symbol = tester.get_stock_by_symbol_or_id(args.stock)
                timeframe_id, minutes, timeframe_name = tester.get_timeframe_by_name_or_id(args.timeframe)
                
                results = tester.run_parameter_test(
                    stock_id, timeframe_id, args.start_date, args.end_date,
                    args.hold_strategy, args.test_all
                )
                
                if results is not None and not results.empty:
                    tester.plot_results(results, stock_symbol, timeframe_name)
                    tester.generate_report(results, stock_id, stock_symbol, 
                                          timeframe_id, timeframe_name)
            
            elif args.stock:
                # Test all timeframes for a specific stock
                tester.run_all_tests(
                    args.stock, args.test_all, args.hold_strategy,
                    args.start_date, args.end_date
                )
            
            else:
                # Test all stocks and timeframes
                tester.run_all_tests(
                    None, args.test_all, args.hold_strategy,
                    args.start_date, args.end_date
                )
        
    except Exception as e:
        print(f"Error during parameter testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up resources
        if 'tester' in locals():
            tester.close()
    
    return 0

if __name__ == "__main__":
    # Initialize tester with GPU support
    tester = MultiThreadedParameterTester(db_path="./Data//Storage/data.db")
    
    try:
        # Check for GPU availability
        if tester.use_gpu:
            print("Using GPU acceleration for parameter testing")
        else:
            print("Using CPU only for parameter testing")
        
        # Optimize database before starting
        tester._optimize_database_for_concurrency()
        
        # Run parameter tests with GPU acceleration
        tester.run_parameter_test(stock_id=1, timeframe_id=5, start_date="2024-01-01", end_date="2025-01-01", test_all=True)
        

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print("Database lock error encountered, attempting to force close connections...")
            tester.force_close_db_connections()
            print("Please restart the process")
        else:
            print(f"SQLite error: {e}")
    finally:
        # Close the tester
        tester.close()
        print("Parameter testing completed successfully")