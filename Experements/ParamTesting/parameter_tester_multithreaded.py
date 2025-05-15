#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Parameter Optimization Framework with Multi-Threading Support

This script systematically tests different combinations of parameters
for pattern mining to determine the optimal settings for Gold across
different timeframes. It implements multi-threading to utilize all CPU cores.

The process:
1. Connects to the normalized database
2. For each timeframe, runs pattern mini                    cursor.execute(
                        INSERT INTO patterns 
                        (pattern_id,stock_id, timeframe_id, config_id, cluster_id, price_points_json,
                            volume, outcome, max_gain, max_drawdown)
                        VALUES (?, ?,?, ?, ?, ?, ?, ?, ?, ?),
                        (pattern_id, stock_id, timeframe_id, config_id, cluster_id, pattern_json,
                        volume_data, outcome, max_gain, max_drawdown)
                    )different parameter combinations in parallel
3. Evaluates cluster performance metrics
4. Stores results in the database for comparison
5. Generates performance reports and visualizations
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

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent # Navigate up to project root
sys.path.append(str(project_root))

# Import custom modules
from Pattern.pip_pattern_miner import Pattern_Miner
from Data.Database.db import Database
from Experements.ParamTesting.parameter_tester import ParameterTester

# Parameter ranges to test
PARAM_RANGES = {
    'n_pips': [3, 4, 5, 6, 7, 8],
    'lookback': [12, 24, 36, 48],
    'hold_period': [3, 6, 12, 24],
}

# Fixed parameter for test
DISTANCE_MEASURE = 2  # Perpendicular distance selected as standard


class MultiThreadedParameterTester(ParameterTester):
    """Class for testing different parameter combinations for pattern mining with multi-threading support."""
    def __init__(self, db_path=None, num_workers=None):
        """Initialize the parameter tester with multi-threading support.
        
        Args:
            db_path: Path to the SQLite database. If None, uses the default connection.
            num_workers: Number of worker threads to use. If None, uses CPU count - 1.
        """
        # Initialize the parent class
        super().__init__(db_path)
        
        # Store the db_path for thread-specific connections
        self.db_path = self.db.db_name
        
        # Thread-local storage for database connections
        self.thread_local = threading.local()
        
        # Database lock for synchronizing access
        self.db_lock = threading.RLock()
        
        # Set up multi-threading
        self.num_workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        print(f"Initialized with {self.num_workers} worker threads")
        
        # Progress tracking
        self.total_combinations = 0
        self.completed_combinations = 0
    
    def get_thread_db_connection(self):
        """Get a thread-specific database connection."""
        if not hasattr(self.thread_local, "connection"):
            self.thread_local.connection = sqlite3.connect(self.db_path)
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
        """Thread-safe version of register_experiment_config using a dedicated connection.
        
        Args:
            conn: Thread-specific database connection
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            n_pips: Number of PIPs
            lookback: Lookback window size
            hold_period: Hold period
            strategy_type: Type of hold period strategy ('timeframe', 'formula', None)
            
        Returns:
            config_id: ID of the registered configuration
        """
        description = (f"Pattern mining with {n_pips} pips, {lookback} bar lookback, "
                      f"{hold_period} bar hold, perpendicular distance")
        
        name = f"Config_P{n_pips}_L{lookback}_H{hold_period}_D{DISTANCE_MEASURE}"
        if strategy_type:
            name = f"{name}_{strategy_type}"
        
        # Check if this config already exists
        cursor = conn.cursor()
        result = cursor.execute(
            """SELECT config_id FROM experiment_configs 
               WHERE stock_id = ? AND timeframe_id = ? AND n_pips = ? AND lookback = ? AND 
               hold_period = ? AND distance_measure = ?""",
            (stock_id, timeframe_id, n_pips, lookback, hold_period, DISTANCE_MEASURE)
        ).fetchone()
        
        if result:
            return result[0]
        
        # Insert new config
        cursor.execute(
            """INSERT INTO experiment_configs 
               (name, stock_id, timeframe_id, n_pips, lookback, hold_period, returns_hold_period, distance_measure, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period, DISTANCE_MEASURE, description)
        )
        conn.commit()
        
        return cursor.lastrowid
    
    def store_cluster_metrics_thread_safe(self, conn, stock_id, timeframe_id, config_id, pip_miner):
        """Thread-safe version of store_cluster_metrics using a dedicated connection.
        
        Args:
            conn: Thread-specific database connection
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration
            pip_miner: Trained Pattern_Miner instance
            
        Returns:
            list: Metrics for each cluster
        """
        # Get stock symbol and timeframe name for descriptions
        cursor = conn.cursor()
         # Get stock symbol and timeframe name for descriptions
        stock_symbol = cursor.execute(
            "SELECT symbol FROM stocks WHERE stock_id = ?", (stock_id,)
        ).fetchone()[0]
        
        timeframe_name = cursor.execute(
            "SELECT name FROM timeframes WHERE timeframe_id = ?", (timeframe_id,)
        ).fetchone()[0]
        
        # Collect cluster metrics
        cluster_metrics = []
        for i, cluster_center in enumerate(pip_miner._cluster_centers):
            # Get patterns in this cluster
            pattern_indices = pip_miner._pip_clusters[i]
            pattern_count = len(pattern_indices)
            
            if pattern_count < 4:
                continue
            
            # Calculate metrics
            outcomes = pip_miner._returns_fixed_hold[pattern_indices]
           
            avg_outcome =pip_miner._cluster_returns[i]
            
            
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
            elif  label == 'Sell':
                max_gain = pip_miner._cluster_mae[i]
                max_drawdown =pip_miner._cluster_mfe[i]            
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
               probability_score_dir =0.5
            
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
            
            # Store cluster in database
            price_points_json = json.dumps(cluster_center)
            
            avg_volume = None
            
            # Check if cluster exists
            cluster_id_result = cursor.execute(
                """SELECT cluster_id FROM clusters 
                   WHERE stock_id = ? AND timeframe_id = ? AND config_id = ? AND 
                   avg_price_points_json = ?""",
                (stock_id, timeframe_id, config_id, price_points_json)
            ).fetchone()
            
            if cluster_id_result:
                cluster_id = cluster_id_result[0]                # Update existing cluster
                cursor.execute(
                    """UPDATE clusters SET 
                       outcome = ?, label = ?, probability_score_stat = ?, probability_score_dir = ?, pattern_count = ?,
                       max_gain = ?, max_drawdown = ?, reward_risk_ratio = ?, profit_factor = ?, description = ?
                       WHERE cluster_id = ?""",
                    (avg_outcome, label, probability_score_stat, probability_score_dir, pattern_count,
                     max_gain, max_drawdown, reward_risk_ratio, profit_factor, cluster_description,
                     cluster_id)
                )
            else:                    # Insert new cluster
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO clusters 
                       (cluster_id,stock_id, timeframe_id, config_id, avg_price_points_json, avg_volume, 
                        outcome, label, probability_score_dir, probability_score_stat, pattern_count,
                        max_gain, max_drawdown, reward_risk_ratio, profit_factor, description,
                        created_at)
                       VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (i,stock_id, timeframe_id, config_id, price_points_json, avg_volume,
                     avg_outcome, label, probability_score_dir, probability_score_stat, pattern_count,
                     max_gain, max_drawdown, reward_risk_ratio, profit_factor, cluster_description,
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
                cluster_id = cursor.lastrowid
              # Store metrics for return
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
        
        conn.commit()
        return cluster_metrics
    
    def store_patterns_thread_safe(self, conn, stock_id, timeframe_id, config_id, pip_miner):
        """Thread-safe version of store_patterns using a dedicated connection.
        
        Args:
            conn: Thread-specific database connection
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration
            pip_miner: Trained Pattern_Miner instance
        """
        cursor = conn.cursor()
         # loop through the clusters
        for j, cluster in enumerate(pip_miner._pip_clusters):
           
            # loop through the patterns in the cluster
            for i, pattern_id in enumerate(cluster):
                pattern = pip_miner._unique_pip_patterns[pattern_id]
                
                # Store pattern
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
                
               
                  # Check if pattern exists
                pattern_exists = cursor.execute(
                    """SELECT pattern_id FROM patterns 
                    WHERE stock_id = ? AND timeframe_id = ? AND config_id = ? AND 
                    price_points_json = ?""",
                    (stock_id, timeframe_id, config_id, pattern_json)
                ).fetchone()
                
                if not pattern_exists:
                    # Store volume data if available
                    volume_data = None
                    # if hasattr(pip_miner, '_volume_data') and pip_miner._volume_data is not None:
                    #     volume_data = json.dumps(pip_miner._volume_data[i].tolist())
                      # Get the cluster ID from the database by matching with the cluster center
                    cluster_center = json.dumps(pip_miner._cluster_centers[j])
                    cluster_id_result = cursor.execute(
                        """SELECT cluster_id FROM clusters 
                        WHERE stock_id = ? AND timeframe_id = ? AND config_id = ? AND 
                        avg_price_points_json = ?""",
                        (stock_id, timeframe_id, config_id, cluster_center)
                    ).fetchone()
                    
                   
                
                try:
                    # Insert new pattern
                    cursor.execute(
                        """INSERT INTO patterns 
                        (pattern_id, stock_id, timeframe_id, config_id, cluster_id, price_points_json,
                            volume, outcome, max_gain, max_drawdown)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (pattern_id, stock_id, timeframe_id, config_id, j, pattern_json,
                            volume_data, outcome, max_gain, max_drawdown)
                    )
                except sqlite3.IntegrityError:
                    # Handle duplicate pattern insertion
                    print(f"Pattern {pattern_id} already exists in the database.")
                    continue
           
        conn.commit()
    
    def store_performance_metrics_thread_safe(self, conn, stock_id, timeframe_id, config_id, metrics, start_date=None, end_date=None):
       
       
        # Store in database
        cursor = conn.cursor()
        """
        Store aggregate performance metrics for a parameter combination.
        
        Args:
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration
            results: Dict containing performance results
            start_date: Start date of test period
            end_date: End date of test period
        """
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
        
        # Check if metrics exist
        metrics_exist = conn.execute(
            """SELECT metric_id FROM performance_metrics 
               WHERE stock_id = ? AND timeframe_id = ? AND config_id = ? AND
               start_date = ? AND end_date = ?""",
            (stock_id, timeframe_id, config_id, start_date_str, end_date_str)
        ).fetchone()
        
        if metrics_exist:
            # Update existing metrics
            conn.execute(
                """UPDATE performance_metrics SET
                   total_trades = ?, win_count = ?, loss_count = ?, win_rate = ?,
                   avg_win = ?, avg_loss = ?, profit_factor = ?, max_drawdown = ?
                   WHERE stock_id = ? AND timeframe_id = ? AND config_id = ? AND
                   start_date = ? AND end_date = ?""",
                (metrics.get('total_trades', 0), metrics.get('win_count', 0), 
                 metrics.get('loss_count', 0), metrics.get('win_rate', 0),
                 metrics.get('avg_win', 0), metrics.get('avg_loss', 0),
                 metrics.get('profit_factor', 0), metrics.get('max_drawdown', 0),
                 stock_id, timeframe_id, config_id, start_date_str, end_date_str)
            )
        else:
            # Insert new metrics
            conn.execute(
                """INSERT INTO performance_metrics
                   (stock_id, timeframe_id, config_id, start_date, end_date,
                    total_trades, win_count, loss_count, win_rate,
                    avg_win, avg_loss, profit_factor, max_drawdown)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (stock_id, timeframe_id, config_id, start_date_str, end_date_str,
                 metrics.get('total_trades', 0), metrics.get('win_count', 0), 
                 metrics.get('loss_count', 0), metrics.get('win_rate', 0),
                 metrics.get('avg_win', 0), metrics.get('avg_loss', 0),
                 metrics.get('profit_factor', 0), metrics.get('max_drawdown', 0))
            )
        
        conn.commit()
    
    def close(self):
        """Close all database connections."""
        # Close parent connection
        if hasattr(self, 'db') and self.db.connection:
            self.db.connection.close()
        
        # Close thread-specific connections if they exist
        if hasattr(self, 'thread_local') and hasattr(self.thread_local, 'connection'):
            self.thread_local.connection.close()
    
    def _test_parameter_combination(self, args):
        """Test a single parameter combination for a given stock and timeframe.
        
        Args:
            args: Tuple containing (stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period_strategy)
            
        Returns:
            Dictionary containing test results or None if error
        """
        stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period_strategy, start_date, end_date = args
        
        # Get thread-specific database connection
        conn = self.get_thread_db_connection()
        
        try:
            # Get stock data using thread-specific connection
            with self.db_lock:
                df = self.get_stock_data_thread_safe(conn, stock_id, timeframe_id, start_date, end_date)
                
            if df is None or len(df) < 100:  # Skip if not enough data
                return None
            
            # Extract close prices for pattern mining
            close_prices = df['close'].values
            volumes = df['volume'].values if 'volume' in df.columns else None
            
            # Split data into train and test (80/20)
            train_size = int(len(close_prices) * 0.8)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            
            if len(train_data) < 100 or len(test_data) < 50:  # Skip if not enough data after split
                return None
            
            # Register this configuration
            with self.db_lock:
                config_id = self.register_experiment_config_thread_safe(
                    conn, stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period_strategy
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
            
            # Store patterns and clusters
            with self.db_lock:
                cluster_metrics = self.store_cluster_metrics_thread_safe(conn, stock_id, timeframe_id, config_id, pip_miner)
                self.store_patterns_thread_safe(conn, stock_id, timeframe_id, config_id, pip_miner)
                
            
            if not cluster_metrics:
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
            
            # Store performance metrics
            perf_metrics = {
                'total_trades': total_patterns,
                'win_count': sum(m['pattern_count'] for m in bullish_clusters),
                'loss_count': sum(m['pattern_count'] for m in bearish_clusters),
                'win_rate': len(bullish_clusters) / len(cluster_metrics) * 100 if cluster_metrics else 0,
                'avg_win': np.mean([m['outcome'] for m in bullish_clusters]) if bullish_clusters else 0,
                'avg_loss': np.mean([m['outcome'] for m in bearish_clusters]) if bearish_clusters else 0,
                'profit_factor': avg_profit_factor,
                'max_drawdown': min([m['max_drawdown'] for m in cluster_metrics]) if cluster_metrics else 0
            }
            
            # Store in database
            with self.db_lock:
                self.store_performance_metrics_thread_safe(
                    conn, stock_id, timeframe_id, config_id, perf_metrics,
                    df.index[0], df.index[-1]
                )
            
            # Return results
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
            print(f"Error testing parameters (n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}): {e}")
            return None
    
    def run_parameter_test(self, stock_id, timeframe_id, start_date=None, end_date=None, 
                          hold_period_strategy="timeframe", test_all=False, single_test=False):
        """Run parameter testing for a specific stock and timeframe with multi-threading.
        
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
        
        # Set up progress tracking
        self.total_combinations = len(param_combinations)
        self.completed_combinations = 0
        
        # Prepare arguments for the thread pool
        thread_args = [
            (stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period_strategy, start_date, end_date)
            for n_pips, lookback, hold_period in param_combinations
        ]
        
        # Set up thread pool and run tests in parallel
        results = []
        
        # Get stock and timeframe info for display
        stock_symbol = self.db.connection.execute(
            "SELECT symbol FROM stocks WHERE stock_id = ?", (stock_id,)
        ).fetchone()[0]
        
        timeframe_name = self.db.connection.execute(
            "SELECT name FROM timeframes WHERE timeframe_id = ?", (timeframe_id,)
        ).fetchone()[0]
        
        print(f"Testing {self.total_combinations} parameter combinations for {stock_symbol} on {timeframe_name} timeframe")
        print(f"Using {self.num_workers} worker threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(self._test_parameter_combination, args): args for args in thread_args}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_args), total=len(future_to_args), desc="Testing Parameters"):
                args = future_to_args[future]
                n_pips, lookback, hold_period = args[2:5]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"  Completed: n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}, "
                              f"Clusters={result['num_clusters']}, Patterns={result['num_patterns']}, "
                              f"Score={result['weighted_score']:.2f}, PF={result['avg_profit_factor']:.2f}")
                    else:
                        print(f"  Skipped: n_pips={n_pips}, lookback={lookback}, hold_period={hold_period} "
                              f"(no valid clusters generated)")
                        
                except Exception as e:
                    print(f"  Failed: n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}, Error: {e}")
                
                self.completed_combinations += 1
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return None
    
    def run_all_tests(self, stock_identifier=None, test_all_params=False, hold_period_strategy="timeframe",
                     start_date=None, end_date=None):
        """Run parameter tests for all stocks and timeframes or a specific stock with multi-threading.
        
        Args:
            stock_identifier: Stock ID or symbol (optional, if None runs for all stocks)
            test_all_params: If True, tests all parameter combinations
            hold_period_strategy: Strategy for hold period determination ('timeframe', 'formula')
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
        """
        if stock_identifier:
            # Get stock info
            stock_id, stock_symbol = self.get_stock_by_symbol_or_id(stock_identifier)
            stocks = [(stock_id, stock_symbol)]
        else:
            # Get all stocks
            stocks, _ = self.get_stocks_and_timeframes()
        
        # Get all timeframes
        _, timeframes = self.get_stocks_and_timeframes()
        
        # Track overall progress
        total_tests = len(stocks) * len(timeframes)
        completed_tests = 0
        
        print(f"Running tests for {len(stocks)} stocks across {len(timeframes)} timeframes")
        print(f"Using {self.num_workers} worker threads for each stock/timeframe combination")
        
        for stock_id, stock_symbol in stocks:
            for timeframe_id, minutes, timeframe_name in timeframes:
                print(f"\nTesting {stock_symbol} on {timeframe_name} timeframe ({completed_tests+1}/{total_tests})")
                
                try:
                    results = self.run_parameter_test(
                        stock_id, timeframe_id, start_date, end_date,
                        hold_period_strategy, test_all=test_all_params
                    )
                    
                    if results is not None and not results.empty:
                        # Plot results
                        try:
                            self.plot_results(results, stock_symbol, timeframe_name)
                        except Exception as e:
                            print(f"Error plotting results: {e}")
                        
                        # Generate report
                        self.generate_report(results, stock_id, stock_symbol, timeframe_id, timeframe_name)
                    
                except Exception as e:
                    print(f"Error testing {stock_symbol} on {timeframe_name}: {e}")
                    
                completed_tests += 1
        
        print("\nAll parameter tests completed")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Threaded Gold Parameter Optimization Framework')
    parser.add_argument('--stock', help='Stock ID or symbol to test (default: tests all)')
    parser.add_argument('--timeframe', help='Timeframe ID or name to test (default: tests all)')
    parser.add_argument('--start-date', help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data (format: YYYY-MM-DD)')
    parser.add_argument('--hold-strategy', choices=['timeframe', 'formula'], default='timeframe',
                       help='Hold period strategy (default: timeframe)')
    parser.add_argument('--test-all', action='store_true', help='Test all parameter combinations')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with default parameters')
    parser.add_argument('--compare', action='store_true', help='Compare hold period strategies')
    parser.add_argument('--threads', type=int, help='Number of worker threads to use (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    try:
        # Create multi-threaded parameter tester
        tester = MultiThreadedParameterTester(num_workers=args.threads)
        
        print(f"System has {multiprocessing.cpu_count()} CPU cores, using {tester.num_workers} worker threads")
        
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
    test =  MultiThreadedParameterTester()
    test.run_parameter_test(stock_id=1, timeframe_id=7, start_date="2024-01-01", end_date="2025-01-01",test_all=True)