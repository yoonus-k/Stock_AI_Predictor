#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Parameter Optimization Framework

This script systematically tests different combinations of parameters
for pattern mining to determine the optimal settings for Gold across
different timeframes.

The process:
1. Connects to the normalized database
2. For each timeframe, runs pattern mining with different parameter combinations
3. Evaluates cluster performance metrics
4. Stores results in the database for comparison
5. Generates performance reports and visualizations
"""

import os
import sys
import json
import numpy as np
import warnings

# Add a warnings attribute to numpy if it doesn't exist
if not hasattr(np, 'warnings'):
    np.warnings = warnings

print("NumPy warnings patch applied successfully")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3 as db
from itertools import product
from datetime import datetime, timedelta
import time

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent # Navigate up to project root
sys.path.append(str(project_root))

# Import custom modules
from pip_pattern_miner import Pattern_Miner


# Parameter ranges to test
PARAM_RANGES = {
    'n_pips': [3, 4, 5, 6, 7, 8],
    'lookback': [12, 24, 36, 48],
    'hold_period': [3, 6, 12, 24],
}

# Fixed parameter for test
DISTANCE_MEASURE = 2  # Perpendicular distance selected as standard

# Output directory for reports
OUTPUT_DIR = project_root / "Data" / "Utils" / "ParamTesting" / "Results"

class ParameterTester:
    """Class for testing different parameter combinations for pattern mining."""
    def __init__(self, db_path=None):
        """Initialize the parameter tester.
        
        Args:
            db_path: Path to the SQLite database. If None, uses the default connection.
        """
        self.connection = db.connect(db_path, check_same_thread=False)
        self.create_directory_if_not_exists(OUTPUT_DIR)
        
        # Enhanced in-memory storage for batch database writes
        self.cluster_results = {}  # {(stock_id, timeframe_id, config_id): [cluster_metrics]}
        self.pattern_results = {}  # {(stock_id, timeframe_id, config_id): [(cluster_id, pattern_data)]}
        self.config_results = {}   # {(stock_id, timeframe_id, n_pips, lookback, hold_period): config_id}
        self.performance_results = {}  # {(stock_id, timeframe_id, config_id): metrics}
        
        # Configuration tracking
        self.next_config_id = 1
        self.config_id_map = {}
        
    def create_directory_if_not_exists(self, directory_path):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
    
    def get_stock_by_symbol_or_id(self, stock_identifier):
        """Get stock ID from symbol or ID.
        
        Args:
            stock_identifier: Stock symbol or ID
            
        Returns:
            tuple: (stock_id, symbol)
        """
        if isinstance(stock_identifier, int) or stock_identifier.isdigit():
            # It's an ID
            stock_id = int(stock_identifier)
            result =self.connection.execute(
                "SELECT stock_id, symbol FROM stocks WHERE stock_id = ?", 
                (stock_id,)
            ).fetchone()
        else:
            # It's a symbol
            result =self.connection.execute(
                "SELECT stock_id, symbol FROM stocks WHERE symbol LIKE ?", 
                (f"%{stock_identifier}%",)
            ).fetchone()
        
        if not result:
            raise ValueError(f"Stock with identifier '{stock_identifier}' not found")
        
        return result
    
    def get_timeframe_by_name_or_id(self, timeframe_identifier):
        """Get timeframe ID from name or ID.
        
        Args:
            timeframe_identifier: Timeframe name or ID
            
        Returns:
            tuple: (timeframe_id, minutes, name)
        """
        if isinstance(timeframe_identifier, int) or timeframe_identifier.isdigit():
            # It's an ID
            timeframe_id = int(timeframe_identifier)
            result =self.connection.execute(
                "SELECT timeframe_id, minutes, name FROM timeframes WHERE timeframe_id = ?", 
                (timeframe_id,)
            ).fetchone()
        else:
            # It's a name
            result =self.connection.execute(
                "SELECT timeframe_id, minutes, name FROM timeframes WHERE name LIKE ?", 
                (f"%{timeframe_identifier}%",)
            ).fetchone()
        
        if not result:
            raise ValueError(f"Timeframe with identifier '{timeframe_identifier}' not found")
        
        return result
    
    def get_stocks_and_timeframes(self):
        """Get all stock IDs and timeframe IDs from the database."""
        stocks =self.connection.execute("SELECT stock_id, symbol FROM stocks").fetchall()
        timeframes =self.connection.execute("SELECT timeframe_id, minutes, name FROM timeframes").fetchall()
        
        return stocks, timeframes
    
    def get_timeframe_category(self, timeframe_id):
        """
        Determine the category of a timeframe (lower, medium, higher).
        
        Args:
            timeframe_id: ID of the timeframe
            
        Returns:
            str: Category of the timeframe ('lower', 'medium', 'higher')
        """
        timeframe_info =self.connection.execute(
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
    def register_experiment_config(self, stock_id, timeframe_id, n_pips, lookback, hold_period, strategy_type=None):
        """Register an experiment configuration in memory.
        
        Args:
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
        
        # Check if this config already exists in memory
        config_key = (stock_id, timeframe_id, n_pips, lookback, hold_period)
        if config_key in self.config_results:
            return self.config_results[config_key]
        
        # Create new config in memory
        config_id = self.next_config_id
        self.next_config_id += 1
        
        self.config_results[config_key] = config_id
        self.config_id_map[config_id] = {
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
        
        return config_id
    
    def get_stock_data(self, stock_id, timeframe_id, start_date=None, end_date=None):
        """Get stock data for a specific stock and timeframe.
        
        Args:
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
        cursor =self.connection.cursor()
        cursor.execute(query, params)
        data = cursor.fetchall()
        
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    def store_cluster_metrics(self, stock_id, timeframe_id, config_id, pip_miner):
        """
        Store cluster metrics for evaluation in memory.
        
        Args:
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration
            pip_miner: Trained Pattern_Miner instance
            
        Returns:
            list: Metrics for each cluster
        """
        # Get stock symbol and timeframe name for descriptions
        stock_symbol = self.connection.execute(
            "SELECT symbol FROM stocks WHERE stock_id = ?", (stock_id,)
        ).fetchone()[0]
        
        timeframe_name = self.connection.execute(
            "SELECT name FROM timeframes WHERE timeframe_id = ?", (timeframe_id,)
        ).fetchone()[0]
        
        # Create key for in-memory storage
        key = (stock_id, timeframe_id, config_id)
        
        # Initialize cluster list if it doesn't exist
        if key not in self.cluster_results:
            self.cluster_results[key] = []
        
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
            
            # Store cluster data in memory
            price_points_json = json.dumps(cluster_center.tolist() if isinstance(cluster_center, np.ndarray) else cluster_center)
            
            cluster_data = {
                'cluster_id': i,
                'stock_id': stock_id,
                'timeframe_id': timeframe_id,
                'config_id': config_id,
                'avg_price_points_json': price_points_json,
                'avg_volume': None,
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
                'created_at': datetime.now()
            }
            
            # Add to in-memory storage
            self.cluster_results[key].append(cluster_data)
            
            # Store metrics for return
            cluster_metrics.append({
                'cluster_id': i,
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
        
        return cluster_metrics
    def store_patterns(self, stock_id, timeframe_id, config_id, pip_miner):
        """
        Store patterns identified by the pip_miner in memory.
        
        Args:
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            config_id: ID of the configuration
            pip_miner: Trained Pattern_Miner instance
        """
        # Create key for in-memory storage
        key = (stock_id, timeframe_id, config_id)
        
        # Initialize pattern list if it doesn't exist
        if key not in self.pattern_results:
            self.pattern_results[key] = []
        
        # loop through the clusters
        for j, cluster in enumerate(pip_miner._pip_clusters):
            # loop through the patterns in the cluster
            for i, pattern_id in enumerate(cluster):
                pattern = pip_miner._unique_pip_patterns[pattern_id]
                
                # Prepare pattern data
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
                    max_gain = pip_miner._returns_mfe[i] if i < len(pip_miner._returns_mfe) else 0
                    max_drawdown = pip_miner._returns_mae[i] if i < len(pip_miner._returns_mae) else 0
                elif pattern_label == 'Sell':
                    max_gain = pip_miner._returns_mae[i] if i < len(pip_miner._returns_mae) else 0
                    max_drawdown = pip_miner._returns_mfe[i] if i < len(pip_miner._returns_mfe) else 0
                else:
                    max_gain = 0
                    max_drawdown = 0
                
                # Store volume data if available
                volume_data = None
                # if hasattr(pip_miner, '_volume_data') and pip_miner._volume_data is not None:
                #     volume_data = json.dumps(pip_miner._volume_data[i].tolist())
                
                # Store pattern in memory
                pattern_data = {
                    'pattern_id': pattern_id,
                    'stock_id': stock_id,
                    'timeframe_id': timeframe_id,
                    'config_id': config_id,
                    'cluster_id': j,
                    'price_points_json': pattern_json,
                    'volume': volume_data,
                    'outcome': outcome,
                    'max_gain': max_gain,
                    'max_drawdown': max_drawdown
                }
                
                # Add to in-memory storage
                self.pattern_results[key].append(pattern_data)
    def store_performance_metrics(self, stock_id, timeframe_id, config_id, results, start_date, end_date):
        """
        Store aggregate performance metrics for a parameter combination in memory.
        
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
        
        # Create key for in-memory storage
        key = (stock_id, timeframe_id, config_id)
        
        # Store performance data in memory
        performance_data = {
            'stock_id': stock_id,
            'timeframe_id': timeframe_id,
            'config_id': config_id,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'total_trades': results.get('total_trades', 0),
            'win_count': results.get('win_count', 0),
            'loss_count': results.get('loss_count', 0),
            'win_rate': results.get('win_rate', 0),
            'avg_win': results.get('avg_win', 0),
            'avg_loss': results.get('avg_loss', 0),
            'profit_factor': results.get('profit_factor', 0),
            'max_drawdown': results.get('max_drawdown', 0)
        }
        
        # Update or add performance metrics
        self.performance_results[key] = performance_data
    
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
    def run_parameter_test(self, stock_id, timeframe_id, start_date=None, end_date=None, 
                          hold_period_strategy="timeframe", test_all=False, single_test=False,
                          save_to_db=True):
        """Run parameter testing for a specific stock and timeframe.
        
        Args:
            stock_id: ID of the stock to test
            timeframe_id: ID of the timeframe to test
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            hold_period_strategy: Strategy for hold period determination ('timeframe', 'formula')
            test_all: If True, tests all parameter combinations. If False, tests a subset.
            single_test: If True, tests only one combination for testing functionality.
            save_to_db: If True, saves results to database after completion.
            
        Returns:
            results: DataFrame containing test results
        """
        # Get stock data
        df = self.get_stock_data(stock_id, timeframe_id, start_date, end_date)
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
        
        results = []
        
        for n_pips, lookback, hold_period in param_combinations:
            print(f"Testing: n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}, "
                 f"distance_measure={DISTANCE_MEASURE}")
            
            try:
                # Register this configuration in memory
                config_id = self.register_experiment_config(
                    stock_id, timeframe_id, n_pips, lookback, hold_period, hold_period_strategy
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
                cluster_metrics = self.store_cluster_metrics(stock_id, timeframe_id, config_id, pip_miner)
                self.store_patterns(stock_id, timeframe_id, config_id, pip_miner)
                
                if not cluster_metrics:
                    print("  No clusters generated for this configuration")
                    continue
                
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
                
                # Store performance metrics in memory
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
                
                # Store performance metrics in memory
                self.store_performance_metrics(
                    stock_id, timeframe_id, config_id, perf_metrics,
                    df.index[0], df.index[-1]
                )
                
                # Store results for comparison
                results.append({
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
                })
                
                print(f"  Results: Clusters={len(cluster_metrics)}, Patterns={total_patterns}, "
                     f"Score={weighted_score:.2f}, PF={avg_profit_factor:.2f}")
            
            except Exception as e:
                print(f"  Error testing parameters: {e}")
        
        # Save all data to database if requested
        if save_to_db:
            self.save_all_to_database()
            self.clear_memory_storage()  # Free up memory after saving
        
        if results:
            return pd.DataFrame(results)
        else:
            return None
    def save_all_to_database(self):
        """Save all in-memory data to the database in batch operations."""
        print("Saving all data to database...")
        
        try:
            # Begin transaction
            self.connection.execute("BEGIN TRANSACTION")
            
            # Insert experiment configs in batch
            print(f"Saving {len(self.config_id_map)} configurations to database...")
            config_params = [
                (config_id, data['name'], data['stock_id'], data['timeframe_id'], 
                 data['n_pips'], data['lookback'], data['hold_period'],
                 data['returns_hold_period'], data['distance_measure'], data['description'])
                for config_id, data in self.config_id_map.items()
            ]
            
            self.connection.executemany(
                """INSERT OR IGNORE INTO experiment_configs 
                   (config_id, name, stock_id, timeframe_id, n_pips, lookback, hold_period, 
                   returns_hold_period, distance_measure, description)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                config_params
            )
            
            # Insert clusters in batch
            print(f"Saving clusters from {len(self.cluster_results)} configurations to database...")
            all_clusters = []
            for key, clusters in self.cluster_results.items():
                for cluster in clusters:
                    all_clusters.append((
                        cluster['cluster_id'], cluster['stock_id'], cluster['timeframe_id'],
                        cluster['config_id'], cluster['avg_price_points_json'], cluster['avg_volume'],
                        cluster['outcome'], cluster['label'], cluster['probability_score_dir'],
                        cluster['probability_score_stat'], cluster['pattern_count'],
                        cluster['max_gain'], cluster['max_drawdown'], cluster['reward_risk_ratio'],
                        cluster['profit_factor'], cluster['description'], cluster['created_at']
                    ))
            
            # Process clusters in batches to avoid SQLite limitations
            batch_size = 500
            total_clusters = len(all_clusters)
            for i in range(0, total_clusters, batch_size):
                batch = all_clusters[i:i+batch_size]
                self.connection.executemany(
                    """INSERT OR IGNORE INTO clusters 
                       (cluster_id, stock_id, timeframe_id, config_id, avg_price_points_json, avg_volume, 
                        outcome, label, probability_score_dir, probability_score_stat, pattern_count,
                        max_gain, max_drawdown, reward_risk_ratio, profit_factor, description,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch
                )
                print(f"  Saved {min(i+batch_size, total_clusters)}/{total_clusters} clusters")
            
            # Insert patterns in batch - largest data volume
            print(f"Saving patterns from {len(self.pattern_results)} configurations to database...")
            pattern_count = 0
            for key, patterns in self.pattern_results.items():
                pattern_count += len(patterns)
                # Process patterns in batches to avoid SQLite limitations
                for i in range(0, len(patterns), batch_size):
                    batch = [
                        (p['pattern_id'], p['stock_id'], p['timeframe_id'], p['config_id'], 
                         p['cluster_id'], p['price_points_json'], p['volume'], 
                         p['outcome'], p['max_gain'], p['max_drawdown'])
                        for p in patterns[i:i+batch_size]
                    ]
                    self.connection.executemany(
                        """INSERT OR IGNORE INTO patterns 
                           (pattern_id, stock_id, timeframe_id, config_id, cluster_id, price_points_json,
                            volume, outcome, max_gain, max_drawdown)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        batch
                    )
                print(f"  Saved {len(patterns)} patterns for configuration {key}")
            print(f"Total patterns saved: {pattern_count}")
            
            # Insert performance metrics
            print(f"Saving {len(self.performance_results)} performance metrics to database...")
            perf_params = [
                (m['stock_id'], m['timeframe_id'], m['config_id'], m['start_date'], m['end_date'], 
                 m['total_trades'], m['win_count'], m['loss_count'], m['win_rate'],
                 m['avg_win'], m['avg_loss'], m['profit_factor'], m['max_drawdown'])
                for m in self.performance_results.values()
            ]
            
            self.connection.executemany(
                """INSERT OR IGNORE INTO performance_metrics
                   (stock_id, timeframe_id, config_id, start_date, end_date,
                    total_trades, win_count, loss_count, win_rate,
                    avg_win, avg_loss, profit_factor, max_drawdown)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                perf_params
            )
            
            # Commit all changes
            self.connection.commit()
            print("All data saved to database successfully!")
            
        except Exception as e:
            # Rollback in case of error
            self.connection.rollback()
            print(f"Error saving to database: {e}")
            raise
        
    def clear_memory_storage(self):
        """Clear in-memory storage to free up memory."""
        self.cluster_results = {}
        self.pattern_results = {}
        self.config_results = {}
        self.performance_results = {}
        self.config_id_map = {}
    
    def plot_results(self, results_df, stock_symbol, timeframe_name):
        """Plot parameter testing results."""
        if results_df is None or len(results_df) == 0:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Parameter Testing Results: {stock_symbol} ({timeframe_name})", fontsize=16)
        
        # Plot 1: Weighted Score by n_pips and lookback
        pivot = results_df.pivot_table(
            index='n_pips', columns='lookback', values='weighted_score', aggfunc='mean'
        )
        sns_heatmap = sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Weighted Score')
        axes[0, 0].set_xlabel('Lookback Window')
        axes[0, 0].set_ylabel('Number of PIPs')
        
        # Plot 2: Avg Profit Factor by n_pips and hold_period
        pivot = results_df.pivot_table(
            index='n_pips', columns='hold_period', values='avg_profit_factor', aggfunc='mean'
        )
        sns_heatmap = sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis', ax=axes[0, 1])
        axes[0, 1].set_title('Average Profit Factor')
        axes[0, 1].set_xlabel('Hold Period')
        axes[0, 1].set_ylabel('Number of PIPs')
        
        # Plot 3: Reward-Risk Ratio by parameter combination
        axes[1, 0].bar(
            range(len(results_df)),
            results_df['avg_reward_risk'],
            tick_label=[f"P{row.n_pips}_L{row.lookback}_H{row.hold_period}" 
                       for _, row in results_df.iterrows()]
        )
        axes[1, 0].set_title('Average Reward-Risk Ratio')
        axes[1, 0].set_xlabel('Parameter Combination')
        axes[1, 0].set_ylabel('Reward-Risk Ratio')
        plt.setp(axes[1, 0].get_xticklabels(), rotation=90)
        
        # Plot 4: Cluster Count and Pattern Count
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        indices = range(len(results_df))
        width = 0.4
        
        ax1.bar(
            [i - width/2 for i in indices],
            results_df['num_clusters'],
            width=width,
            color='blue',
            label='Cluster Count'
        )
        ax2.bar(
            [i + width/2 for i in indices],
            results_df['num_patterns'],
            width=width,
            color='orange',
            label='Pattern Count'
        )
        
        ax1.set_xlabel('Parameter Combination')
        ax1.set_ylabel('Number of Clusters', color='blue')
        ax2.set_ylabel('Number of Patterns', color='orange')
        
        ax1.set_title('Cluster and Pattern Counts')
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f"P{row.n_pips}_L{row.lookback}_H{row.hold_period}" 
                            for _, row in results_df.iterrows()], rotation=90)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        output_file = OUTPUT_DIR / f"param_test_{stock_symbol}_{timeframe_name}.png"
        plt.savefig(output_file)
        plt.close()
        
        print(f"Results plot saved to {output_file}")
        
        # Create additional plots
        
        # Bullish vs Bearish Clusters
        plt.figure(figsize=(12, 6))
        plt.bar(
            range(len(results_df)),
            results_df['bullish_clusters'],
            label='Bullish Clusters',
            color='green',
            alpha=0.7
        )
        plt.bar(
            range(len(results_df)),
            results_df['bearish_clusters'],
            bottom=results_df['bullish_clusters'],
            label='Bearish Clusters',
            color='red',
            alpha=0.7
        )
        plt.bar(
            range(len(results_df)),
            results_df['neutral_clusters'],
            bottom=results_df['bullish_clusters'] + results_df['bearish_clusters'],
            label='Neutral Clusters',
            color='gray',
            alpha=0.7
        )
        
        plt.xlabel('Parameter Combination')
        plt.ylabel('Number of Clusters')
        plt.title(f'Cluster Distribution by Type: {stock_symbol} ({timeframe_name})')
        plt.xticks(range(len(results_df)), 
                  [f"P{row.n_pips}_L{row.lookback}_H{row.hold_period}" 
                  for _, row in results_df.iterrows()], 
                  rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        output_file = OUTPUT_DIR / f"cluster_distribution_{stock_symbol}_{timeframe_name}.png"
        plt.savefig(output_file)
        plt.close()
        
        print(f"Cluster distribution plot saved to {output_file}")
    
    def generate_report(self, results_df, stock_id, stock_symbol, timeframe_id, timeframe_name):
        """Generate a text report of parameter testing results."""
        if results_df is None or len(results_df) == 0:
            return
        
        # Sort results by weighted score
        results_df = results_df.sort_values('weighted_score', ascending=False)
        
        # Create report
        report = f"Parameter Testing Report for Gold\n"
        report += f"==============================\n\n"
        report += f"Stock: {stock_symbol} (ID: {stock_id})\n"
        report += f"Timeframe: {timeframe_name} (ID: {timeframe_id})\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += f"Top 5 Parameter Combinations by Weighted Score\n"
        report += f"-------------------------------------------\n"
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            report += (f"{i}. n_pips={row.n_pips}, lookback={row.lookback}, "
                     f"hold_period={row.hold_period}\n"
                     f"   - Weighted Score: {row.weighted_score:.2f}\n"
                     f"   - Profit Factor: {row.avg_profit_factor:.2f}\n"
                     f"   - Reward-Risk Ratio: {row.avg_reward_risk:.2f}\n"
                     f"   - Number of Clusters: {row.num_clusters}\n"
                     f"   - Number of Patterns: {row.num_patterns}\n"
                     f"   - Cluster Distribution: {row.bullish_clusters} bullish, "
                     f"{row.bearish_clusters} bearish, {row.neutral_clusters} neutral\n\n")
        
        report += f"Top 5 Parameter Combinations by Profit Factor\n"
        report += f"------------------------------------------\n"
        top_by_pf = results_df.sort_values('avg_profit_factor', ascending=False).head(5)
        for i, (_, row) in enumerate(top_by_pf.iterrows(), 1):
            report += (f"{i}. n_pips={row.n_pips}, lookback={row.lookback}, "
                     f"hold_period={row.hold_period}\n"
                     f"   - Profit Factor: {row.avg_profit_factor:.2f}\n"
                     f"   - Weighted Score: {row.weighted_score:.2f}\n"
                     f"   - Reward-Risk Ratio: {row.avg_reward_risk:.2f}\n"
                     f"   - Number of Clusters: {row.num_clusters}\n"
                     f"   - Number of Patterns: {row.num_patterns}\n\n")
        
        report += f"Parameter Impact Analysis\n"
        report += f"------------------------\n"
        
        # Analyze impact of number of PIPs
        pips_impact = results_df.groupby('n_pips').agg({
            'weighted_score': 'mean',
            'avg_profit_factor': 'mean',
            'avg_reward_risk': 'mean',
            'num_patterns': 'mean'
        }).reset_index()
        
        report += f"Impact of Number of PIPs:\n"
        for _, row in pips_impact.iterrows():
            report += (f"   - {row.n_pips} PIPs: Score={row.weighted_score:.2f}, "
                     f"PF={row.avg_profit_factor:.2f}, RR={row.avg_reward_risk:.2f}, "
                     f"Avg Patterns={row.num_patterns:.1f}\n")
        
        # Analyze impact of lookback window
        lookback_impact = results_df.groupby('lookback').agg({
            'weighted_score': 'mean',
            'avg_profit_factor': 'mean',
            'avg_reward_risk': 'mean',
            'num_patterns': 'mean'
        }).reset_index()
        
        report += f"\nImpact of Lookback Window:\n"
        for _, row in lookback_impact.iterrows():
            report += (f"   - {row.lookback} periods: Score={row.weighted_score:.2f}, "
                     f"PF={row.avg_profit_factor:.2f}, RR={row.avg_reward_risk:.2f}, "
                     f"Avg Patterns={row.num_patterns:.1f}\n")
        
        # Analyze impact of hold period
        hold_impact = results_df.groupby('hold_period').agg({
            'weighted_score': 'mean',
            'avg_profit_factor': 'mean',
            'avg_reward_risk': 'mean',
            'num_patterns': 'mean'
        }).reset_index()
        
        report += f"\nImpact of Hold Period:\n"
        for _, row in hold_impact.iterrows():
            report += (f"   - {row.hold_period} periods: Score={row.weighted_score:.2f}, "
                     f"PF={row.avg_profit_factor:.2f}, RR={row.avg_reward_risk:.2f}, "
                     f"Avg Patterns={row.num_patterns:.1f}\n")
        
        # Save report
        output_file = OUTPUT_DIR / f"param_test_{stock_symbol}_{timeframe_name}.txt"
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_file}")
        
        # Also save the raw results
        results_df.to_csv(OUTPUT_DIR / f"param_test_data_{stock_symbol}_{timeframe_name}.csv", index=False)
    def run_quick_test(self, stock_identifier, timeframe_identifier, start_date=None, end_date=None, save_to_db=True):
        """
        Run a quick test with a single parameter combination.
        
        Args:
            stock_identifier: Stock ID or symbol
            timeframe_identifier: Timeframe ID or name
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            save_to_db: If True, saves results to database after completion
        """
        try:
            # Get stock and timeframe
            stock_id, stock_symbol = self.get_stock_by_symbol_or_id(stock_identifier)
            timeframe_id, minutes, timeframe_name = self.get_timeframe_by_name_or_id(timeframe_identifier)
            
            print(f"Running quick test for {stock_symbol} on {timeframe_name} timeframe")
            
            # Run single test
            results = self.run_parameter_test(
                stock_id, timeframe_id, start_date, end_date,
                hold_period_strategy="timeframe", test_all=False, single_test=True,
                save_to_db=save_to_db
            )
            
            if results is not None and not results.empty:
                # Plot results
                self.plot_results(results, stock_symbol, timeframe_name)
                
                # Generate report
                self.generate_report(results, stock_id, stock_symbol, timeframe_id, timeframe_name)
                
                print(f"Quick test completed for {stock_symbol} on {timeframe_name}")
                return True
            else:
                print(f"Quick test failed - no results returned")
                return False
            
        except Exception as e:
            print(f"Error in quick test: {e}")
            return False
            
    def run_quick_test_all(self, stock_identifier, timeframe_identifier, start_date=None, end_date=None, save_to_db=True):
        """
        Run a test with all parameter combinations.
        
        Args:
            stock_identifier: Stock ID or symbol
            timeframe_identifier: Timeframe ID or name
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            save_to_db: If True, saves results to database after completion
        """
        try:
            # Get stock and timeframe
            stock_id, stock_symbol = self.get_stock_by_symbol_or_id(stock_identifier)
            timeframe_id, minutes, timeframe_name = self.get_timeframe_by_name_or_id(timeframe_identifier)
            
            print(f"Running test with all parameters for {stock_symbol} on {timeframe_name} timeframe")
            
            # Run all test combinations
            results = self.run_parameter_test(
                stock_id, timeframe_id, start_date, end_date,
                hold_period_strategy="timeframe", test_all=True, single_test=False,
                save_to_db=save_to_db
            )
            
            if results is not None and not results.empty:
                # Plot results
                self.plot_results(results, stock_symbol, timeframe_name)
                
                # Generate report
                self.generate_report(results, stock_id, stock_symbol, timeframe_id, timeframe_name)
                
                print(f"Test completed for {stock_symbol} on {timeframe_name}")
                return True
            else:
                print(f"Test failed - no results returned")
                return False
            
        except Exception as e:
            print(f"Error in test: {e}")
            return False
    def run_all_tests(self, stock_identifier=None, test_all_params=False, hold_period_strategy="timeframe",
                     start_date=None, end_date=None, save_to_db=True):
        """Run parameter tests for all stocks and timeframes or a specific stock.
        
        Args:
            stock_identifier: Stock ID or symbol (optional, if None runs for all stocks)
            test_all_params: If True, tests all parameter combinations
            hold_period_strategy: Strategy for hold period determination ('timeframe', 'formula')
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            save_to_db: If True, saves all results to database at once after completion
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
        
        # Turn off auto-save if we're planning to save everything at once
        save_individual_runs = False if save_to_db else True
        
        for stock_id, stock_symbol in stocks:
            for timeframe_id, minutes, timeframe_name in timeframes:
                print(f"\nTesting {stock_symbol} on {timeframe_name} timeframe")
                
                try:
                    results = self.run_parameter_test(
                        stock_id, timeframe_id, start_date, end_date,
                        hold_period_strategy, test_all=test_all_params,
                        save_to_db=save_individual_runs
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
        
        # Save all results to database at once if requested
        if save_to_db:
            print("\nSaving all results to database...")
            self.save_all_to_database()
            self.clear_memory_storage()
            
        print("\nAll parameter tests completed")
    
    def compare_hold_period_strategies(self, stock_identifier=None):
        """Compare the performance of different hold period strategies.
        
        Args:
            stock_identifier: Stock ID or symbol (optional, if None runs for all stocks)
        """
        if stock_identifier:
            # Get stock info
            stock_id, stock_symbol = self.get_stock_by_symbol_or_id(stock_identifier)
            stock_filter = f"AND pm.stock_id = {stock_id}"
        else:
            stock_filter = ""
            stock_symbol = "All Stocks"
        
        # Query the database for results from both strategies
        results =self.connection.execute(f"""
            SELECT 
                s.symbol as stock_symbol,
                t.name as timeframe_name,
                c.n_pips,
                c.lookback,
                c.hold_period,
                c.name as strategy,
                avg(pm.profit_factor) as avg_profit_factor,
                avg(pm.win_rate) as avg_win_rate
            FROM performance_metrics pm
            JOIN experiment_configs c ON pm.config_id = c.config_id
            JOIN timeframes t ON pm.timeframe_id = t.timeframe_id
            JOIN stocks s ON pm.stock_id = s.stock_id
            WHERE c.name LIKE '%timeframe%' OR c.name LIKE '%formula%'
            {stock_filter}
            GROUP BY s.symbol, t.name, c.n_pips, c.lookback, c.hold_period, c.name
            ORDER BY avg_profit_factor DESC
        """).fetchall()
        
        if not results:
            print("No comparison data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=[
            'stock_symbol', 'timeframe_name', 'n_pips', 'lookback', 
            'hold_period', 'strategy', 'avg_profit_factor', 'avg_win_rate'
        ])
        
        # Determine which strategy performs better
        timeframe_df = df.pivot_table(
            index=['timeframe_name'], 
            columns=['strategy'], 
            values=['avg_profit_factor'], 
            aggfunc='mean'
        )
        
        # Generate comparison report
        report = "Hold Period Strategy Comparison\n"
        report += "=============================\n\n"
        report += f"Stock: {stock_symbol}\n\n"
        
        report += "Performance by Timeframe:\n"
        for timeframe, data in timeframe_df.iterrows():
            report += f"  {timeframe}:\n"
            for strategy in data.index:
                if 'timeframe' in strategy[1]:
                    report += f"    Timeframe-based: PF={data[strategy]:.2f}\n"
                elif 'formula' in strategy[1]:
                    report += f"    Formula-based: PF={data[strategy]:.2f}\n"
        
        report += "\nTop 5 Timeframe-Based Strategy Combinations:\n"
        timeframe_results = df[df['strategy'].str.contains('timeframe')].sort_values('avg_profit_factor', ascending=False).head(5)
        for i, row in enumerate(timeframe_results.itertuples(), 1):
            report += f"{i}. {row.stock_symbol} ({row.timeframe_name}): n_pips={row.n_pips}, lookback={row.lookback}, hold_period={row.hold_period}, PF={row.avg_profit_factor:.2f}, WR={row.avg_win_rate:.2f}%\n"
        
        report += "\nTop 5 Formula-Based Strategy Combinations:\n"
        formula_results = df[df['strategy'].str.contains('formula')].sort_values('avg_profit_factor', ascending=False).head(5)
        for i, row in enumerate(formula_results.itertuples(), 1):
            report += f"{i}. {row.stock_symbol} ({row.timeframe_name}): n_pips={row.n_pips}, lookback={row.lookback}, hold_period={row.hold_period}, PF={row.avg_profit_factor:.2f}, WR={row.avg_win_rate:.2f}%\n"
        
        # Save report
        output_file = OUTPUT_DIR / "hold_period_strategy_comparison.txt"
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Hold period strategy comparison saved to {output_file}")
        
        # Also create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot bar chart comparing strategies by timeframe
        timeframe_names = timeframe_df.index.tolist()
        timeframe_based = [timeframe_df.loc[tf][('avg_profit_factor', col)] 
                           for tf in timeframe_names 
                           for col in timeframe_df.columns.levels[1] if 'timeframe' in col]
        formula_based = [timeframe_df.loc[tf][('avg_profit_factor', col)] 
                         for tf in timeframe_names 
                         for col in timeframe_df.columns.levels[1] if 'formula' in col]
        
        x = np.arange(len(timeframe_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, timeframe_based, width, label='Timeframe-based')
        rects2 = ax.bar(x + width/2, formula_based, width, label='Formula-based')
        
        ax.set_ylabel('Avg Profit Factor')
        ax.set_title('Hold Period Strategy Comparison by Timeframe')
        ax.set_xticks(x)
        ax.set_xticklabels(timeframe_names)
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        output_file = OUTPUT_DIR / "hold_period_strategy_comparison.png"
        plt.savefig(output_file)
        plt.close()
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print(f"Closed connection to database")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gold Parameter Optimization Framework')
    parser.add_argument('--stock', help='Stock ID or symbol to test (default: tests all)')
    parser.add_argument('--timeframe', help='Timeframe ID or name to test (default: tests all)')
    parser.add_argument('--start-date', help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data (format: YYYY-MM-DD)')
    parser.add_argument('--hold-strategy', choices=['timeframe', 'formula'], default='timeframe',
                       help='Hold period strategy (default: timeframe)')
    parser.add_argument('--test-all', action='store_true', help='Test all parameter combinations')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with default parameters')
    parser.add_argument('--compare', action='store_true', help='Compare hold period strategies')
    
    args = parser.parse_args()
    
    try:
        # Create parameter tester
        tester = ParameterTester()
        
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
        return 1
    finally:
        # Clean up resources
        if 'tester' in locals():
            tester.close()
    
    return 0

if __name__ == "__main__":
    # sys.exit(main())
    try:
        tester = ParameterTester(db_path="./Data/Storage/data.db")
        
        # Examples of using in-memory approach:
        # Single parameter test
        # tester.run_quick_test(stock_identifier=1, timeframe_identifier=3, start_date="2025-02-09", end_date="2025-05-09")
        
        # Test all parameter combinations
        tester.run_quick_test_all(stock_identifier=1, timeframe_identifier=4, start_date="2024-01-01", end_date="2025-01-01")
        
        # Test across multiple stocks/timeframes and save all at once
        # tester.run_all_tests(stock_identifier=1, test_all_params=False, hold_period_strategy="timeframe",
        #                     start_date="2024-01-01", end_date="2025-01-01", save_to_db=True)
    finally:
        if 'tester' in locals():
            tester.close()
