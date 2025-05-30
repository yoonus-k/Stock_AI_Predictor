#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for in-memory features of the parameter tester classes.
Tests both the standard and multi-threaded parameter testers.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path
import tempfile
import sqlite3
from datetime import datetime

# Add project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the classes to test
from Colab.parameter_tester import ParameterTester
from Colab.parameter_tester_multithreaded import MultiThreadedParameterTester
from Colab.pip_pattern_miner import Pattern_Miner

class MockPatternMiner:
    """Mock Pattern_Miner class for testing."""
    
    def __init__(self, num_clusters=3, patterns_per_cluster=5):
        """Initialize with mock data."""
        self._unique_pip_patterns = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [7.0, 8.0, 9.0, 10.0, 11.0],
            [8.0, 9.0, 10.0, 11.0, 12.0],
            [9.0, 10.0, 11.0, 12.0, 13.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
        ]
        
        self._returns_fixed_hold = np.array([0.8, -0.3, 0.5, -0.2, 0.1, 0.9, -0.7, 0.6, -0.4, 0.2])
        self._returns_mfe = np.array([1.2, 0.4, 0.9, 0.3, 0.7, 1.5, 0.2, 1.1, 0.6, 0.8])
        self._returns_mae = np.array([-0.4, -0.8, -0.3, -0.6, -0.2, -0.3, -1.1, -0.5, -0.9, -0.3])
        
        # Create clusters
        self._pip_clusters = []
        self._cluster_centers = []
        self._cluster_returns = []
        self._cluster_mfe = []
        self._cluster_mae = []
        self._avg_patterns_count = patterns_per_cluster
        
        # Create mock clusters
        for i in range(num_clusters):
            start_idx = i * patterns_per_cluster
            end_idx = min(start_idx + patterns_per_cluster, len(self._unique_pip_patterns))
            
            if start_idx >= len(self._unique_pip_patterns):
                break
                
            # Add cluster patterns
            self._pip_clusters.append(list(range(start_idx, end_idx)))
            
            # Create a mock cluster center (average of patterns)
            center = np.zeros_like(self._unique_pip_patterns[0], dtype=float)
            for idx in range(start_idx, end_idx):
                center += np.array(self._unique_pip_patterns[idx])
            center /= (end_idx - start_idx)
            self._cluster_centers.append(center.tolist())
            
            # Calculate mock returns
            returns = np.mean(self._returns_fixed_hold[start_idx:end_idx])
            self._cluster_returns.append(returns)
            
            # Calculate mock MFE/MAE
            mfe = np.mean(self._returns_mfe[start_idx:end_idx])
            mae = np.mean(self._returns_mae[start_idx:end_idx])
            self._cluster_mfe.append(mfe)
            self._cluster_mae.append(mae)
        
        # Add volume data for testing
        self._volume_data = np.array([
            [100, 120, 90, 110, 130],
            [120, 110, 100, 90, 95],
            [90, 100, 110, 120, 130],
            [130, 120, 110, 100, 90],
            [100, 110, 120, 110, 100],
            [90, 100, 110, 120, 130],
            [130, 120, 110, 100, 90],
            [100, 110, 120, 110, 100],
            [110, 120, 130, 120, 110],
            [100, 110, 120, 130, 140],
        ])


class TestParameterTesterInMemory(unittest.TestCase):
    """Test case for the ParameterTester in-memory features."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Initialize the database schema
        self._setup_test_database()
        
        # Create a parameter tester instance
        self.tester = ParameterTester(self.db_path)
        
        # Mock data
        self.stock_id = 1
        self.timeframe_id = 2
        self.pip_miner = MockPatternMiner()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Close the database connection
        if hasattr(self, 'tester') and self.tester:
            self.tester.close()
            
        # Remove the temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def _setup_test_database(self):
        """Set up a test database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                stock_id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                name TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create timeframes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeframes (
                timeframe_id INTEGER PRIMARY KEY,
                minutes INTEGER NOT NULL,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create experiment_configs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_configs (
                config_id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                n_pips INTEGER,
                lookback INTEGER,
                hold_period INTEGER,
                returns_hold_period INTEGER,
                distance_measure INTEGER,
                name TEXT,
                description TEXT,
                strategy_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id)
            )
        ''')
        
        # Create clusters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                config_id INTEGER,
                avg_price_points_json TEXT,
                avg_volume TEXT,
                outcome REAL,
                label TEXT,
                probability_score_dir REAL,
                probability_score_stat REAL,
                pattern_count INTEGER,
                max_gain REAL,
                max_drawdown REAL,
                reward_risk_ratio REAL,
                profit_factor REAL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id),
                FOREIGN KEY (config_id) REFERENCES experiment_configs(config_id)
            )
        ''')
        
        # Create patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                pattern_id INTEGER,
                stock_id INTEGER,
                timeframe_id INTEGER,
                config_id INTEGER,
                cluster_id INTEGER,
                price_points_json TEXT,
                volume TEXT,
                outcome REAL,
                max_gain REAL,
                max_drawdown REAL,
                label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id),
                FOREIGN KEY (config_id) REFERENCES experiment_configs(config_id),
                FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
            )
        ''')
        
        # Create performance_metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                config_id INTEGER,
                start_date TEXT,
                end_date TEXT,
                total_trades INTEGER,
                win_count INTEGER,
                loss_count INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id),
                FOREIGN KEY (config_id) REFERENCES experiment_configs(config_id)
            )
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO stocks (stock_id, symbol, name) VALUES (?, ?, ?)",
                       (1, "TEST", "Test Stock"))
        
        cursor.execute("INSERT INTO timeframes (timeframe_id, minutes, name) VALUES (?, ?, ?)",
                       (2, 60, "1h"))
        
        conn.commit()
        conn.close()
    
    def test_register_experiment_config_stores_in_memory(self):
        """Test that register_experiment_config stores data in memory."""
        n_pips = 5
        lookback = 24
        hold_period = 6
        
        # Register config
        self.tester._register_experiment_config(self.stock_id, self.timeframe_id, n_pips, lookback, hold_period)
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, n_pips, lookback, hold_period)
        self.assertIn(expected_key, self.tester.config_results)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM experiment_configs").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Config was written to database before save_all_to_database was called")
    
    def test_store_cluster_metrics_stores_in_memory(self):
        """Test that store_cluster_metrics stores data in memory."""
        # Register config first
        config_id = self.tester._register_experiment_config(self.stock_id, self.timeframe_id, 5, 24, 6)
        
        # Store cluster metrics
        self.tester._store_cluster_metrics(self.stock_id, self.timeframe_id, config_id, self.pip_miner)
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, config_id)
        self.assertIn(expected_key, self.tester.cluster_results)
        self.assertTrue(len(self.tester.cluster_results[expected_key]) > 0)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Clusters were written to database before save_all_to_database was called")
    
    def test_store_patterns_stores_in_memory(self):
        """Test that store_patterns stores data in memory."""
        # Register config first
        config_id = self.tester._register_experiment_config(self.stock_id, self.timeframe_id, 5, 24, 6)
        
        # Store patterns
        self.tester._store_patterns(self.stock_id, self.timeframe_id, config_id, self.pip_miner)
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, config_id)
        self.assertIn(expected_key, self.tester.pattern_results)
        self.assertTrue(len(self.tester.pattern_results[expected_key]) > 0)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Patterns were written to database before save_all_to_database was called")
    
    def test_store_performance_metrics_stores_in_memory(self):
        """Test that store_performance_metrics stores data in memory."""
        # Register config first
        config_id = self.tester._register_experiment_config(self.stock_id, self.timeframe_id, 5, 24, 6)
        
        # Sample metrics
        metrics = {
            'total_trades': 100,
            'win_count': 60,
            'loss_count': 40,
            'win_rate': 60.0,
            'avg_win': 1.5,
            'avg_loss': -0.8,
            'profit_factor': 2.0,
            'max_drawdown': -5.0
        }
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Store metrics
        self.tester._store_performance_metrics(
            self.stock_id, self.timeframe_id, config_id, metrics, start_date, end_date
        )
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, config_id)
        self.assertIn(expected_key, self.tester.performance_results)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM performance_metrics").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Metrics were written to database before save_all_to_database was called")
    
    def test_save_all_to_database(self):
        """Test that save_all_to_database writes all in-memory data to the database."""
        # Register config first
        config_id = self.tester._register_experiment_config(self.stock_id, self.timeframe_id, 5, 24, 6)
        
        # Store all data types
        self.tester._store_cluster_metrics(self.stock_id, self.timeframe_id, config_id, self.pip_miner)
        self.tester._store_patterns(self.stock_id, self.timeframe_id, config_id, self.pip_miner)
        
        metrics = {
            'total_trades': 100,
            'win_count': 60,
            'loss_count': 40,
            'win_rate': 60.0,
            'avg_win': 1.5,
            'avg_loss': -0.8,
            'profit_factor': 2.0,
            'max_drawdown': -5.0
        }
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        self.tester._store_performance_metrics(
            self.stock_id, self.timeframe_id, config_id, metrics, start_date, end_date
        )
        
        # Now save everything to database
        self.tester.save_all_to_database()
        
        # Verify data was written to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check configs
        config_count = cursor.execute("SELECT COUNT(*) FROM experiment_configs").fetchone()[0]
        self.assertTrue(config_count > 0, "No configs were written to database")
        
        # Check clusters
        cluster_count = cursor.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        self.assertTrue(cluster_count > 0, "No clusters were written to database")
        
        # Check patterns
        pattern_count = cursor.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        self.assertTrue(pattern_count > 0, "No patterns were written to database")
        
        # Check metrics
        metrics_count = cursor.execute("SELECT COUNT(*) FROM performance_metrics").fetchone()[0]
        self.assertTrue(metrics_count > 0, "No metrics were written to database")
        
        conn.close()
        
        # Verify memory was cleared
        self.assertEqual(len(self.tester.config_results), 0, "Config results were not cleared from memory")
        self.assertEqual(len(self.tester.cluster_results), 0, "Cluster results were not cleared from memory")
        self.assertEqual(len(self.tester.pattern_results), 0, "Pattern results were not cleared from memory")
        self.assertEqual(len(self.tester.performance_results), 0, "Performance results were not cleared from memory")
    
    def test_clear_memory_storage(self):
        """Test that clear_memory_storage clears all in-memory data."""
        # Register config first
        config_id = self.tester._register_experiment_config(self.stock_id, self.timeframe_id, 5, 24, 6)
        
        # Store all data types
        self.tester._store_cluster_metrics(self.stock_id, self.timeframe_id, config_id, self.pip_miner)
        self.tester._store_patterns(self.stock_id, self.timeframe_id, config_id, self.pip_miner)
        
        metrics = {
            'total_trades': 100,
            'win_count': 60,
            'loss_count': 40,
            'win_rate': 60.0,
            'avg_win': 1.5,
            'avg_loss': -0.8,
            'profit_factor': 2.0,
            'max_drawdown': -5.0
        }
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        self.tester._store_performance_metrics(
            self.stock_id, self.timeframe_id, config_id, metrics, start_date, end_date
        )
        
        # Verify data is in memory
        self.assertTrue(len(self.tester.config_results) > 0)
        self.assertTrue(len(self.tester.cluster_results) > 0)
        self.assertTrue(len(self.tester.pattern_results) > 0)
        self.assertTrue(len(self.tester.performance_results) > 0)
        
        # Clear memory
        self.tester.clear_memory_storage()
        
        # Verify memory was cleared
        self.assertEqual(len(self.tester.config_results), 0, "Config results were not cleared from memory")
        self.assertEqual(len(self.tester.cluster_results), 0, "Cluster results were not cleared from memory")
        self.assertEqual(len(self.tester.pattern_results), 0, "Pattern results were not cleared from memory")
        self.assertEqual(len(self.tester.performance_results), 0, "Performance results were not cleared from memory")


class TestMultiThreadedParameterTesterInMemory(unittest.TestCase):
    """Test case for the MultiThreadedParameterTester in-memory features."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Initialize the database schema
        self._setup_test_database()
        
        # Create a parameter tester instance with 2 worker threads
        self.tester = MultiThreadedParameterTester(db_path=self.db_path, num_workers=2)
        
        # Mock data
        self.stock_id = 1
        self.timeframe_id = 2
        self.pip_miner = MockPatternMiner()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Close the database connection
        if hasattr(self, 'tester') and self.tester:
            self.tester.close()
            
        # Remove the temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def _setup_test_database(self):
        """Set up a test database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                stock_id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                name TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create timeframes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeframes (
                timeframe_id INTEGER PRIMARY KEY,
                minutes INTEGER NOT NULL,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create experiment_configs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_configs (
                config_id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                n_pips INTEGER,
                lookback INTEGER,
                hold_period INTEGER,
                returns_hold_period INTEGER,
                distance_measure INTEGER,
                name TEXT,
                description TEXT,
                strategy_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id)
            )
        ''')
        
        # Create clusters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                config_id INTEGER,
                avg_price_points_json TEXT,
                avg_volume TEXT,
                outcome REAL,
                label TEXT,
                probability_score_dir REAL,
                probability_score_stat REAL,
                pattern_count INTEGER,
                max_gain REAL,
                max_drawdown REAL,
                reward_risk_ratio REAL,
                profit_factor REAL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id),
                FOREIGN KEY (config_id) REFERENCES experiment_configs(config_id)
            )
        ''')
        
        # Create patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                pattern_id INTEGER,
                stock_id INTEGER,
                timeframe_id INTEGER,
                config_id INTEGER,
                cluster_id INTEGER,
                price_points_json TEXT,
                volume TEXT,
                outcome REAL,
                max_gain REAL,
                max_drawdown REAL,
                label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id),
                FOREIGN KEY (config_id) REFERENCES experiment_configs(config_id),
                FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
            )
        ''')
        
        # Create performance_metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                config_id INTEGER,
                start_date TEXT,
                end_date TEXT,
                total_trades INTEGER,
                win_count INTEGER,
                loss_count INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id),
                FOREIGN KEY (config_id) REFERENCES experiment_configs(config_id)
            )
        ''')
        
        # Create stock_data table for testing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                timeframe_id INTEGER,
                timestamp TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(timeframe_id)
            )
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO stocks (stock_id, symbol, name) VALUES (?, ?, ?)",
                       (1, "TEST", "Test Stock"))
        
        cursor.execute("INSERT INTO timeframes (timeframe_id, minutes, name) VALUES (?, ?, ?)",
                       (2, 60, "1h"))
        
        # Insert sample price data (200 data points)
        dates = [f"2023-{(i//30)+1:02d}-{(i%30)+1:02d}" for i in range(200)]
        for i in range(200):
            # Generate sample OHLC data with a general upward trend plus some noise
            base = 100 + i * 0.1  # Base price with slight upward trend
            noise = np.random.normal(0, 1)  # Random noise
            
            open_price = base + noise
            high_price = open_price + abs(np.random.normal(0, 0.5))
            low_price = open_price - abs(np.random.normal(0, 0.5))
            close_price = (open_price + high_price + low_price) / 3 + np.random.normal(0, 0.2)
            volume = int(np.random.normal(1000, 200))
            
            cursor.execute(
                "INSERT INTO stock_data (stock_id, timeframe_id, timestamp, open_price, high_price, low_price, close_price, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (1, 2, dates[i], open_price, high_price, low_price, close_price, volume)
            )
        
        conn.commit()
        conn.close()
    
    def test_register_experiment_config_thread_safe(self):
        """Test that register_experiment_config_thread_safe stores data in memory."""
        thread_conn = self.tester.get_thread_db_connection()
        n_pips = 5
        lookback = 24
        hold_period = 6
        
        # Register config
        config_id = self.tester.register_experiment_config_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, n_pips, lookback, hold_period
        )
        
        # Verify it's stored in memory and has a temporary ID
        self.assertTrue(config_id < 0, "Config ID should be negative for temporary IDs")
        
        expected_key = (self.stock_id, self.timeframe_id, n_pips, lookback, hold_period)
        self.assertIn(expected_key, self.tester.config_results)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM experiment_configs").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Config was written to database before write_all_results_to_database was called")
    
    def test_store_cluster_metrics_thread_safe(self):
        """Test that store_cluster_metrics_thread_safe stores data in memory."""
        thread_conn = self.tester.get_thread_db_connection()
        
        # Register config first
        config_id = self.tester.register_experiment_config_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, 5, 24, 6
        )
        
        # Store cluster metrics
        self.tester.store_cluster_metrics_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, config_id, self.pip_miner
        )
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, config_id)
        self.assertIn(expected_key, self.tester.cluster_results)
        self.assertTrue(len(self.tester.cluster_results[expected_key]) > 0)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Clusters were written to database before write_all_results_to_database was called")
    
    def test_store_patterns_thread_safe(self):
        """Test that store_patterns_thread_safe stores data in memory."""
        thread_conn = self.tester.get_thread_db_connection()
        
        # Register config first
        config_id = self.tester.register_experiment_config_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, 5, 24, 6
        )
        
        # Store patterns
        self.tester.store_patterns_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, config_id, self.pip_miner
        )
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, config_id)
        self.assertIn(expected_key, self.tester.pattern_results)
        self.assertTrue(len(self.tester.pattern_results[expected_key]) > 0)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Patterns were written to database before write_all_results_to_database was called")
    
    def test_store_performance_metrics_thread_safe(self):
        """Test that store_performance_metrics_thread_safe stores data in memory."""
        thread_conn = self.tester.get_thread_db_connection()
        
        # Register config first
        config_id = self.tester.register_experiment_config_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, 5, 24, 6
        )
        
        # Sample metrics
        metrics = {
            'total_trades': 100,
            'win_count': 60,
            'loss_count': 40,
            'win_rate': 60.0,
            'avg_win': 1.5,
            'avg_loss': -0.8,
            'profit_factor': 2.0,
            'max_drawdown': -5.0
        }
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Store metrics
        self.tester.store_performance_metrics_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, config_id, metrics, start_date, end_date
        )
        
        # Verify it's stored in memory
        expected_key = (self.stock_id, self.timeframe_id, config_id)
        self.assertIn(expected_key, self.tester.performance_results)
        
        # Verify no database write has occurred yet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM performance_metrics").fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0, "Metrics were written to database before write_all_results_to_database was called")
    
    def test_write_all_results_to_database(self):
        """Test that write_all_results_to_database writes all in-memory data to the database."""
        thread_conn = self.tester.get_thread_db_connection()
        
        # Register config first
        config_id = self.tester.register_experiment_config_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, 5, 24, 6
        )
        
        # Store all data types
        self.tester.store_cluster_metrics_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, config_id, self.pip_miner
        )
        
        self.tester.store_patterns_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, config_id, self.pip_miner
        )
        
        metrics = {
            'total_trades': 100,
            'win_count': 60,
            'loss_count': 40,
            'win_rate': 60.0,
            'avg_win': 1.5,
            'avg_loss': -0.8,
            'profit_factor': 2.0,
            'max_drawdown': -5.0
        }
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        self.tester.store_performance_metrics_thread_safe(
            thread_conn, self.stock_id, self.timeframe_id, config_id, metrics, start_date, end_date
        )
        
        # Verify data is in memory
        self.assertTrue(len(self.tester.config_results) > 0)
        self.assertTrue(len(self.tester.cluster_results) > 0)
        self.assertTrue(len(self.tester.pattern_results) > 0)
        self.assertTrue(len(self.tester.performance_results) > 0)
        
        # Now save everything to database
        result = self.tester.write_all_results_to_database()
        
        # Check the result
        self.assertTrue(result, "write_all_results_to_database should return True on success")
        
        # Verify data was written to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check configs
        config_count = cursor.execute("SELECT COUNT(*) FROM experiment_configs").fetchone()[0]
        self.assertTrue(config_count > 0, "No configs were written to database")
        
        # Check clusters
        cluster_count = cursor.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        self.assertTrue(cluster_count > 0, "No clusters were written to database")
        
        # Check patterns
        pattern_count = cursor.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        self.assertTrue(pattern_count > 0, "No patterns were written to database")
        
        # Check metrics
        metrics_count = cursor.execute("SELECT COUNT(*) FROM performance_metrics").fetchone()[0]
        self.assertTrue(metrics_count > 0, "No metrics were written to database")
        
        conn.close()
        
        # Verify memory was cleared
        self.assertEqual(len(self.tester.config_results), 0, "Config results were not cleared from memory")
        self.assertEqual(len(self.tester.cluster_results), 0, "Cluster results were not cleared from memory")
        self.assertEqual(len(self.tester.pattern_results), 0, "Pattern results were not cleared from memory")
        self.assertEqual(len(self.tester.performance_results), 0, "Performance results were not cleared from memory")
    
    def test_thread_db_connections(self):
        """Test that thread-specific database connections work correctly."""
        # Get connections from different "threads"
        conn1 = self.tester.get_thread_db_connection()
        
        # Store the current thread local
        current_thread_local = self.tester.thread_local
        
        # Reset thread local to simulate a different thread
        self.tester.thread_local = threading.local()
        conn2 = self.tester.get_thread_db_connection()
        
        # Restore original thread local
        self.tester.thread_local = current_thread_local
        
        # Verify connections are different objects
        self.assertIsNot(conn1, conn2, "Thread connections should be different objects")
        
        # Verify both connections work
        cursor1 = conn1.cursor()
        result1 = cursor1.execute("SELECT * FROM stocks").fetchone()
        self.assertIsNotNone(result1, "First connection should be able to query database")
        
        cursor2 = conn2.cursor()
        result2 = cursor2.execute("SELECT * FROM stocks").fetchone()
        self.assertIsNotNone(result2, "Second connection should be able to query database")


if __name__ == "__main__":
    unittest.main()
