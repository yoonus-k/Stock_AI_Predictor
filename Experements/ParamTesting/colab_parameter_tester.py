#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Parameter Optimization for Google Colab

This script is designed to run parameter testing on Google Colab's 
high-performance environment. It includes functionality to:
1. Upload and download the database 
2. Process parameter combinations in parallel using Colab GPU/TPU resources
3. Store results back to the database
4. Generate visualization reports
5. Test all parameter combinations for all stocks and timeframes
6. Support all the same functionality as parameter_tester_multithreaded.py
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
import sqlite3
import threading
import argparse

# Add project root to path when running in Colab
try:
    # If running in Colab
    from google.colab import files, drive
    IN_COLAB = True
    # Mount Google Drive to access files
    drive.mount('/content/drive')
    # Adjust this path to where your project is stored in Drive
    project_root = '/content/drive/MyDrive/Stock_AI_Predictor'
    sys.path.append(project_root)
except ImportError:
    # If running locally (for testing)
    IN_COLAB = False
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    sys.path.append(str(project_root))
  # Import required modules
from Pattern.pip_pattern_miner import Pattern_Miner
from Experements.ParamTesting.parameter_tester import ParameterTester

# Parameter ranges to test
PARAM_RANGES = {
    'n_pips': [3, 4, 5, 6, 7, 8],
    'lookback': [12, 24, 36, 48],
    'hold_period': [3, 6, 12, 24],
}

# Fixed parameter for test
DISTANCE_MEASURE = 2  # Perpendicular distance selected as standard

class ColabParameterTester(ParameterTester):
    """Parameter tester optimized for Google Colab environment with full functionality."""
    
    def __init__(self, db_path=None, num_workers=None):
        """Initialize the Colab parameter tester.
        
        Args:
            db_path: Path to the SQLite database
            num_workers: Number of worker threads (defaults to available CPU count)
        """
        # Use the database path in Google Drive if in Colab
        if IN_COLAB and db_path is None:
            db_path = os.path.join(project_root, 'Data/Storage/data.db')
            
        # Initialize parent class
        super().__init__(db_path)
        
        # Store the db_path for thread-specific connections
        self.db_path = self.db.db_name
        
        # Thread-local storage for database connections
        self.thread_local = threading.local()
        
        # Database lock for synchronizing access
        self.db_lock = threading.RLock()
        
        # Set up multi-threading optimized for Colab
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        print(f"Initialized with {self.num_workers} worker threads on {'Colab' if IN_COLAB else 'local'} environment")
        
        # Progress tracking
        self.total_combinations = 0
        self.completed_combinations = 0

    def get_thread_db_connection(self):
        """Get a thread-specific database connection."""
        if not hasattr(self.thread_local, "connection"):
            self.thread_local.connection = sqlite3.connect(self.db_path)
        return self.thread_local.connection
        
    def get_stock_data_thread_safe(self, conn, stock_id, timeframe_id, start_date=None, end_date=None):
        """Thread-safe version of get_stock_data using a dedicated connection."""
        try:
            # Use the provided connection
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT timestamp, open, high, low, close, volume 
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
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Handle empty results
            if df.empty:
                print(f"No data found for stock_id={stock_id}, timeframe_id={timeframe_id}")
                return pd.DataFrame()
                
            # Convert timestamp to datetime index
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error getting stock data: {e}")
            return pd.DataFrame()
        
    def upload_database_to_drive(self, local_db_path):
        """Upload local database to Google Drive (when developing locally)."""
        if not IN_COLAB:
            print("This function can only be used in Google Colab")
            return
            
        # Create directory in Drive if it doesn't exist
        drive_dir = os.path.join('/content/drive/MyDrive/Stock_AI_Predictor/Data/Storage')
        os.makedirs(drive_dir, exist_ok=True)
        
        # Copy database to Drive
        import shutil
        drive_db_path = os.path.join(drive_dir, 'data.db')
        shutil.copy(local_db_path, drive_db_path)
        print(f"Database uploaded to Google Drive at {drive_db_path}")
        
        # Update the database path
        self.db_path = drive_db_path
        self.db.db_name = drive_db_path
        
    def download_database_from_drive(self, local_save_path):
        """Download database from Google Drive after processing."""
        if not IN_COLAB:
            print("This function can only be used in Google Colab")
            return
            
        # Copy database from Drive to local path
        import shutil
        drive_db_path = self.db_path
        shutil.copy(drive_db_path, local_save_path)
        print(f"Database downloaded from Google Drive to {local_save_path}")

    def run_parameter_test(self, stock_id, timeframe_id, start_date=None, end_date=None, 
                          hold_period_strategy="timeframe", test_all=False):
        """Run parameter testing for a specific stock and timeframe.
        
        This method coordinates the testing of parameter combinations.
        
        Args:
            stock_id: ID of the stock to test
            timeframe_id: ID of the timeframe to test
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            hold_period_strategy: Strategy for hold period determination ('timeframe', 'formula')
            test_all: If True, tests all parameter combinations
            
        Returns:
            DataFrame containing the test results
        """
        # Get stock and timeframe info
        stock_info = self.get_stock_info(stock_id)
        timeframe_info = self.get_timeframe_info(timeframe_id)
        
        if not stock_info or not timeframe_info:
            print(f"Stock ID {stock_id} or timeframe ID {timeframe_id} not found")
            return None
            
        stock_symbol = stock_info[1]
        timeframe_name = timeframe_info[2]
        
        print(f"\nRunning parameter tests for {stock_symbol} on {timeframe_name} timeframe")
        print(f"Using {self.num_workers} worker threads")
        
        # Get test parameter combinations
        param_combinations = self.get_test_parameter_combinations(timeframe_id, test_all)
        
        # Track number of combinations to test
        self.total_combinations = len(param_combinations)
        self.completed_combinations = 0
        
        print(f"Testing {self.total_combinations} parameter combinations")
        
        # Convert parameter tuples to argument lists
        args_list = []
        for n_pips, lookback, hold_period in param_combinations:
            args_list.append((stock_id, timeframe_id, n_pips, lookback, hold_period, start_date, end_date))
        
        # Execute tests in parallel
        results_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self._test_parameter_combination, *args) for args in args_list]
            
            # Process results as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                              desc=f"Testing {stock_symbol} ({timeframe_name})"):
                try:
                    result = future.result()
                    if result is not None:
                        results_list.append(result)
                    self.completed_combinations += 1
                except Exception as e:
                    print(f"Error in parameter testing: {e}")
        
        print(f"Completed {self.completed_combinations}/{self.total_combinations} parameter combinations")
        
        # Combine results
        if not results_list:
            print("No valid results obtained from parameter testing")
            return None
            
        results_df = pd.concat(results_list, ignore_index=True)
        
        # Sort by performance metrics        results_df.sort_values('avg_profit', ascending=False, inplace=True)
        
        print(f"Found {len(results_df)} parameter combinations with valid results")
        
        return results_df
        
    def run_parameter_testing_for_stock_timeframe_parallel(self, stock_symbol, timeframe_name, 
                                          param_ranges=None, start_date=None, end_date=None):
        """Run parameter testing for a specific stock and timeframe with optimal Colab parallelization.
        
        Args:
            stock_symbol: Stock symbol to test (e.g., 'GOLD')
            timeframe_name: Timeframe to test (e.g., 'M1', 'H1')
            param_ranges: Dictionary of parameter ranges to test
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
        """
        # Get stock and timeframe IDs
        stock_id = self.get_stock_id(stock_symbol)
        timeframe_id = self.get_timeframe_id(timeframe_name)
        
        if stock_id is None or timeframe_id is None:
            print(f"Error: Stock '{stock_symbol}' or timeframe '{timeframe_name}' not found in database")
            return
            
        # Run full parameter test
        results = self.run_parameter_test(
            stock_id, timeframe_id, start_date, end_date, 
            hold_period_strategy="timeframe", test_all=True
        )
        
        if results is not None and not results.empty:
            # Plot results
            self.plot_results(results, stock_symbol, timeframe_name)
            
            # Generate report
            report = self.generate_report(results, stock_id, stock_symbol, timeframe_id, timeframe_name)
            return results, report
        
        return None, None
    
    def _test_parameter_combination(self, stock_id, timeframe_id, n_pips, lookback, hold_period, 
                                   start_date=None, end_date=None):
        """Test a single parameter combination and store results."""
        try:
            # Create a thread-local database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get training data
            data = self.get_stock_data_thread_safe(conn, stock_id, timeframe_id, start_date, end_date)
            
            if data.empty:
                return None
                
            # Set up pattern miner with current parameters
            pattern_miner = Pattern_Miner(
                n_pips=n_pips,
                lookback_period=lookback,
                distance_measure=DISTANCE_MEASURE
            )
            
            # Extract patterns from the data
            patterns, pattern_indices = pattern_miner.mine_patterns(
                prices=data['Close'].values,
                volumes=data['Volume'].values,
                times=data.index,
                return_indices=True
            )
            
            # Calculate returns and performance metrics
            results = self.evaluate_patterns(
                data, patterns, pattern_indices, 
                lookback, hold_period, n_pips
            )
            
            # Create a config ID for this parameter combination
            config_id = self.get_config_id(n_pips, lookback, hold_period, DISTANCE_MEASURE)
            
            # Store results in the database
            self.store_parameter_test_results(
                conn, cursor, stock_id, timeframe_id, config_id, results
            )
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            return results
            
        except Exception as e:
            print(f"Error testing parameters (n_pips={n_pips}, lookback={lookback}, hold={hold_period}): {e}")
            return None
            
    def run_all_tests(self, stock_identifier=None, test_all_params=False, hold_period_strategy="timeframe",
                      start_date=None, end_date=None):
        """Run parameter tests for all stocks and timeframes or a specific stock.
        
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
                
        print(f"\nCompleted all parameter tests ({completed_tests}/{total_tests})")
        
    def run_quick_test(self, stock_identifier, timeframe_identifier, start_date=None, end_date=None):
        """Run a quick test with default parameters.
        
        Args:
            stock_identifier: Stock ID or symbol
            timeframe_identifier: Timeframe ID or name
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
        """
        # Get stock and timeframe
        stock_id, stock_symbol = self.get_stock_by_symbol_or_id(stock_identifier)
        timeframe_id, minutes, timeframe_name = self.get_timeframe_by_name_or_id(timeframe_identifier)
        
        if stock_id is None or timeframe_id is None:
            print(f"Stock or timeframe not found")
            return
            
        print(f"Running quick test for {stock_symbol} on {timeframe_name} timeframe")
        
        # Default parameters for quick test
        n_pips = 5
        lookback = 24
        hold_period = 12
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get stock data
            data = self.get_stock_data_thread_safe(conn, stock_id, timeframe_id, start_date, end_date)
            
            if data.empty:
                print("No data found")
                return
                
            # Create pattern miner
            pattern_miner = Pattern_Miner(
                n_pips=n_pips,
                lookback_period=lookback,
                distance_measure=DISTANCE_MEASURE
            )
            
            # Mine patterns
            print("Mining patterns...")
            patterns, pattern_indices = pattern_miner.mine_patterns(
                prices=data['Close'].values,
                volumes=data['Volume'].values,
                times=data.index,
                return_indices=True
            )
            
            print(f"Found {len(patterns)} patterns")
            
            # Evaluate patterns
            print("Evaluating pattern performance...")
            results = self.evaluate_patterns(
                data, patterns, pattern_indices, 
                lookback, hold_period, n_pips
            )
            
            # Display results
            print("\nQuick Test Results:")
            print(f"Stock: {stock_symbol}")
            print(f"Timeframe: {timeframe_name}")
            print(f"Parameters: n_pips={n_pips}, lookback={lookback}, hold_period={hold_period}")
            print(f"Total patterns found: {len(patterns)}")
            print(f"Profitable patterns: {results['profitable_count']} ({results['profitable_percent']:.1f}%)")
            print(f"Average profit: {results['avg_profit']:.2f}%")
            print(f"Average max gain: {results['avg_max_gain']:.2f}%")
            print(f"Average max drawdown: {results['avg_max_drawdown']:.2f}%")
            
            # Store results in database
            config_id = self.get_config_id(n_pips, lookback, hold_period, DISTANCE_MEASURE)
            self.store_parameter_test_results(
                conn, cursor, stock_id, timeframe_id, config_id, results
            )
            conn.commit()
            
        except Exception as e:
            print(f"Error in quick test: {e}")
        finally:
            conn.close()
            
    def compare_hold_period_strategies(self, stock_identifier):
        """Compare different hold period strategies.
        
        Args:
            stock_identifier: Stock ID or symbol
        """
        # Get stock info
        stock_id, stock_symbol = self.get_stock_by_symbol_or_id(stock_identifier)
        
        if stock_id is None:
            print(f"Stock {stock_identifier} not found")
            return
            
        print(f"Comparing hold period strategies for {stock_symbol}")
        
        # Get all timeframes
        _, timeframes = self.get_stocks_and_timeframes()
        
        # Default parameters
        n_pips = 5
        lookback = 24
        
        results = []
        
        for timeframe_id, minutes, timeframe_name in timeframes:
            print(f"\nTesting {timeframe_name} timeframe")
            
            try:
                # Get a database connection
                conn = sqlite3.connect(self.db_path)
                
                # Test timeframe-based hold period
                hold_period_tf = self.calculate_hold_period(timeframe_id, strategy="timeframe")
                results_tf = self._test_parameter_combination(
                    stock_id, timeframe_id, n_pips, lookback, hold_period_tf
                )
                
                # Test formula-based hold period
                hold_period_formula = self.calculate_hold_period(timeframe_id, strategy="formula")
                results_formula = self._test_parameter_combination(
                    stock_id, timeframe_id, n_pips, lookback, hold_period_formula
                )
                
                if results_tf and results_formula:
                    # Add to comparison
                    results.append({
                        'timeframe': timeframe_name,
                        'timeframe_hp': hold_period_tf,
                        'formula_hp': hold_period_formula,
                        'tf_profit': results_tf['avg_profit'],
                        'formula_profit': results_formula['avg_profit'],
                        'tf_profit_pct': results_tf['profitable_percent'],
                        'formula_profit_pct': results_formula['profitable_percent']
                    })
                
            except Exception as e:
                print(f"Error comparing strategies for {timeframe_name}: {e}")
            finally:
                conn.close()
        
        # Display comparison
        if results:
            df = pd.DataFrame(results)
            print("\nHold Period Strategy Comparison:")
            print(df)
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            
            # Plot average profit
            plt.subplot(1, 2, 1)
            plt.bar(
                [f"{row['timeframe']}\nTF:{row['timeframe_hp']}\nF:{row['formula_hp']}" for _, row in df.iterrows()],
                df['tf_profit'],
                label='Timeframe-based'
            )
            plt.bar(
                [f"{row['timeframe']}\nTF:{row['timeframe_hp']}\nF:{row['formula_hp']}" for _, row in df.iterrows()],
                df['formula_profit'],
                alpha=0.7,
                label='Formula-based'
            )
            plt.title('Average Profit Comparison')
            plt.ylabel('Average Profit (%)')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Plot profitable percent
            plt.subplot(1, 2, 2)
            plt.bar(
                [f"{row['timeframe']}\nTF:{row['timeframe_hp']}\nF:{row['formula_hp']}" for _, row in df.iterrows()],
                df['tf_profit_pct'],
                label='Timeframe-based'
            )
            plt.bar(
                [f"{row['timeframe']}\nTF:{row['timeframe_hp']}\nF:{row['formula_hp']}" for _, row in df.iterrows()],
                df['formula_profit_pct'],
                alpha=0.7,
                label='Formula-based'
            )
            plt.title('Profitable Patterns Comparison')
            plt.ylabel('Profitable Patterns (%)')
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
        else:
            print("No valid comparison results")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Parameter testing for stock pattern mining on Google Colab"
    )
    
    # Stock and timeframe arguments
    parser.add_argument('--stock', help='Stock to test (symbol or ID)')
    parser.add_argument('--timeframe', help='Timeframe to test (name or ID)')
    
    # Date range arguments
    parser.add_argument('--start-date', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data (YYYY-MM-DD)')
    
    # Testing options
    parser.add_argument('--test-all', action='store_true', help='Test all parameter combinations')
    parser.add_argument('--hold-strategy', default='timeframe', choices=['timeframe', 'formula'],
                        help='Strategy for determining hold periods')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with default parameters')
    parser.add_argument('--compare', action='store_true', help='Compare hold period strategies')
    
    # Colab-specific options
    parser.add_argument('--download-db', help='Download database to local path after testing')
    parser.add_argument('--threads', type=int, help='Number of worker threads to use (default: CPU count)')
    
    args = parser.parse_args()
    
    try:
        # Create Colab parameter tester
        tester = ColabParameterTester(num_workers=args.threads)
        
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
                tester.run_parameter_testing_for_stock_timeframe_parallel(
                    args.stock, args.timeframe, start_date=args.start_date, end_date=args.end_date
                )
            
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
        
        # Download database if requested
        if args.download_db and IN_COLAB:
            tester.download_database_from_drive(args.download_db)
        
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


# When run directly
if __name__ == "__main__":
    # If no command-line arguments, we're likely in a notebook
    if len(sys.argv) == 1:
        # Create tester instance
        tester = ColabParameterTester()
        
        # For notebook interactive use
        print("ColabParameterTester initialized for interactive use.")
        print("Example usage:")
        print("  tester.run_parameter_testing_for_stock_timeframe_parallel('GOLD', 'M1')")
        print("  tester.run_all_tests('GOLD', test_all_params=True)")
        print("  tester.run_all_tests(test_all_params=True)  # All stocks & timeframes")
    else:
        # CLI mode
        sys.exit(main())
