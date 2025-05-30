"""
Configuration Batch Testing Example

This script demonstrates how to use the batch testing module to test all 
configurations in the database or a selected subset of configurations.

Usage:
    cd c:\Users\yoonus\Documents\GitHub\Stock_AI_Predictor
    python Experements\Backtesting\run_batch_tests.py
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))

# Import custom modules
from Data.Database.db import Database
from Experements.Backtesting.batch_tester import test_all_configs, test_config, compare_configs_by_date_range

def parse_args():
    parser = argparse.ArgumentParser(description='Run batch backtests on configurations')
    
    parser.add_argument('--stock-id', type=int, default=None, 
                        help='Stock ID to test (default: test all stocks)')
    
    parser.add_argument('--timeframe-id', type=int, default=None,
                        help='Timeframe ID to test (default: test all timeframes)')
    
    parser.add_argument('--config-id', type=int, default=None,
                        help='Test a specific configuration ID (default: test all configs)')
    
    parser.add_argument('--train-start', type=str, default='2024-01-01',
                        help='Training start date (YYYY-MM-DD)')
    
    parser.add_argument('--train-end', type=str, default='2025-01-01',
                        help='Training end date (YYYY-MM-DD)')
    
    parser.add_argument('--test-start', type=str, default='2025-01-01',
                        help='Test start date (YYYY-MM-DD)')
    
    parser.add_argument('--test-end', type=str, default='2025-05-01',
                        help='Test end date (YYYY-MM-DD)')
    
    parser.add_argument('--save-reports', action='store_true',
                        help='Save performance reports for each configuration')
    
    parser.add_argument('--parallel', action='store_true',
                        help='Run tests in parallel (faster but uses more memory)')
    
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers when using parallel mode')
    
    parser.add_argument('--compare-dates', action='store_true',
                        help='Run comparison across multiple date ranges')
                        
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create database connection
    db = Database()
    
    # Convert date strings to timestamps
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    
    print(f"=== Backtesting Configuration Tester ===")
    print(f"Training period: {train_start} to {train_end}")
    print(f"Test period: {test_start} to {test_end}")
    
    if args.config_id is not None:
        # Test a single configuration
        print(f"Testing configuration {args.config_id}...")
        
        result = test_config(
            db=db,
            config_id=args.config_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start, 
            test_end=test_end,
            save_report=args.save_reports
        )
        
        if result and 'summary' in result:
            summary = result['summary']
            print("\n=== Test Results ===")
            print(f"Configuration: {summary['config_id']}")
            print(f"Stock: {summary['stock_id']}")
            print(f"Technique: {summary['technique']}")
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Win Rate: {summary['win_rate']:.2f}%")
            print(f"Total Return: {summary['total_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    
    elif args.compare_dates:
        # Run comparison across multiple date ranges
        print("Running comparison across multiple date ranges...")
        
        # Get all configs to test
        config_ids = get_configs_to_test(db, args.stock_id, args.timeframe_id)
        
        if not config_ids:
            print("No configurations found to test.")
            return
        
        # Create date ranges for testing (6 months each, spanning 2 years)
        start_dates = []
        end_dates = []
        
        base_date = test_start
        for i in range(4):  # 4 six-month periods
            period_start = base_date + timedelta(days=i*180)
            period_end = period_start + timedelta(days=180)
            start_dates.append(period_start)
            end_dates.append(period_end)
        
        print(f"Testing {len(config_ids)} configurations across {len(start_dates)} time periods...")
        
        comparison_df = compare_configs_by_date_range(
            db=db,
            config_ids=config_ids,
            start_dates=start_dates,
            end_dates=end_dates,
            train_period=365,  # 1 year training
            stock_id=args.stock_id
        )
        
        if not comparison_df.empty:
            print("\n=== Comparison Results ===")
            print(comparison_df[['config_id', 'period', 'total_trades', 'win_rate', 'total_return_pct']])
    
    else:
        # Test all configurations matching criteria
        print(f"Testing all configurations for stock_id={args.stock_id}, timeframe_id={args.timeframe_id}...")
        
        results = test_all_configs(
            db=db,
            stock_id=args.stock_id,
            timeframe_id=args.timeframe_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            save_reports=args.save_reports,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
        if 'summary' in results and not results['summary'].empty:
            summary_df = results['summary']
            
            print("\n=== Test Results ===")
            if 'error' in summary_df.columns:
                error_configs = summary_df[summary_df['error'].notna()]
                if not error_configs.empty:
                    print(f"\n{len(error_configs)} configs had errors:")
                    for _, row in error_configs.iterrows():
                        print(f"  Config {row['config_id']}: {row['error']}")
            
            # Display successful results
            if 'total_return_pct' in summary_df.columns:
                print("\nTop performing configurations:")
                top_configs = summary_df.sort_values('total_return_pct', ascending=False).head(10)
                
                for _, row in top_configs.iterrows():
                    print(f"  Config {row['config_id']}: {row['total_return_pct']:.2f}% return, "
                         f"{row['win_rate']:.2f}% win rate ({row['total_trades']} trades)")
                
                print(f"\nResults saved to: {results['summary_path']}")
        else:
            print("No valid results found.")
    
    # Close database connection
    db.close()

def get_configs_to_test(db, stock_id=None, timeframe_id=None):
    """Get configuration IDs to test based on filters."""
    configs_df = db.get_configs(stock_id=stock_id, timeframe_id=timeframe_id)
    
    if configs_df.empty:
        return []
    
    return configs_df['config_id'].tolist()

if __name__ == "__main__":
    main()
