"""
Backtest CLI Module

This module provides a command-line interface for running backtests
with various parameters and configurations.

Usage:
    python backtest_cli.py --stock_id 1 --timeframe_id 2 --technique svm --start_date "2023-01-01" --end_date "2023-12-31"
"""

import argparse
import pandas as pd
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))

# Import custom modules
from Data.Database.db import Database
from Experements.Backtesting.backtest_v3 import run_backtest
from Experements.Backtesting.backtest_config import (
    BacktestConfig, RecognitionTechnique, ExitStrategy,
    create_optimization_configs, get_recommended_config
)
from Experements.Backtesting.performance_metrics import compare_techniques
from Experements.Backtesting.plot_backtest import (
    plot_performance_metrics, plot_exit_distribution, plot_trade_type_analysis
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run backtests for trading strategies.")
    
    # Basic parameters
    parser.add_argument("--stock_id", type=int, required=True, help="Stock ID to backtest")
    parser.add_argument("--timeframe_id", type=int, required=True, help="Timeframe ID to use")
    
    # Date ranges
    parser.add_argument("--train_start", type=str, default="2019-01-01", help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train_end", type=str, default="2022-12-31", help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--test_start", type=str, default="2023-01-01", help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test_end", type=str, default="2023-12-31", help="Test end date (YYYY-MM-DD)")
    
    # Recognition technique
    parser.add_argument("--technique", type=str, default="svm", 
                      choices=["svm", "random_forest", "combined", "distance_based"],
                      help="Pattern recognition technique")
    
    # Pattern parameters
    parser.add_argument("--n_pips", type=int, default=5, help="Number of perceptually important points")
    parser.add_argument("--lookback", type=int, default=24, help="Lookback period for pattern identification")
    parser.add_argument("--hold_period", type=int, default=6, help="Maximum hold period for trades")
    parser.add_argument("--mse_threshold", type=float, default=0.03, help="MSE threshold for pattern matching")
    
    # Exit strategy
    parser.add_argument("--exit_strategy", type=str, default="pattern_based",
                      choices=["pattern_based", "fixed", "trailing", "time_based", "dual"],
                      help="Exit strategy for trades")
    parser.add_argument("--tp_pct", type=float, help="Take profit percentage (for fixed exit)")
    parser.add_argument("--sl_pct", type=float, help="Stop loss percentage (for fixed exit)")
    parser.add_argument("--trailing_pct", type=float, help="Trailing stop percentage (for trailing exit)")
    parser.add_argument("--time_periods", type=int, help="Exit after N periods (for time-based exit)")
    parser.add_argument("--reward_risk_min", type=float, default=1.0, help="Minimum reward-to-risk ratio")
    
    # Special modes
    parser.add_argument("--config_id", type=int, help="Use a stored configuration ID")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--compare", action="store_true", help="Compare multiple techniques")
    parser.add_argument("--recommended", action="store_true", help="Use recommended configuration")
    
    # Output options
    parser.add_argument("--save_report", action="store_true", help="Save HTML performance report")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory for output files")
    
    return parser.parse_args()


def run_optimization(args):
    """Run parameter optimization based on command-line arguments."""
    print(f"Running parameter optimization for Stock ID {args.stock_id}, Timeframe ID {args.timeframe_id}")
    
    # Set up base configuration
    base_config = BacktestConfig(
        stock_id=args.stock_id,
        timeframe_id=args.timeframe_id,
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        test_start=pd.Timestamp(args.test_start),
        test_end=pd.Timestamp(args.test_end),
        recognition_technique=args.technique,
        n_pips=args.n_pips,
        lookback=args.lookback,
        hold_period=args.hold_period,
        mse_threshold=args.mse_threshold,
        exit_strategy=args.exit_strategy,
        reward_risk_min=args.reward_risk_min
    )
    
    # Set parameter ranges for optimization
    param_ranges = {
        'n_pips': [3, 5, 7],
        'lookback': [12, 24, 36],
        'hold_period': [4, 6, 8],
        'mse_threshold': [0.02, 0.03, 0.05],
        'reward_risk_min': [1.0, 1.5, 2.0]
    }
    
    # Create optimization configs
    configs = create_optimization_configs(base_config, param_ranges)
    print(f"Created {len(configs)} configurations for optimization")
    
    # Connect to the database
    db = Database()
    
    # Run backtests for each configuration
    results = []
    for i, config in enumerate(configs):
        print(f"\nRunning backtest {i+1}/{len(configs)}")
        print(f"  n_pips: {config.n_pips}")
        print(f"  lookback: {config.lookback}")
        print(f"  hold_period: {config.hold_period}")
        print(f"  mse_threshold: {config.mse_threshold}")
        print(f"  reward_risk_min: {config.reward_risk_min}")
        
        # Run backtest
        result = run_backtest(
            db=db,
            stock_id=config.stock_id,
            timeframe_id=config.timeframe_id,
            train_start=config.train_start,
            train_end=config.train_end,
            test_start=config.test_start,
            test_end=config.test_end,
            recognition_technique=config.recognition_technique.value,
            n_pips=config.n_pips,
            lookback=config.lookback,
            hold_period=config.hold_period,
            mse_threshold=config.mse_threshold,
            exit_strategy=config.exit_strategy.value,
            reward_risk_min=config.reward_risk_min,
            save_report=False  # Don't save reports for optimization runs
        )
        
        # Save results
        metrics = result['metrics']
        results.append({
            'n_pips': config.n_pips,
            'lookback': config.lookback,
            'hold_period': config.hold_period,
            'mse_threshold': config.mse_threshold,
            'reward_risk_min': config.reward_risk_min,
            'total_return_pct': metrics.total_return_pct,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'total_trades': metrics.total_trades
        })
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Sort by combined metric (win rate * profit factor)
    results_df['combined_score'] = results_df['win_rate'] * results_df['profit_factor']
    results_df = results_df.sort_values('combined_score', ascending=False)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(
        output_dir,
        f"optimization_stock{args.stock_id}_tf{args.timeframe_id}_{args.technique}_{timestamp}.csv"
    )
    
    results_df.to_csv(csv_path, index=False)
    print(f"\nOptimization results saved to {csv_path}")
    
    # Print top 3 configurations
    print("\nTop 3 Configurations:")
    for i, row in results_df.head(3).iterrows():
        print(f"\nRank {i+1}:")
        print(f"  n_pips: {row['n_pips']}")
        print(f"  lookback: {row['lookback']}")
        print(f"  hold_period: {row['hold_period']}")
        print(f"  mse_threshold: {row['mse_threshold']}")
        print(f"  reward_risk_min: {row['reward_risk_min']}")
        print(f"  Total Return: {row['total_return_pct']:.2f}%")
        print(f"  Win Rate: {row['win_rate']:.2f}%")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
    
    # Close database connection
    db.close()


def run_comparison(args):
    """Run comparison of different techniques based on command-line arguments."""
    print(f"Running technique comparison for Stock ID {args.stock_id}, Timeframe ID {args.timeframe_id}")
    
    # Define techniques to compare
    techniques = ["svm", "random_forest", "combined", "distance_based"]
    
    # Connect to the database
    db = Database()
    
    # Run backtests for each technique
    metrics_list = []
    for technique in techniques:
        print(f"\nRunning backtest with {technique} technique")
        
        # Run backtest
        result = run_backtest(
            db=db,
            stock_id=args.stock_id,
            timeframe_id=args.timeframe_id,
            train_start=pd.Timestamp(args.train_start),
            train_end=pd.Timestamp(args.train_end),
            test_start=pd.Timestamp(args.test_start),
            test_end=pd.Timestamp(args.test_end),
            recognition_technique=technique,
            n_pips=args.n_pips,
            lookback=args.lookback,
            hold_period=args.hold_period,
            mse_threshold=args.mse_threshold,
            exit_strategy=args.exit_strategy,
            reward_risk_min=args.reward_risk_min,
            save_report=args.save_report
        )
        
        # Save metrics
        metrics_list.append(result['metrics'])
    
    # Compare techniques
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    compare_techniques(
        metrics_list,
        title=f"Comparison of Recognition Techniques - Stock ID {args.stock_id}",
        save_path=os.path.join(
            output_dir,
            f"comparison_stock{args.stock_id}_tf{args.timeframe_id}_{timestamp}.png"
        )
    )
    
    # Close database connection
    db.close()


def main():
    """Main function to parse arguments and run backtests."""
    args = parse_arguments()
    
    # Special modes
    if args.optimize:
        run_optimization(args)
        return
    
    if args.compare:
        run_comparison(args)
        return
    
    # Connect to the database
    db = Database()
    
    # Check for recommended configuration
    if args.recommended:
        config = get_recommended_config(args.stock_id, args.timeframe_id)
        print(f"Using recommended configuration for Stock ID {args.stock_id}, Timeframe ID {args.timeframe_id}")
        
        # Run backtest with recommended config
        result = run_backtest(
            db=db,
            stock_id=args.stock_id,
            timeframe_id=args.timeframe_id,
            train_start=pd.Timestamp(args.train_start),
            train_end=pd.Timestamp(args.train_end),
            test_start=pd.Timestamp(args.test_start),
            test_end=pd.Timestamp(args.test_end),
            recognition_technique=config.recognition_technique.value,
            n_pips=config.n_pips,
            lookback=config.lookback,
            hold_period=config.hold_period,
            mse_threshold=config.mse_threshold,
            exit_strategy=config.exit_strategy.value,
            reward_risk_min=config.reward_risk_min,
            save_report=args.save_report
        )
    elif args.config_id is not None:
        # Run backtest with stored configuration
        print(f"Using stored configuration ID {args.config_id}")
        result = run_backtest(
            db=db,
            stock_id=args.stock_id,
            timeframe_id=args.timeframe_id,
            train_start=pd.Timestamp(args.train_start),
            train_end=pd.Timestamp(args.train_end),
            test_start=pd.Timestamp(args.test_start),
            test_end=pd.Timestamp(args.test_end),
            config_id=args.config_id,
            save_report=args.save_report
        )
    else:
        # Run backtest with command-line parameters
        # Prepare exit strategy parameters
        exit_params = {}
        if args.exit_strategy == "fixed" and args.tp_pct is not None and args.sl_pct is not None:
            exit_params = {
                'fixed_tp_pct': args.tp_pct,
                'fixed_sl_pct': args.sl_pct
            }
        elif args.exit_strategy == "trailing" and args.trailing_pct is not None:
            exit_params = {
                'trailing_sl_pct': args.trailing_pct
            }
        elif args.exit_strategy == "time_based" and args.time_periods is not None:
            exit_params = {
                'time_exit_periods': args.time_periods
            }
        elif args.exit_strategy == "dual":
            if args.time_periods is not None:
                exit_params['time_exit_periods'] = args.time_periods
            if args.tp_pct is not None:
                exit_params['fixed_tp_pct'] = args.tp_pct
            if args.sl_pct is not None:
                exit_params['fixed_sl_pct'] = args.sl_pct
            if args.trailing_pct is not None:
                exit_params['trailing_sl_pct'] = args.trailing_pct
        
        # Run backtest
        result = run_backtest(
            db=db,
            stock_id=args.stock_id,
            timeframe_id=args.timeframe_id,
            train_start=pd.Timestamp(args.train_start),
            train_end=pd.Timestamp(args.train_end),
            test_start=pd.Timestamp(args.test_start),
            test_end=pd.Timestamp(args.test_end),
            recognition_technique=args.technique,
            n_pips=args.n_pips,
            lookback=args.lookback,
            hold_period=args.hold_period,
            mse_threshold=args.mse_threshold,
            exit_strategy=args.exit_strategy,
            reward_risk_min=args.reward_risk_min,
            save_report=args.save_report,
            **exit_params
        )
    
    # Print metrics summary
    print("\nBacktest completed.")
    result['metrics'].print_summary()
    
    # Close database connection
    db.close()


if __name__ == "__main__":
    main()
