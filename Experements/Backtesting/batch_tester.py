"""
Batch Configuration Testing Module

This module implements functions to batch test multiple backtest configurations
saved in the database. It leverages the functionality from backtest_v3.py to run
multiple backtests and aggregate their performance metrics.

Usage:
    from Experements.Backtesting.batch_tester import test_all_configs
    
    # Test all configurations in the database for a specific stock
    results = test_all_configs(
        db=db,
        stock_id=1,
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2025-01-01"),
        test_start=pd.Timestamp("2025-01-01"), 
        test_end=pd.Timestamp("2025-05-01")
    )
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import logging

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))

# Import custom modules
from Data.Database.db import Database
from Experements.Backtesting.backtest_v3 import run_backtest, Backtester
from Experements.Backtesting.backtest_config import BacktestConfig
from Experements.Backtesting.performance_metrics import PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchTester")


def test_config(
    db: Database,
    config_id: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp, 
    test_end: pd.Timestamp,
    save_report: bool = False,
    recognition_technique: str = "svm"
) -> Dict[str, Any]:
    """
    Test a specific configuration from the database.
    
    Args:
        db: Database connection
        config_id: Configuration ID to test
        train_start: Start date for training
        train_end: End date for training
        test_start: Start date for testing
        test_end: End date for testing
        save_report: Whether to save a performance report
        
    Returns:
        Dictionary containing test results or None if testing failed
    """
    try:
        # Get configuration details
        config_df = db.get_configs(config_id=config_id)
        
        if config_df.empty:
            logger.warning(f"Configuration with ID {config_id} not found")
            return None
        
        config_data = config_df.iloc[0]
        
        # Extract stock_id from config
        stock_id = int(config_data['stock_id']) if 'stock_id' in config_data and pd.notna(config_data['stock_id']) else None
        
        if stock_id is None:
            logger.warning(f"Configuration {config_id} does not have a valid stock_id")
            return None
        
        # Log the test
        logger.info(f"Testing configuration {config_id} for stock {stock_id}")
        
        # Run backtest with this configuration
        try:
            results = run_backtest(
                db=db,
                stock_id=stock_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                config_id=config_id,
                recognition_technique=recognition_technique,
                save_report=save_report
                
            )
            
            # Add configuration ID to results
            results['config_id'] = config_id
            
            # Extract key performance metrics
            metrics = results['metrics']
            results['summary'] = {
                'config_id': config_id,
                'stock_id': stock_id,
                'technique': metrics.recognition_technique,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'total_return_pct': metrics.total_return_pct
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Error testing configuration {config_id}: {e}")
            traceback.print_exc()
            return {
                'config_id': config_id,
                'error': str(e),
                'summary': {
                    'config_id': config_id,
                    'stock_id': stock_id,
                    'error': str(e)
                }
            }
    
    except Exception as e:
        logger.error(f"Error in test_config for {config_id}: {e}")
        traceback.print_exc()
        return None


def test_all_configs(
    db: Database,
    stock_id: Optional[int] = None,
    timeframe_id: Optional[int] = None,
    train_start: pd.Timestamp = pd.Timestamp("2024-01-01"),
    train_end: pd.Timestamp = pd.Timestamp("2025-01-01"),
    test_start: pd.Timestamp = pd.Timestamp("2025-01-01"), 
    test_end: pd.Timestamp = pd.Timestamp("2025-05-01"),
    recognition_technique: str = "svm",
    save_reports: bool = False,
    parallel: bool =False,
    max_workers: int = 15
) -> Dict[str, Any]:
    """
    Test all configurations in the database matching the criteria.
    
    Args:
        db: Database connection
        stock_id: Filter by stock ID (optional)
        timeframe_id: Filter by timeframe ID (optional)
        train_start: Start date for training
        train_end: End date for training
        test_start: Start date for testing
        test_end: End date for testing
        save_reports: Whether to save performance reports
        parallel: Whether to run tests in parallel
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary containing aggregated test results
    """
    # Get all relevant configurations
    configs_df = db.get_configs(stock_id=stock_id, timeframe_id=timeframe_id)
    
    if configs_df.empty:
        logger.warning(f"No configurations found for stock_id={stock_id}, timeframe_id={timeframe_id}")
        return {"error": "No configurations found", "results": []}
    
    # Extract config IDs
    config_ids = configs_df['config_id'].tolist()
    
    logger.info(f"Found {len(config_ids)} configurations to test")
    
    all_results = []
    summary_data = []
    
    if parallel and len(config_ids) > 1:
        # Run tests in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    test_config, db, config_id, train_start, train_end, test_start, test_end, save_reports , recognition_technique
                ): config_id for config_id in config_ids
            }
            
            for future in as_completed(futures):
                config_id = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                        if 'summary' in result:
                            summary_data.append(result['summary'])
                except Exception as e:
                    logger.error(f"Error testing configuration {config_id}: {e}")
                    traceback.print_exc()
    else:
        # Run tests sequentially
        for config_id in config_ids:
            result = test_config(
                db, config_id, train_start, train_end, test_start, test_end, save_reports , recognition_technique
            )
            if result is not None:
                all_results.append(result)
                if 'summary' in result:
                    summary_data.append(result['summary'])
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by performance
    if not summary_df.empty and 'total_return_pct' in summary_df.columns:
        summary_df = summary_df.sort_values('total_return_pct', ascending=False)
    
    # Save summary to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        current_dir, 
        "reports", 
        f"config_test_summary_{timestamp}.csv"
    )
    
    os.makedirs(os.path.join(current_dir, "reports"), exist_ok=True)
    
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
    
    # Create visualization of results
    if not summary_df.empty and 'total_return_pct' in summary_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Plot sorted returns
        plt.subplot(2, 1, 1)
        plt.bar(
            summary_df['config_id'].astype(str), 
            summary_df['total_return_pct'],
            color=[
                'green' if x > 0 else 'red' 
                for x in summary_df['total_return_pct']
            ]
        )
        plt.title('Total Return by Configuration')
        plt.xlabel('Configuration ID')
        plt.ylabel('Total Return (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add win rate as text on each bar
        for i, (_, row) in enumerate(summary_df.iterrows()):
            if 'win_rate' in row and 'total_trades' in row:
                plt.text(
                    i, 
                    row['total_return_pct'] + 1, 
                    f"{row['win_rate']:.1f}%\n({row['total_trades']} trades)",
                    ha='center', 
                    va='bottom', 
                    fontsize=8
                )
        
        # Plot correlation between win rate and return
        if 'win_rate' in summary_df.columns:
            plt.subplot(2, 1, 2)
            plt.scatter(
                summary_df['win_rate'], 
                summary_df['total_return_pct'],
                c=summary_df['total_return_pct'],
                cmap='RdYlGn',
                alpha=0.7
            )
            
            # Add linear trend line
            if len(summary_df) > 1:
                z = np.polyfit(summary_df['win_rate'], summary_df['total_return_pct'], 1)
                p = np.poly1d(z)
                plt.plot(
                    summary_df['win_rate'], 
                    p(summary_df['win_rate']), 
                    "r--", 
                    alpha=0.8
                )
            
            # Add config_id as annotation
            for i, row in summary_df.iterrows():
                plt.annotate(
                    str(row['config_id']),
                    (row['win_rate'], row['total_return_pct']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
            
            plt.title('Win Rate vs. Total Return')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Total Return (%)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(
            current_dir, 
            "reports", 
            f"config_test_summary_{timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Performance visualization saved to {plot_path}")
        
        # Display the plot if not running in background
        try:
            plt.show()
        except:
            logger.info("Could not display plot (likely running in non-GUI environment)")
    
    return {
        "results": all_results,
        "summary": summary_df,
        "summary_path": summary_path if not summary_df.empty else None
    }


def compare_configs_by_date_range(
    db: Database,
    config_ids: List[int],
    start_dates: List[pd.Timestamp],
    end_dates: List[pd.Timestamp],
    train_period: int = 365,  # in days
    stock_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare multiple configurations across different date ranges.
    
    Args:
        db: Database connection
        config_ids: List of configuration IDs to test
        start_dates: List of test start dates
        end_dates: List of test end dates
        train_period: Training period in days
        stock_id: Override stock_id for all configs
        
    Returns:
        DataFrame with comparison results
    """
    if len(start_dates) != len(end_dates):
        raise ValueError("start_dates and end_dates must have the same length")
    
    results = []
    
    for config_id in config_ids:
        config_df = db.get_configs(config_id=config_id)
        
        if config_df.empty:
            logger.warning(f"Configuration with ID {config_id} not found")
            continue
        
        config_data = config_df.iloc[0]
        config_stock_id = stock_id if stock_id is not None else (
            int(config_data['stock_id']) if 'stock_id' in config_data and pd.notna(config_data['stock_id']) else None
        )
        
        if config_stock_id is None:
            logger.warning(f"Configuration {config_id} does not have a valid stock_id")
            continue
        
        for i, (test_start, test_end) in enumerate(zip(start_dates, end_dates)):
            # Calculate train dates
            train_end = test_start - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=train_period)
            
            # Run backtest
            try:
                result = test_config(
                    db=db,
                    config_id=config_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    save_report=False
                )
                
                if result and 'summary' in result:
                    period_name = f"Period {i+1}: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                    summary = result['summary']
                    summary['period'] = period_name
                    summary['test_start'] = test_start
                    summary['test_end'] = test_end
                    results.append(summary)
            except Exception as e:
                logger.error(f"Error testing config {config_id} for period {i+1}: {e}")
    
    # Create and return comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Save comparison to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(
        current_dir, 
        "reports", 
        f"config_comparison_{timestamp}.csv"
    )
    
    if not comparison_df.empty:
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison saved to {comparison_path}")
        
        # Create visualization of comparison
        try:
            if len(config_ids) > 1 and len(start_dates) > 0:
                plot_config_comparison(comparison_df, timestamp)
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {e}")
    
    return comparison_df


def plot_config_comparison(comparison_df: pd.DataFrame, timestamp: str = None) -> None:
    """
    Create a visualization comparing configurations across different time periods.
    
    Args:
        comparison_df: DataFrame with comparison results
        timestamp: Timestamp for saving the plot
    """
    if comparison_df.empty:
        return
    
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a pivot table
    if 'period' in comparison_df.columns and 'config_id' in comparison_df.columns and 'total_return_pct' in comparison_df.columns:
        pivot_df = comparison_df.pivot(index='config_id', columns='period', values='total_return_pct')
        
        # Sort by average performance
        pivot_df['avg'] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values('avg', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot heatmap of returns
        plt.subplot(2, 1, 1)
        plt.pcolormesh(
            pivot_df.drop('avg', axis=1), 
            cmap='RdYlGn', 
            vmin=-20, 
            vmax=20
        )
        plt.colorbar(label='Return (%)')
        plt.yticks(
            np.arange(0.5, len(pivot_df.index)), 
            pivot_df.index
        )
        plt.xticks(
            np.arange(0.5, len(pivot_df.columns) - 1), 
            pivot_df.columns.drop('avg'),
            rotation=45
        )
        plt.title('Configuration Performance by Time Period')
        
        # Plot average performance
        plt.subplot(2, 1, 2)
        bars = plt.barh(
            pivot_df.index.astype(str), 
            pivot_df['avg'],
            color=[
                'green' if x > 0 else 'red' 
                for x in pivot_df['avg']
            ]
        )
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else 0
            plt.text(
                label_x_pos + 0.5, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                va='center'
            )
        
        plt.title('Average Performance Across All Periods')
        plt.xlabel('Average Return (%)')
        plt.ylabel('Configuration ID')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(
            current_dir, 
            "reports", 
            f"config_comparison_{timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Comparison visualization saved to {plot_path}")
        
        # Display the plot if not running in background
        try:
            plt.show()
        except:
            logger.info("Could not display plot (likely running in non-GUI environment)")


if __name__ == "__main__":
    # Example usage
    db = Database()
    
    # Test all configurations for a specific stock
    results = test_all_configs(
        db=db,
        stock_id=1,  # Gold
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2025-01-01"),
        test_start=pd.Timestamp("2024-05-01"),
        test_end=pd.Timestamp("2025-05-01"),
        save_reports=True,
        recognition_technique="svm",
    )
    
    # Print summary
    if 'summary' in results and not results['summary'].empty:
        print("\n===== Configuration Test Results =====")
        print(results['summary'][['config_id', 'total_trades', 'win_rate', 'total_return_pct', 'sharpe_ratio']])
    
    # Close database connection
    db.close()
