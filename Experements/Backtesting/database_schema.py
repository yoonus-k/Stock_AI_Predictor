"""
Database Schema Module

This module defines the database schema for backtesting-related tables
and provides functions to create or update the tables.

Usage:
    Import this module to create or update database tables for backtesting.
"""

from pathlib import Path
import sys
from typing import Optional
import sqlite3


def setup_performance_metrics_table(connection: sqlite3.Connection) -> None:
    """
    Set up the performance_metrics table in the database.
    
    Args:
        connection: SQLite database connection
    """
    # Check if table exists
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        # Check if table needs updating
        cursor.execute("PRAGMA table_info(performance_metrics)")
        columns = {col[1] for col in cursor.fetchall()}
        
        # Required columns
        required_columns = {
            'metric_id', 'stock_id', 'timeframe_id', 'config_id', 'start_date', 'end_date',
            'total_trades', 'win_count', 'loss_count', 'win_rate', 'avg_win', 'avg_loss',
            'max_consecutive_wins', 'max_consecutive_losses', 'profit_factor', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'recognition_technique'
        }
        
        # Additional columns for enhanced metrics
        additional_columns = {
            'total_return_pct', 'annualized_return_pct', 'volatility', 'calmar_ratio',
            'avg_trade_duration'
        }
        
        # Check for missing columns
        missing_required = required_columns - columns
        missing_additional = additional_columns - columns
        
        # Add missing required columns
        for column in missing_required:
            data_type = "TEXT" if column in ['start_date', 'end_date', 'recognition_technique'] else "REAL"
            data_type = "INTEGER" if column in ['metric_id', 'stock_id', 'timeframe_id', 'config_id', 
                                              'total_trades', 'win_count', 'loss_count',
                                              'max_consecutive_wins', 'max_consecutive_losses'] else data_type
            
            cursor.execute(f"ALTER TABLE performance_metrics ADD COLUMN {column} {data_type}")
        
        # Add missing additional columns
        for column in missing_additional:
            cursor.execute(f"ALTER TABLE performance_metrics ADD COLUMN {column} REAL")
        
        connection.commit()
    else:
        # Create the table
        cursor.execute("""
        CREATE TABLE performance_metrics (
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
            max_consecutive_wins INTEGER,
            max_consecutive_losses INTEGER,
            profit_factor REAL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            recognition_technique TEXT,
            total_return_pct REAL,
            annualized_return_pct REAL,
            volatility REAL,
            calmar_ratio REAL,
            avg_trade_duration REAL
        )
        """)
        connection.commit()


def setup_experiment_configs_table(connection: sqlite3.Connection) -> None:
    """
    Set up the experiment_configs table in the database.
    
    Args:
        connection: SQLite database connection
    """
    # Check if table exists
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_configs'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        # Check if table needs updating
        cursor.execute("PRAGMA table_info(experiment_configs)")
        columns = {col[1] for col in cursor.fetchall()}
        
        # Required columns
        required_columns = {
            'config_id', 'name', 'stock_id', 'timeframe_id', 'n_pips', 'lookback',
            'hold_period', 'returns_hold_period', 'distance_measure', 'description', 'created_at'
        }
        
        # Additional columns for enhanced configurations
        additional_columns = {
            'recognition_technique', 'model_params', 'exit_strategy',
            'fixed_tp_pct', 'fixed_sl_pct', 'trailing_sl_pct',
            'time_exit_periods', 'reward_risk_min', 'mse_threshold'
        }
        
        # Check for missing columns
        missing_required = required_columns - columns
        missing_additional = additional_columns - columns
        
        # Add missing required columns
        for column in missing_required:
            data_type = "TEXT" if column in ['name', 'description'] else "INTEGER"
            data_type = "TIMESTAMP" if column == 'created_at' else data_type
            
            cursor.execute(f"ALTER TABLE experiment_configs ADD COLUMN {column} {data_type}")
        
        # Add missing additional columns
        for column in missing_additional:
            if column in ['recognition_technique', 'model_params', 'exit_strategy']:
                data_type = "TEXT"
            else:
                data_type = "REAL"
            
            cursor.execute(f"ALTER TABLE experiment_configs ADD COLUMN {column} {data_type}")
        
        connection.commit()
    else:
        # Create the table
        cursor.execute("""
        CREATE TABLE experiment_configs (
            config_id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            stock_id INTEGER,
            timeframe_id INTEGER NOT NULL,
            n_pips INTEGER NOT NULL,
            lookback INTEGER NOT NULL,
            hold_period INTEGER NOT NULL,
            returns_hold_period INTEGER NOT NULL,
            distance_measure INTEGER NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recognition_technique TEXT,
            model_params TEXT,
            exit_strategy TEXT,
            fixed_tp_pct REAL,
            fixed_sl_pct REAL,
            trailing_sl_pct REAL,
            time_exit_periods INTEGER,
            reward_risk_min REAL,
            mse_threshold REAL
        )
        """)
        connection.commit()


def setup_backtest_trades_table(connection: sqlite3.Connection) -> None:
    """
    Set up the backtest_trades table in the database for storing detailed trade information.
    
    Args:
        connection: SQLite database connection
    """
    # Check if table exists
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_trades'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create the table
        cursor.execute("""
        CREATE TABLE backtest_trades (
            trade_id INTEGER PRIMARY KEY,
            metric_id INTEGER,
            entry_time TEXT,
            exit_time TEXT,
            trade_type TEXT,
            entry_price REAL,
            exit_price REAL,
            return_pct REAL,
            profit_loss REAL,
            outcome TEXT,
            exit_reason TEXT,
            duration INTEGER,
            pattern_cluster_id INTEGER,
            max_gain REAL,
            max_drawdown REAL,
            reward_risk REAL,
            confidence REAL,
            FOREIGN KEY (metric_id) REFERENCES performance_metrics(metric_id)
        )
        """)
        connection.commit()


def setup_all_tables(connection: sqlite3.Connection) -> None:
    """
    Set up all backtesting-related tables in the database.
    
    Args:
        connection: SQLite database connection
    """
    setup_experiment_configs_table(connection)
    setup_performance_metrics_table(connection)
    setup_backtest_trades_table(connection)


if __name__ == "__main__":
        # Setup paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent # Navigate up to project root
    sys.path.append(str(project_root))

    from Data.Database.db import Database
    
    # Connect to the database
    db = Database()
    
    # Set up all tables
    setup_all_tables(db.connection)
    
    print("Database tables set up successfully.")
    
    # Close the connection
    db.close()
