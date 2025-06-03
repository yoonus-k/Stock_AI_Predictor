#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preparation Script for RL Model Training

This script loads stock data, processes it with technical indicators,
time-based features, COT data, and sentiment data, then prepares training
samples for reinforcement learning models.

Usage:
    python data_preparation.py --stock_id=1 --timeframe_id=5 --start_date="2024-01-01" --end_date="2025-01-01"

Author: Yoonus
Date: June 3, 2025
"""

import os
import sys
from pathlib import Path
import cProfile
import pstats
import argparse
import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from Pattern.perceptually_important import find_pips
from sklearn.metrics import mean_squared_error
from Pattern.Utils.multi_config_recognizer import ConfigBasedRecognizer

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Data.Database.db import Database
from RL.Data.Utils.preprocessor import preprocess_data


def load_training_data(stock_id=1, timeframe_id=5, start_date="2024-01-01", end_date="2025-01-01", window_size=48):
    """
    Load training data using a configurable pattern recognizer.
    
    Args:
        stock_id: ID of the stock to process
        timeframe_id: ID of the timeframe to use
        start_date: Start date for data range
        end_date: End date for data range
        window_size: Size of price window to analyze
        
    Returns:
        list: List of dictionaries containing training samples
    """
    
    # Connect to database
    db = Database("../../data/Storage/data.db")
    
    # Fetch stock data
    stock_data = db.get_stock_data_range(
            stock_id, timeframe_id, start_date, end_date
        )
    print(f"Fetched {len(stock_data)} rows of stock data")
    
    # Preprocess data with technical indicators, time features, sentiment, and COT data
    try:
        stock_data = preprocess_data(
            stock_data,
            start_date=start_date,
            end_date=end_date,
            sentiment_file_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                         "Data/Raw/Sentiment/sentiments.json"),
            cot_file_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                   "Data/Raw/Sentiment/XAUUSD_cot_data.csv")
        )
        print("Data preprocessing completed successfully")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return []
    
    print(f"Processed data shape: {stock_data.shape}")
    close_prices = stock_data["close_price"].values
    
    # Train the pattern recognizer
    try:
        recognizer = ConfigBasedRecognizer(db, default_technique="svm")
        recognizer.train_recognizer(stock_id, timeframe_id)
        print("Pattern recognizer trained successfully")
    except Exception as e:
        print(f"Error training recognizer: {e}")
        return []

    data_samples = []
    
    # Get all configs for the stock
    configs = db.get_configs_by_stock_and_timeframe(stock_id, timeframe_id)
    print(f"Found {len(configs)} configuration patterns")
    
    # Process each window of price data
    for i in tqdm(range(window_size, len(close_prices) - window_size - 1), desc="Processing price windows"):
        date = stock_data.index[i]
        window = close_prices[i - window_size:i + 1] 
        
        # Find the best matching cluster for the current window
        best_cluster = recognizer.predict_best_cluster(stock_id, timeframe_id, window, configs)
        
        if best_cluster is None:
            continue
        
        best_cluster = best_cluster.to_dict()

        # Filter based on expected value criteria
        if best_cluster["expected_value"] < 0.01:
            continue
        
        best_cluster_config_id = best_cluster["config_id"]
        # Get the config data for the best cluster
        config = configs[configs["config_id"] == best_cluster_config_id]
        best_cluster_hold_period = config.iloc[0]["hold_period"]
    
        # === Calculate Actual Return ===
        current_price = close_prices[i]
        future_price = close_prices[i + best_cluster_hold_period]
        actual_return = (future_price - current_price) / current_price
        
        # === Calculate MFE and MAE ===
        window_mfe_mae = close_prices[i:i+best_cluster_hold_period]
        max_price = np.max(window_mfe_mae)
        min_price = np.min(window_mfe_mae)
        mfe = (max_price - current_price) / current_price
        mae = (min_price - current_price) / current_price

        # === Build Training Sample ===
        data_samples.append({
                "date": date,
                "config_id": best_cluster_config_id,
                "timeframe_id": timeframe_id,
                "probability": best_cluster["probability_score_dir"],
                "action": (lambda x: 1 if x == "Buy" else (2 if x == "Sell" else 0))(best_cluster["label"]),
                "reward_risk_ratio": best_cluster["reward_risk_ratio"],
                "max_gain": best_cluster["max_gain"],
                "max_drawdown": best_cluster["max_drawdown"],
                "mfe": mfe,
                "mae": mae,
                "mse": best_cluster["mse"],
                "actual_return": actual_return,
                "expected_value": best_cluster["expected_value"],
                "rsi": stock_data.loc[date]["rsi"],
                "atr": stock_data.loc[date]["atr"],
                "atr_ratio": stock_data.loc[date]["atr_ratio"],
                "unified_sentiment": stock_data.loc[date]["unified_sentiment"],
                "change_nonrept_long": stock_data.loc[date]["change_nonrept_long"],
                "change_nonrept_short": stock_data.loc[date]["change_nonrept_short"],
                "change_noncommercial_long": stock_data.loc[date]["change_noncommercial_long"],
                "change_noncommercial_short": stock_data.loc[date]["change_noncommercial_short"],
                "change_noncommercial_delta": stock_data.loc[date]["change_noncommercial_delta"],
                "change_nonreportable_delta": stock_data.loc[date]["change_nonreportable_delta"],
                "hour_sin": stock_data.loc[date]["hour_sin"],
                "hour_cos": stock_data.loc[date]["hour_cos"],
                "day_sin": stock_data.loc[date]["day_sin"],
                "day_cos": stock_data.loc[date]["day_cos"],
                "asian_session": stock_data.loc[date]["asian_session"],
                "london_session": stock_data.loc[date]["london_session"],
                "ny_session": stock_data.loc[date]["ny_session"]
        })

    print(f"Created {len(data_samples)} training samples")
    return data_samples


def save_data_samples(data_samples, output_csv=None, output_db=None):
    """
    Save the generated data samples to CSV and/or SQLite database
    
    Args:
        data_samples: List of data sample dictionaries
        output_csv: Path to save CSV file (optional)
        output_db: Path to save SQLite database (optional)
    """
    if len(data_samples) == 0:
        print("No data samples to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_samples)
    
    # Save to CSV if path provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} samples to CSV: {output_csv}")
    
    # Save to SQLite if path provided
    if output_db:
        conn = sqlite3.connect(output_db)
        df.to_sql("dataset", conn, if_exists="replace", index=False)
        conn.close()
        print(f"Saved {len(df)} samples to database: {output_db}")
        
        # Verify data was saved correctly
        conn = sqlite3.connect(output_db)
        verification_df = pd.read_sql_query("SELECT COUNT(*) FROM dataset", conn)
        conn.close()
        print(f"Verification: Database contains {verification_df.iloc[0, 0]} records")


def main():
    """Main function to run the data preparation process"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare data for RL model training")
    parser.add_argument("--stock_id", type=int, default=1, help="Stock ID to process")
    parser.add_argument("--timeframe_id", type=int, default=5, help="Timeframe ID to use")
    parser.add_argument("--start_date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--window_size", type=int, default=48, help="Size of price window to analyze")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--output_csv", type=str, default="data_samples_output.csv", help="Path to save CSV output")
    parser.add_argument("--output_db", type=str, default="../../RL/Data/Storage/samples.db", help="Path to save SQLite output")
    
    args = parser.parse_args()
    
    print(f"Starting data preparation for stock_id={args.stock_id}, timeframe_id={args.timeframe_id}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    
    if args.profile:
        # Run with profiling
        with cProfile.Profile() as pr:
            data_samples = load_training_data(
                stock_id=args.stock_id,
                timeframe_id=args.timeframe_id,
                start_date=args.start_date,
                end_date=args.end_date,
                window_size=args.window_size
            )
            
        # Save profiling results
        stats_enhanced = pstats.Stats(pr)
        stats_enhanced.sort_stats(pstats.SortKey.TIME)
        stats_enhanced.dump_stats('rl_data_process.prof')
        print("Profiling data saved to rl_data_process.prof")
    else:
        # Run without profiling
        data_samples = load_training_data(
            stock_id=args.stock_id,
            timeframe_id=args.timeframe_id,
            start_date=args.start_date,
            end_date=args.end_date,
            window_size=args.window_size
        )
    
    # Save results
    save_data_samples(data_samples, args.output_csv, args.output_db)


if __name__ == "__main__":
    main()
