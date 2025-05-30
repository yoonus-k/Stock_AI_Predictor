#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Uploader for Hugging Face Datasets

This script prepares and uploads the trading data to Hugging Face Datasets.
It exports data from the SQLite database and formats it for the Datasets library.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset, DatasetDict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import database module
from Data.Database.db import Database

def export_stock_prices(db: Database, stock_id: int, timeframe_id: int) -> pd.DataFrame:
    """
    Export stock price data from the database
    
    Args:
        db: Database instance
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        
    Returns:
        pd.DataFrame: Stock price data
    """
    query = """
    SELECT * FROM prices
    WHERE stock_id = ? AND timeframe_id = ?
    ORDER BY datetime
    """
    
    df = pd.read_sql_query(query, db.connection, params=(stock_id, timeframe_id))
    return df

def export_patterns(db: Database, stock_id: int, timeframe_id: int) -> pd.DataFrame:
    """
    Export pattern data from the database
    
    Args:
        db: Database instance
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        
    Returns:
        pd.DataFrame: Pattern data
    """
    query = """
    SELECT * FROM patterns
    WHERE stock_id = ? AND timeframe_id = ?
    ORDER BY datetime
    """
    
    df = pd.read_sql_query(query, db.connection, params=(stock_id, timeframe_id))
    return df

def export_clusters(db: Database, stock_id: int, timeframe_id: int) -> pd.DataFrame:
    """
    Export cluster data from the database
    
    Args:
        db: Database instance
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        
    Returns:
        pd.DataFrame: Cluster data
    """
    query = """
    SELECT * FROM clusters
    WHERE stock_id = ? AND timeframe_id = ?
    """
    
    df = pd.read_sql_query(query, db.connection, params=(stock_id, timeframe_id))
    return df

def export_sentiment(db: Database, stock_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Export sentiment data from the database
    
    Args:
        db: Database instance
        stock_id: Stock ID
        start_date: Start date
        end_date: End date
        
    Returns:
        pd.DataFrame: Sentiment data
    """
    # Query articles
    articles_query = """
    SELECT * FROM articles
    WHERE stock_id = ? AND datetime BETWEEN ? AND ?
    ORDER BY datetime
    """
    
    # Query tweets
    tweets_query = """
    SELECT * FROM tweets
    WHERE stock_id = ? AND datetime BETWEEN ? AND ?
    ORDER BY datetime
    """
    
    # Get data
    articles_df = pd.read_sql_query(articles_query, db.connection, params=(stock_id, start_date, end_date))
    tweets_df = pd.read_sql_query(tweets_query, db.connection, params=(stock_id, start_date, end_date))
    
    # Combine data
    articles_df["source"] = "article"
    tweets_df["source"] = "tweet"
    
    # Select common columns
    common_columns = ["datetime", "sentiment_score", "source", "stock_id"]
    
    # Concatenate dataframes
    sentiment_df = pd.concat([
        articles_df[common_columns],
        tweets_df[common_columns]
    ], ignore_index=True)
    
    # Sort by datetime
    sentiment_df = sentiment_df.sort_values("datetime")
    
    return sentiment_df

def prepare_dataset(db_path: str, output_dir: str) -> Dict[str, Dataset]:
    """
    Prepare dataset for Hugging Face Datasets
    
    Args:
        db_path: Path to the database file
        output_dir: Output directory for dataset files
        
    Returns:
        Dict[str, Dataset]: Dataset dictionary
    """
    # Connect to database
    db = Database(db_path)
    
    # Get stock and timeframe IDs
    stocks_query = "SELECT id FROM stocks"
    timeframes_query = "SELECT id FROM timeframes"
    
    stock_ids = db.execute_query(stocks_query).fetchall()
    timeframe_ids = db.execute_query(timeframes_query).fetchall()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare datasets
    datasets_dict = {}
    
    for stock_id in [x[0] for x in stock_ids]:
        for timeframe_id in [x[0] for x in timeframe_ids]:
            # Get stock and timeframe names
            stock_name = db.execute_query(
                "SELECT name FROM stocks WHERE id = ?", 
                (stock_id,)
            ).fetchone()[0]
            
            timeframe_name = db.execute_query(
                "SELECT name FROM timeframes WHERE id = ?", 
                (timeframe_id,)
            ).fetchone()[0]
            
            # Export data
            try:
                prices_df = export_stock_prices(db, stock_id, timeframe_id)
                patterns_df = export_patterns(db, stock_id, timeframe_id)
                clusters_df = export_clusters(db, stock_id, timeframe_id)
                
                # Skip if no data
                if len(prices_df) == 0:
                    continue
                
                # Get date range for sentiment data
                start_date = prices_df["datetime"].min()
                end_date = prices_df["datetime"].max()
                
                sentiment_df = export_sentiment(db, stock_id, start_date, end_date)
                
                # Convert to datasets
                prices_dataset = Dataset.from_pandas(prices_df)
                patterns_dataset = Dataset.from_pandas(patterns_df) if len(patterns_df) > 0 else None
                clusters_dataset = Dataset.from_pandas(clusters_df) if len(clusters_df) > 0 else None
                sentiment_dataset = Dataset.from_pandas(sentiment_df) if len(sentiment_df) > 0 else None
                
                # Create dataset dictionary
                dataset_key = f"{stock_name}_{timeframe_name}"
                datasets_dict[dataset_key] = DatasetDict({
                    "prices": prices_dataset,
                    "patterns": patterns_dataset if patterns_dataset else Dataset.from_dict({}),
                    "clusters": clusters_dataset if clusters_dataset else Dataset.from_dict({}),
                    "sentiment": sentiment_dataset if sentiment_dataset else Dataset.from_dict({})
                })
                
                # Save dataset to JSON files
                output_subdir = os.path.join(output_dir, dataset_key)
                os.makedirs(output_subdir, exist_ok=True)
                
                prices_df.to_json(os.path.join(output_subdir, "prices.json"), orient="records", date_format="iso")
                
                if len(patterns_df) > 0:
                    patterns_df.to_json(os.path.join(output_subdir, "patterns.json"), orient="records", date_format="iso")
                
                if len(clusters_df) > 0:
                    clusters_df.to_json(os.path.join(output_subdir, "clusters.json"), orient="records", date_format="iso")
                
                if len(sentiment_df) > 0:
                    sentiment_df.to_json(os.path.join(output_subdir, "sentiment.json"), orient="records", date_format="iso")
                
                print(f"Exported dataset for {dataset_key}")
            except Exception as e:
                print(f"Error exporting dataset for {stock_id}_{timeframe_id}: {e}")
    
    # Close database connection
    db.close()
    
    return datasets_dict

def upload_to_hub(dataset_dir: str, repo_id: str, token: str) -> None:
    """
    Upload dataset to Hugging Face Datasets
    
    Args:
        dataset_dir: Path to the dataset directory
        repo_id: Repository ID on Hugging Face Datasets
        token: Hugging Face API token
    """
    from huggingface_hub import HfApi
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create or get repository
    api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
    
    # Upload dataset files
    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    print(f"Dataset successfully uploaded to {repo_id}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Uploader for Hugging Face Datasets")
    parser.add_argument("--db_path", type=str, default="Data/Storage/data.db", help="Path to the database file")
    parser.add_argument("--output_dir", type=str, default="dataset_export", help="Output directory for dataset files")
    parser.add_argument("--upload", action="store_true", help="Upload dataset to Hugging Face Datasets")
    parser.add_argument("--repo_id", type=str, help="Repository ID on Hugging Face Datasets")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    
    args = parser.parse_args()
    
    # Prepare dataset
    datasets_dict = prepare_dataset(args.db_path, args.output_dir)
    
    # Upload to hub if requested
    if args.upload:
        if not args.repo_id or not args.token:
            raise ValueError("--repo_id and --token are required for uploading")
        
        upload_to_hub(args.output_dir, args.repo_id, args.token)

if __name__ == "__main__":
    main()
