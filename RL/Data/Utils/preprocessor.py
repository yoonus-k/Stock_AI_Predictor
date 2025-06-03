"""
This module contains functions for preprocessing data for the RL trading model.
It includes functions for adding technical indicators, time-based features,
COT report data, and creating a unified sentiment score.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import talib
import os

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from Data.Database.db import Database


def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with price data

    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Ensure price columns exist
    required_cols = ["close_price", "high_price", "low_price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain {col} column")

    # Calculate RSI (14-period)
    if len(df) >= 24:  # Ensure enough data points
        df["rsi"] = talib.RSI(df["close_price"], timeperiod=24)
    else:
        df["rsi"] = 50  # Neutral value for insufficient data

    # Calculate ATR (14-period)
    if len(df) >= 24:
        df["atr"] = talib.ATR(
            df["high_price"], df["low_price"], df["close_price"], timeperiod=24
        )
        # Normalize ATR by price to get relative volatility
        df["atr_ratio"] = df["atr"] / df["close_price"]
    else:
        df["atr"] = 0
        df["atr_ratio"] = 0

    # Fill NaN values created by the indicators requiring lookback periods
    df.fillna(0, inplace=True)

    return df


def add_time_based_features(df):
    """
    Add time-based features to the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index or column

    Returns:
        pd.DataFrame: DataFrame with added time-based features
    """
    # Ensure datetime column exists or index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            datetime_col = df["datetime"]
        else:
            raise ValueError("DataFrame must have datetime index or column")
    else:
        datetime_col = df.index

    # Hour of day - cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * datetime_col.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * datetime_col.hour / 24)

    # Day of week - cyclical encoding
    df["day_sin"] = np.sin(2 * np.pi * datetime_col.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * datetime_col.dayofweek / 7)

    # Trading sessions (UTC time)
    df["asian_session"] = ((datetime_col.hour >= 0) & (datetime_col.hour < 8)).astype(
        float
    )
    df["london_session"] = ((datetime_col.hour >= 8) & (datetime_col.hour < 16)).astype(
        float
    )
    df["ny_session"] = ((datetime_col.hour >= 13) & (datetime_col.hour < 21)).astype(
        float
    )

    return df


def fetch_cot_data(symbol=None, start_date=None, end_date=None, file_path=None):
    """
    Load COT report data from the CSV file.

    Parameters:
        symbol (str): Symbol to fetch COT data for (used for filtering if needed)
        start_date (str): Start date in YYYY-MM-DD format (for filtering)
        end_date (str): End date in YYYY-MM-DD format (for filtering)
        file_path (str): Path to the COT data CSV file

    Returns:
        pd.DataFrame: DataFrame with COT data
    """
    # Default file path
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "XAUUSD_cot_data.csv")
    else:
        # Use the provided path directly or join with base directory
        if os.path.isabs(file_path):
            # If it's already an absolute path
            pass
        else:
            # Join with project root directory (assuming you're in a subdirectory)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            file_path = os.path.join(project_root, file_path)
    try:
        # Load the COT data from CSV
        cot_data = pd.read_csv(file_path)

        # Convert to datetime index if dates are present
        if "date" in cot_data.columns:
            cot_data["date"] = pd.to_datetime(cot_data["date"])
            cot_data.set_index("date", inplace=True)

        # Extract the specific COT fields we need
        cot_features = [
            "change_nonrept_long",
            "change_nonrept_short",
            "change_noncommercial_long",
            "change_noncommercial_short",
            "change_noncommercial_delta",
            "change_nonreportable_delta",
        ]

        # Filter for available columns
        available_cols = [col for col in cot_features if col in cot_data.columns]
        cot_data = cot_data[available_cols]

        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            cot_data = cot_data[cot_data.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            cot_data = cot_data[cot_data.index <= end_date]

        return cot_data

    except Exception as e:
        print(f"Error loading COT data from {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure


def process_cot_data(cot_df, price_df):
    """
    Process COT data and merge with price dataframe.

    Parameters:
        cot_df (pd.DataFrame): DataFrame with COT data
        price_df (pd.DataFrame): DataFrame with price data

    Returns:
        pd.DataFrame: Merged DataFrame with COT features
    """
    if cot_df.empty:
        # If COT data is unavailable, rise an error
        raise ValueError("COT data is empty or not available")
       
        
    cot_feature_names = [
                "change_nonrept_long",
                "change_nonrept_short",
                "change_noncommercial_long",
                "change_noncommercial_short",
                "change_noncommercial_delta",
                "change_nonreportable_delta",
            ]
    # Normalize column names to lowercase for consistency
    cot_df.columns = [col.lower() for col in cot_df.columns]

    # Only keep columns that are actually in the dataframe
    available_features = [col for col in cot_feature_names if col in cot_df.columns]
    cot_features = cot_df[available_features]

    # Resample to daily frequency (COT is weekly)
    cot_daily = cot_features.resample("D").ffill()

    # Ensure price_df has datetime index
    if not isinstance(price_df.index, pd.DatetimeIndex):
        if "datetime" in price_df.columns:
            price_df.set_index("datetime", inplace=True)

    # Merge with price data on dates
    merged_df = price_df.copy()

    # First add columns with zeros to ensure they exist
    for feature in cot_feature_names:
        if feature not in merged_df.columns:
            merged_df[feature] = 0.0

    # Update with actual data where available
    for date in merged_df.index:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in cot_daily.index:
            for feature in available_features:
                merged_df.at[date, feature] = cot_daily.loc[date_str, feature]
        else:
            # Use most recent available data
            available_dates = cot_daily.index[cot_daily.index < date_str]
            if len(available_dates) > 0:
                most_recent = available_dates[-1]
                for feature in available_features:
                    merged_df.at[date, feature] = cot_daily.loc[most_recent, feature]

    return merged_df


def load_sentiment_data(symbol="XAUUSD", file_path=None):
    """
    Load sentiment data from the sentiments.json file.

    Parameters:
        symbol (str): Symbol to fetch sentiment data for
        file_path (str): Path to the sentiments JSON file

    Returns:
        pd.DataFrame: DataFrame with sentiment data
    """
    # Default file path
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "sentiments.json")
    else:
        # Use the provided path directly or join with base directory
        if os.path.isabs(file_path):
            # If it's already an absolute path
            pass
        else:
            # Join with project root directory (assuming you're in a subdirectory)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            file_path = os.path.join(project_root, file_path)

    try:
        # Load the sentiment data from JSON
        with open(file_path, "r") as f:
            sentiment_data = json.load(f)
            # print(sentiment_data)

        # Extract data for the specified symbol
        if symbol in sentiment_data:
            symbol_data = sentiment_data[symbol]

            # Convert to DataFrame
            df = pd.DataFrame(symbol_data)

            # Convert date to datetime and set as index
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            return df
        else:
            print(f"Symbol {symbol} not found in sentiment data")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error loading sentiment data from {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure


def process_sentiment_data(sentiment_df, price_df):
    """
    Process sentiment data and merge with price dataframe.

    Parameters:
        sentiment_df (pd.DataFrame): DataFrame with sentiment data
        price_df (pd.DataFrame): DataFrame with price data

    Returns:
        pd.DataFrame: Merged DataFrame with sentiment features
    """
    if sentiment_df.empty:
        # If sentiment data is unavailable raise an error
        raise ValueError("Sentiment data is empty or not available")
        

    # Merge with price data on dates
    merged_df = price_df.copy()
    merged_df["unified_sentiment"] = 0.0  # Initialize with neutral sentiment
    # print(merged_df.head(5))
    # Add sentiment data for each date in price_df
    for date in merged_df.index:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in sentiment_df.index:
            normilized_sentiment = sentiment_df.loc[date_str, "normalized"]
            # Add sentiment score and count
            merged_df.at[date, "unified_sentiment"] = normilized_sentiment
        else:
            # Use most recent available data within a reasonable timeframe (e.g., 7 days)
            window_start = pd.to_datetime(date) - pd.Timedelta(days=7)
            available_dates = sentiment_df.index[
                (sentiment_df.index <= date_str) & (sentiment_df.index >= window_start)
            ]

            if len(available_dates) > 0:
                # Use the most recent sentiment within the window
                most_recent = available_dates[-1]
                merged_df.at[date, "unified_sentiment"] = sentiment_df.loc[
                    most_recent, "normalized"
                ]
    return merged_df


def preprocess_data(
    stock_data,
    symbol="XAUUSD",
    start_date="2024-01-01",
    end_date="2026-01-01",
    sentiment_file_path="Data/Raw/Sentiment/sentiments.json",
    cot_file_path="Data/Raw/Sentiment/XAUUSD_cot_data.csv",
):
    """
    Apply all preprocessing steps to the dataframe.

    Parameters:
        stock_data (pd.DataFrame): DataFrame with stock price data
        symbol (str): Symbol for which to preprocess data
        start_date (str): Start date for filtering data
        end_date (str): End date for filtering data
        sentiment_file_path (str): Path to the sentiment data JSON file
        cot_file_path (str): Path to the COT data CSV file
    Returns:
        pd.DataFrame: Processed dataframe with all features
    """
    stock_data = add_technical_indicators(stock_data)

    stock_data = add_time_based_features(stock_data)

    sentiment_data = load_sentiment_data(symbol=symbol, file_path=sentiment_file_path)
    stock_data = process_sentiment_data(sentiment_data, stock_data)

    cot_data = fetch_cot_data(
        symbol=symbol, start_date=start_date, end_date=end_date, file_path=cot_file_path
    )
    # print(cot_data.head(5))
    stock_data = process_cot_data(cot_data, stock_data)

    return stock_data


if __name__ == "__main__":
    # Example usage
    # Load a sample DataFrame (replace with actual data loading)
    # test load sentiment_data
    db = Database()

    stock_data = db.get_stock_data_range("1", "5", "2024-01-01", "2026-01-01")

    stock_data = preprocess_data(
        stock_data,
        start_date="2024-01-01",
        end_date="2026-01-01",
        sentiment_file_path="Data/Raw/Sentiment/sentiments.json",
        cot_file_path="Data/Raw/Sentiment/XAUUSD_cot_data.csv",
    )

    print(stock_data.iloc[-1])
