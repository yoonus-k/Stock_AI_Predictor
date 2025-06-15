"""
Data Provider for Backtesting.py

Handles data loading from database and data preparation
for backtesting.py framework. Maintains compatibility with existing RL infrastructure.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging

from RL.Envs.Components.observations import ObservationHandler

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import existing data infrastructure
from RL.Data.Utils.loader import load_data_from_db
from Data.Database.db import Database

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Data provider for backtesting.py integration.
    
    Handles:
    - Database connectivity and data loading  
    - OHLCV data preparation for backtesting.py
    - RL observation management
    """
    
    # Timeframe mapping (consistent with existing system)
    TIMEFRAME_MAP = {
        "1M": 1, "5M": 2, "15M": 3, "30M": 4, 
        "1H": 5, "4H": 6, "1D": 7
    }
    
    def __init__(self, db_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize data provider.
        
        Args:
            db_path: Path to database (None for auto-detection)
            verbose: Enable verbose logging
        """
        self.db_path = db_path or self._find_database()
        self.verbose = verbose
        
        # Cache for loaded data
        self._data_cache = {}
        
        logger.info(f"DataProvider initialized with database: {self.db_path}")
    
    def _find_database(self) -> str:
        """Find database file in common locations."""
        possible_paths = [
            "Data/data.db",
            "RL/Data/Storage/samples.db", 
            "../Data/data.db",
            "data.db"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        raise FileNotFoundError("Database not found in common locations")
    
    def load_price_data(
        self, 
        stock_id: int = 1,
        timeframe: str = "1H",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV market data for backtesting.py.
        
        Args:
            stock_id: Stock identifier (1=GOLD by default)
            timeframe: Trading timeframe (1M, 5M, 15M, 30M, 1H, 4H, 1D)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data formatted for backtesting.py
        """
        try:
            # Get timeframe ID
            timeframe_id = self.TIMEFRAME_MAP.get(timeframe, 5)
            
            # Cache key for avoiding repeated queries
            cache_key = f"{stock_id}_{timeframe_id}_{start_date}_{end_date}"
            
            if cache_key in self._data_cache:
                if self.verbose:
                    logger.info(f"Using cached data for {cache_key}")
                return self._data_cache[cache_key]
            
            # Load data using existing infrastructure
            df = self.load_price_data_from_db(stock_id, timeframe_id)
            # limit to 200 bars 
            #df = df.head(1000)
            
            if df.empty:
                logger.warning(f"No data found for stock_id={stock_id}, timeframe={timeframe}")
                return None
            
            # Cache the result
            self._data_cache[cache_key] = df
            
            logger.info(f"Loaded {len(df)} bars for {timeframe} timeframe")
            return df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return None

    def load_price_data_from_db(self, stock_id=1, timeframe_id=5):
        """
        Load price data (OHLCV) from database with optimized processing for backtesting.py.

        Args:
            stock_id: Stock ID (1=GOLD, etc.)
            timeframe_id: Timeframe ID (5=5min, etc.)
            
        Returns:
            pd.DataFrame: Clean OHLCV data ready for backtesting.py
        """
        try:
            db = Database('Data/Storage/data.db')
            
            # Get date range from observation data if available
            try:
                obs_data = load_data_from_db(timeframe_id=timeframe_id)
                if not obs_data.empty:
                    start_date = obs_data.index.min().strftime('%Y-%m-%d %H:%M:%S')
                    end_date = obs_data.index.max().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Default date range
                    start_date = '2024-01-01 00:00:00'
                    end_date = '2024-12-31 23:59:59'
            except:
                start_date = '2024-01-01 00:00:00'
                end_date = '2024-12-31 23:59:59'
            
            # Load stock data
            stock_data = db.get_stock_data_range(
                stock_id=stock_id,
                timeframe_id=timeframe_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if stock_data.empty:
                logger.error("No stock data found in database")
                return pd.DataFrame()
            
            # Clean column names for backtesting.py (must be capitalized)
            price_data = stock_data.rename(columns={
                'open_price': 'Open',
                'high_price': 'High',
                'low_price': 'Low',
                'close_price': 'Close',
                'volume': 'Volume',
            })
            
            # Important: backtesting.py requires specific capitalized column names
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            price_data = price_data[ohlcv_columns].copy()
            
            # Make volume data more realistic if it's missing
            if 'Volume' in price_data.columns and price_data['Volume'].sum() == 0:
                # Generate synthetic volume based on price changes
                price_changes = price_data['Close'].pct_change().abs()
                price_data['Volume'] = ((price_changes * 1000) + 1).fillna(1) * 1000
                price_data['Volume'] = price_data['Volume'].astype(int)
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            return pd.DataFrame()
    
    
    def create_observation_manager(
        self,
        stock_id: int = 1,
        timeframe: str = "1H"
    ) -> Optional['ObservationManager']:
        """
        Create observation manager for RL features.
        
        Uses existing ObservationManager from Backtrader infrastructure
        to maintain compatibility with trained models.
        
        Args:
            stock_id: Stock identifier
            timeframe: Trading timeframe
            
        Returns:
            ObservationManager instance or None if creation fails
        """
        try:
            timeframe_id = self.TIMEFRAME_MAP.get(timeframe, 5)
            
            # Create data feed and observation manager using existing infrastructure
            market_data = load_data_from_db()
            
            
            if market_data.empty:
                logger.warning("No market observation data found")
                return ObservationManager()
        
            logger.info(f"Loaded {len(market_data)} market observation bars for timeframe_id={timeframe_id}")
            return ObservationManager(market_data)
                
        except Exception as e:
            logger.error(f"Error creating observation manager: {e}")
            return None
    
    def prepare_complete_dataset(
        self,
        stock_id: int = 1,
        timeframe: str = "1H",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional['ObservationManager'], Dict[str, Any]]:
        """
        Prepare complete dataset optimized for backtesting.py.
        
        Args:
            stock_id: Stock ID to load
            timeframe: Timeframe string (e.g., "1H")
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Tuple of (price_data, observation_manager, data_info)
        """
        try:
            # Load market data (OHLCV)
            price_data = self.load_price_data(
                stock_id=stock_id,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if price_data is None or price_data.empty:
                logger.warning("No price data available")
                return None, None, {}
            
            # Create observation manager for RL features
            observation_manager = self.create_observation_manager(
                stock_id=stock_id,
                timeframe=timeframe
            )
            
            # Ensure all required columns exist with correct types
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in price_data.columns:
                    logger.error(f"Required column {col} missing")
                    return None, None, {}
                
                # Convert columns to float
                try:
                    price_data[col] = price_data[col].astype(float)
                except:
                    logger.error(f"Failed to convert {col} to float")
                    return None, None, {}
            
            # Get basic data information
            data_info = {
                'total_bars': len(price_data),
                'date_range': {
                    'start': str(price_data.index.min()),
                    'end': str(price_data.index.max())
                },
                'timeframe': timeframe,
                'stock_id': stock_id
            }
            
            logger.info(f"Dataset prepared: {data_info['total_bars']} bars from "
                       f"{data_info['date_range']['start']} to {data_info['date_range']['end']}")
            
            return price_data, observation_manager, data_info
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return None, None, {}
    

class ObservationManager:
    """
    Manages RL observations by separating market features (from database) 
    and portfolio features (calculated at runtime)
    """
    
    # Market features from database (24 features)
    MARKET_FEATURE_NAMES = [
        # Pattern features (7)
        "probability", "action", "reward_risk_ratio", "max_gain", "max_drawdown", "mse", "expected_value",
        # Technical indicators (3)
        "rsi", "atr", "atr_ratio",
        # Sentiment features (1)
        "unified_sentiment",
        # COT data (6)
        "change_nonrept_long", "change_nonrept_short",
        "change_noncommercial_long", "change_noncommercial_short",
        "change_noncommercial_delta", "change_nonreportable_delta",
        # Time features (7)
        "hour_sin", "hour_cos", "day_sin", "day_cos", "asian_session", "london_session", "ny_session"
    ]
    
    # Portfolio features calculated at runtime (6 features)
    PORTFOLIO_FEATURE_NAMES = [
        "balance_ratio", "portfolio_max_drawdown", "win_rate", "avg_pnl_per_hour", "decisive_exits", "recovery_factor"
    ]
    
    # Complete feature list (24 + 6 = 30 features)
    ALL_FEATURE_NAMES = MARKET_FEATURE_NAMES + PORTFOLIO_FEATURE_NAMES
    
    def __init__(self, market_data=None):
        """
        Initialize with market observation data
        
        Args:
            market_data: DataFrame with market features from database
        """
        # check the columns, and store only the market features
        if market_data is not None:
            # Ensure only market features are stored
            market_data = market_data[self.MARKET_FEATURE_NAMES]
 
        self.market_observations = market_data if market_data is not None else pd.DataFrame()
        
        # Create a numpy-based cache for highly optimized lookups
        self._index_array = self.market_observations.index.values
        self._index_to_row = {
            ts: i for i, ts in enumerate(self.market_observations.index)
        }
        # Store values as numpy array for faster access
        self._values_array = self.market_observations.values
        
    def get_market_observation(self, datetime_key):
        """
        Get market observation features for current datetime
        
        Args:
            datetime_key: Current datetime from price data
            
        Returns:
            dict: Market observation features only
        """
        # Find closest observation by datetime
        try:
            # Get the row index directly from mapping
            row_idx = self._index_to_row[datetime_key]
            # Extract values from numpy array (much faster than pandas loc)
            features = self._values_array[row_idx]

            return features
            
        except Exception as e:
            logger.warning(f"Could not get market observation for {datetime_key}: {e}")
            return {name: 0.0 for name in self.MARKET_FEATURE_NAMES}
   