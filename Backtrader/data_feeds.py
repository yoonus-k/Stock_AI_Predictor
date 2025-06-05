"""
Simplified Data Feed for Backtrader - OHLCV Only

Clean separation: 
- PandasData handles only price data (OHLCV)
- Strategy handles RL observations separately
- No synthetic data generation
"""

import sys
import os
from pathlib import Path
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from RL.Data.Utils.loader import load_data_from_db
from Data.Database.db import Database

logger = logging.getLogger(__name__)


class SimplePriceFeed(bt.feeds.PandasData):
    """
    Simple price-only data feed for Backtrader
    Just handles OHLCV - no complex features
    """
    
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )


class PortfolioFeatureCalculator:
    """
    Calculates portfolio features from backtrader runtime data
    These features are not stored in the database but computed dynamically
    """
    
    def __init__(self):
        """Initialize portfolio feature calculator"""
        self.initial_balance = 10000.0
        self.current_balance = 10000.0
        self.peak_balance = 10000.0
        self.max_drawdown = 0.0
        self.winning_trades = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.total_holding_hours = 1.0
        self.tp_exits = 0
        self.sl_exits = 0
        self.timeout_exits = 0
        self.total_gains = 0.0
        
    def reset(self, initial_balance: float = 10000.0):
        """Reset calculator for new trading session"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.winning_trades = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.total_holding_hours = 1.0
        self.tp_exits = 0
        self.sl_exits = 0
        self.timeout_exits = 0
        self.total_gains = 0.0
    
    def update_balance(self, new_balance: float):
        """Update current balance and recalculate drawdown"""
        self.current_balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)
        
        # Calculate current drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def record_trade(self, pnl: float, exit_reason: str = 'unknown', holding_hours: float = 1.0):
        """Record completed trade"""
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_holding_hours += holding_hours
        
        if pnl > 0:
            self.winning_trades += 1
            self.total_gains += pnl
            
        # Track exit reasons
        if exit_reason == 'tp':
            self.tp_exits += 1
        elif exit_reason == 'sl':
            self.sl_exits += 1
        elif exit_reason in ['timeout', 'time']:
            self.timeout_exits += 1
    
    def get_portfolio_features(self) -> np.array:
        """
        Calculate current portfolio features
        
        Returns:
            np.array: [balance_ratio, portfolio_max_drawdown, win_rate, avg_pnl_per_hour, decisive_exits, recovery_factor]
        """
        # 1. Balance ratio (current balance / initial balance)
        balance_ratio = self.current_balance / self.initial_balance if self.initial_balance > 0 else 1.0
        
        # 2. Portfolio max drawdown (negative value)
        portfolio_max_drawdown = -self.max_drawdown
        
        # 3. Win rate (winning trades / total trades)
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # 4. Average P&L per hour
        avg_pnl_per_hour = self.total_pnl / self.total_holding_hours if self.total_holding_hours > 0 else 0.0
        
        # 5. Decisive exits ratio (TP/SL vs timeout)
        total_exits = self.tp_exits + self.sl_exits + self.timeout_exits
        decisive_exits = (self.tp_exits + self.sl_exits) / total_exits if total_exits > 0 else 0.0
        
        # 6. Recovery factor (gains / max drawdown)
        recovery_factor = self.total_gains / max(self.max_drawdown, 0.001) if self.max_drawdown > 0 else 1.0
          # Validate and clip values to prevent overflow
        balance_ratio = np.clip(balance_ratio, -1e6, 1e6)
        portfolio_max_drawdown = np.clip(portfolio_max_drawdown, -1.0, 0.0)
        win_rate = np.clip(win_rate, 0.0, 1.0)
        avg_pnl_per_hour = np.clip(avg_pnl_per_hour, -1e6, 1e6)
        decisive_exits = np.clip(decisive_exits, 0.0, 1.0)
        recovery_factor = np.clip(recovery_factor, -1e6, 1e6)
        
        return np.array([
            balance_ratio,
            portfolio_max_drawdown,
            win_rate,
            avg_pnl_per_hour,
            decisive_exits,
            recovery_factor
        ], dtype=np.float32)


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
    def __init__(self, market_data=None, portfolio_calculator=None):
        """
        Initialize with market observation data and portfolio calculator
        
        Args:
            market_data: DataFrame with market features from database
            portfolio_calculator: PortfolioFeatureCalculator instance
        """
        self.market_observations = market_data if market_data is not None else pd.DataFrame()
        self.portfolio_calculator = portfolio_calculator if portfolio_calculator is not None else PortfolioFeatureCalculator()
        self.current_index = 0
        
    def get_market_observation(self, datetime_key):
        """
        Get market observation features for current datetime
        
        Args:
            datetime_key: Current datetime from price data
            
        Returns:
            dict: Market observation features only
        """
        if self.market_observations.empty:
            return {name: 0.0 for name in self.MARKET_FEATURE_NAMES}
        
        # Find closest observation by datetime
        try:
            if datetime_key in self.market_observations.index:
                row = self.market_observations.loc[datetime_key]
            else:
                # Find nearest datetime
                idx = self.market_observations.index.get_loc(datetime_key, method='nearest')
                row = self.market_observations.iloc[idx]
            
            # Convert to feature dictionary (market features only)
            features = {}
            for feature_name in self.MARKET_FEATURE_NAMES:
                if feature_name in row:
                    features[feature_name] = row[feature_name]
                else:
                    features[feature_name] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Could not get market observation for {datetime_key}: {e}")
            return {name: 0.0 for name in self.MARKET_FEATURE_NAMES}
    
    def get_complete_observation(self, datetime_key):
        """
        Get complete observation with both market and portfolio features
        
        Args:
            datetime_key: Current datetime
            
        Returns:
            dict: Complete observation features (market + portfolio)
        """
        # Get market features
        market_features = self.get_market_observation(datetime_key)
        
        # Get portfolio features
        portfolio_features_array = self.portfolio_calculator.get_portfolio_features()
        portfolio_features = {
            name: portfolio_features_array[i] 
            for i, name in enumerate(self.PORTFOLIO_FEATURE_NAMES)
        }
        
        # Combine all features
        complete_features = {**market_features, **portfolio_features}
        return complete_features
    
    def get_observation_array(self, datetime_key):
        """
        Get complete observation as array (for model prediction)
        
        Args:
            datetime_key: Current datetime
            
        Returns:
            np.array: Complete observation array for RL model (market + portfolio features)
        """
        # Get market features as array
        market_features = self.get_market_observation(datetime_key)
        market_array = np.array([market_features[name] for name in self.MARKET_FEATURE_NAMES])
        
        # Get portfolio features as array
        portfolio_array = self.portfolio_calculator.get_portfolio_features()
        
        # Concatenate market and portfolio features
        complete_observation = np.concatenate([market_array, portfolio_array])
        return complete_observation
    
    def get_normalized_observation_array(self, datetime_key, normalizer=None):
        """
        Get normalized complete observation array
        
        Args:
            datetime_key: Current datetime
            normalizer: ObservationNormalizer instance (optional)
            
        Returns:
            np.array: Normalized complete observation array
        """
        observation_array = self.get_observation_array(datetime_key)
        
        if normalizer is not None:
            observation_array = normalizer.normalize_observation(observation_array)
            
        return observation_array


def load_price_data_from_db(stock_id=1, timeframe_id=5):
    """
    Load only price data (OHLCV) from database
    
    Args:
        stock_id: Stock ID (1=GOLD, etc.)
        timeframe_id: Timeframe ID (5=5min, etc.)
        
    Returns:
        pd.DataFrame: Clean OHLCV data ready for Backtrader
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
        
        # Clean column names for Backtrader
        price_data = stock_data.rename(columns={
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close'
        })
        
        # Keep only OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        price_data = price_data[ohlcv_columns].copy()
        
        # Clean data
        price_data = price_data.dropna()
        price_data = price_data[price_data > 0].dropna()  # Remove negative/zero prices
        
        logger.info(f"Loaded {len(price_data)} price bars for stock_id={stock_id}")
        return price_data
        
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        return pd.DataFrame()


def load_observation_data_from_db(timeframe_id=5):
    """
    Load market observation data separately from samples.db
    
    Args:
        timeframe_id: Timeframe ID
        
    Returns:
        ObservationManager: Manager with loaded market observations and portfolio calculator
    """
    try:
        market_data = load_data_from_db(timeframe_id=timeframe_id)
        
        if market_data.empty:
            logger.warning("No market observation data found")
            return ObservationManager()
        
        # Create portfolio calculator
        portfolio_calculator = PortfolioFeatureCalculator()
        
        logger.info(f"Loaded {len(market_data)} market observation bars for timeframe_id={timeframe_id}")
        return ObservationManager(market_data, portfolio_calculator)
        
    except Exception as e:
        logger.error(f"Error loading market observation data: {e}")
        return ObservationManager()


# Factory functions
def create_price_feed(stock_id=1, timeframe_id=5):
    """
    Create price-only data feed for Backtrader
    
    Args:
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        
    Returns:
        SimplePriceFeed: Ready for Backtrader or None if no data
    """
    price_data = load_price_data_from_db(stock_id, timeframe_id)
    if price_data.empty:
        logger.error("Cannot create price feed - no data available")
        return None
    return SimplePriceFeed(dataname=price_data)


def create_observation_manager(timeframe_id=5):
    """
    Create observation manager for strategy
    
    Args:
        timeframe_id: Timeframe ID
        
    Returns:
        ObservationManager: Ready for strategy use
    """
    return load_observation_data_from_db(timeframe_id)


def create_complete_backtest_data(stock_id=1, timeframe_id=5):
    """
    Create both price feed and observation manager
    
    Args:
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        
    Returns:
        tuple: (price_feed, observation_manager) or (None, None) if no data
    """
    price_feed = create_price_feed(stock_id, timeframe_id)
    obs_manager = create_observation_manager(timeframe_id)
    
    if price_feed is None:
        logger.error("Cannot create backtest data - no price data available")
        return None, None
    
    return price_feed, obs_manager


# Legacy compatibility (deprecated)
class EnvironmentDataFeed(SimplePriceFeed):
    """Legacy compatibility class - use SimplePriceFeed instead"""
    pass


class PandasEnvironmentFeed(SimplePriceFeed):
    """Legacy compatibility class - use SimplePriceFeed instead"""
    pass


# Example usage
if __name__ == "__main__":
    # Test the simplified approach
   
    print("Testing simplified data feed approach...")
      # Create price feed
    try:
        price_feed = create_price_feed(stock_id=1, timeframe_id=5)
        if price_feed is not None:
            print("✓ Price feed created successfully")
            print(price_feed.p.dataname.head())
        else:
            print("✗ Failed to create price feed - no data available")
    except Exception as e:
        print(f"✗ Error creating price feed: {e}")
        price_feed = None
          # Create observation manager
    obs_manager = create_observation_manager(timeframe_id=5)
    print("✓ Observation manager created successfully")
      # Test getting current observation
    if price_feed is not None and hasattr(price_feed, 'p') and hasattr(price_feed.p, 'dataname'):
        price_data = price_feed.p.dataname
        if not price_data.empty:
            test_datetime = price_data.index[0]
            
            # Test market features only
            market_features = obs_manager.get_market_observation(test_datetime)
            print(f"✓ Retrieved {len(market_features)} market features for {test_datetime}")
            
            # Test complete observation with portfolio features
            complete_features = obs_manager.get_complete_observation(test_datetime)
            print(f"✓ Retrieved {len(complete_features)} complete features (market + portfolio)")
            
            # Test observation array
            obs_array = obs_manager.get_observation_array(test_datetime)
            print(f"✓ Observation array shape: {obs_array.shape}")
            print(f"✓ Market features: {len(obs_manager.MARKET_FEATURE_NAMES)}")
            print(f"✓ Portfolio features: {len(obs_manager.PORTFOLIO_FEATURE_NAMES)}")
            print(f"✓ Total features: {len(obs_manager.ALL_FEATURE_NAMES)}")
        else:
            print("⚠ Price data is empty")
    else:
        print("⚠ Could not access price data for testing")
    
    print("Testing completed! ✓")

