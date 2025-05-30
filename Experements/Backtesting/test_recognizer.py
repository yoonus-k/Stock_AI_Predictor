import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Experements.Backtesting.db_extensions import Database
from Pattern.Utils.multi_config_recognizer import (
    ConfigBasedRecognizer, 
    pattern_matcher,
    extract_pips
)
from Experements.Backtesting.backtest_config import RecognitionTechnique

def test_recognizer_creation():
    """Test creating recognizers for different configurations"""
    print("Testing recognizer creation...")
    
    # Connect to database
    db = Database("Data/Storage/data.db")
    print(f"Connected to database: {db.db_path}")
    
    # Create a config-based recognizer
    multi_recognizer = ConfigBasedRecognizer(db, default_technique='random_forest')
    
    # Test with different stocks and timeframes
    test_configs = [
        {"stock_id": 1, "timeframe_id": 6, "config_id": 49, "n_pips": 5},
        {"stock_id": 1, "timeframe_id": 6, "config_id": 48, "n_pips": 6}
    ]
    
    for config in test_configs:
        try:
            print(f"\nTesting config: {config}")
            
            # Explicitly convert to Python primitives
            stock_id = int(config["stock_id"])
            timeframe_id = int(config["timeframe_id"])
            config_id = int(config["config_id"])
            n_pips = int(config["n_pips"])
            
            recognizer, feature_length, clusters_df = multi_recognizer.get_or_create_recognizer(
                stock_id, 
                timeframe_id,
                config_id,
                n_pips
            )
            
            print(f"✓ Successfully created recognizer")
            print(f"  - Feature length: {feature_length}")
            print(f"  - Clusters: {len(clusters_df)}")
            print(f"  - Recognizer type: {recognizer.name}")
            
            # Test caching by getting the same recognizer again
            cached_recognizer, _, _ = multi_recognizer.get_or_create_recognizer(
                stock_id, 
                timeframe_id,
                config_id,
                n_pips
            )
            
            print(f"✓ Successfully retrieved cached recognizer")
            
            # Verify it's the same object
            assert recognizer is cached_recognizer, "Cached recognizer is not the same object"
            
        except Exception as e:
            print(f"✗ Error creating recognizer: {e}")

def test_pattern_matching(config_id=None):
    """Test pattern matching with real data"""
    print("\nTesting pattern matching...")
    
    # Connect to database
    db = Database("Data/Storage/data.db")
    print(f"Connected to database: {db.db_path}")
    
    # For direct testing without config retrieval
    if config_id is None:
        stock_id = 1
        timeframe_id = 6
        n_pips = 5
        window_size = 24
    else:
        # Get the config from the database
        config = db.get_configs(config_id=config_id)
        if config is None:
            print(f"✗ No config found for config_id={config_id}")
            return
            
        # Ensure we extract primitive values from the config
        # Use python primitive types to prevent SQLite errors
        if isinstance(config, pd.DataFrame):
            if not config.empty:
                config = config.iloc[0].to_dict()
            else:
                print("✗ Empty config DataFrame")
                return
                
        # Extract and convert to Python primitives
        try:
            stock_id = int(config.get('stock_id', 1))
            timeframe_id = int(config.get('timeframe_id', 6))
            n_pips = int(config.get('n_pips', 5))
            window_size = int(config.get('lookback', 24))
        except (ValueError, TypeError) as e:
            print(f"✗ Error converting config values: {e}")
            print(f"  Config: {config}")
            # Fallback to defaults
            stock_id = 1
            timeframe_id = 6
            n_pips = 5
            window_size = 24
    
    # Print parameters for debugging
    print(f"Parameters: stock_id={stock_id} (type: {type(stock_id)}), "
          f"timeframe_id={timeframe_id} (type: {type(timeframe_id)}), "
          f"config_id={config_id} (type: {type(config_id)}), "
          f"n_pips={n_pips} (type: {type(n_pips)})")
    
    # Create a config-based recognizer
    multi_recognizer = ConfigBasedRecognizer(db, default_technique='random_forest')
    
    try:
        recognizer, feature_length, clusters_df = multi_recognizer.get_or_create_recognizer(
            stock_id, timeframe_id, config_id, n_pips
        )
        
        print(f"Successfully retrieved recognizer with feature_length={feature_length}")
        print(f"Retrieved {len(clusters_df)} clusters")
    except Exception as e:
        print(f"✗ Error getting recognizer: {e}")
        return
    
    # Get some price data to test with
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Format dates for SQL query
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Get price data directly using raw SQL to avoid pandas Series issues
    try:
        query = """
            SELECT timestamp, close_price
            FROM stock_data 
            WHERE stock_id = ? AND timeframe_id = ?
                AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        cursor = db.connection.cursor()
        cursor.execute(query, (stock_id, timeframe_id, start_str, end_str))
        price_data = cursor.fetchall()
        
        if not price_data:
            print("✗ No price data found for testing")
            return
            
        # Convert to simple numpy arrays
        timestamps = [p[0] for p in price_data]
        prices = np.array([float(p[1]) for p in price_data])
        
        print(f"Retrieved {len(prices)} price records for testing")
        
        if len(prices) < window_size + 5:
            print("✗ Not enough price data for testing")
            return
    except Exception as e:
        print(f"✗ Error fetching price data: {e}")
        return
    
    # Test pattern matching with a window of price data
    matches_found = 0
    
    for i in range(window_size, len(prices) - 5):
        window = prices[i-window_size:i+1]
        
        pattern_info = pattern_matcher(
            recognizer,
            clusters_df,
            window,
            n_pips=n_pips,
            dist_type=2,
            mse_threshold=0.01  # More permissive threshold for testing
        )
        
        if pattern_info:
            matches_found += 1
            # Print detailed info for the first 3 matches
            if matches_found <= 3:
                print(f"\nMatch {matches_found}:")
                print(f"  - Cluster ID: {pattern_info['cluster_id']}")
                print(f"  - Label: {pattern_info['label']}")
                print(f"  - Confidence: {pattern_info['confidence']:.4f}")
                if pattern_info.get('mse') is not None:
                    print(f"  - MSE: {pattern_info['mse']:.4f}")
                print(f"  - Outcome: {pattern_info.get('outcome_str', 'unknown')}")
    
    print(f"\nTotal matches found: {matches_found}/{len(prices) - window_size - 5}")
    if len(prices) > window_size + 5:
        print(f"Match rate: {matches_found/(len(prices) - window_size - 5):.2%}")

def test_different_recognition_techniques():
    """Test different recognition techniques"""
    print("\nTesting different recognition techniques...")
    
    # Connect to database
    db = Database("Data/Storage/data.db")
    
    techniques = ['svm', 'random_forest']
    
    for technique in techniques:
        print(f"\nTesting {technique} recognizer...")
        
        # Create a config-based recognizer with the specific technique
        multi_recognizer = ConfigBasedRecognizer(db, default_technique=technique)
        
        try:
            # Get a recognizer with explicit Python primitive types
            stock_id = 1
            timeframe_id = 6
            config_id = None
            n_pips = 5
            
            recognizer, feature_length, clusters_df = multi_recognizer.get_or_create_recognizer(
                stock_id=stock_id, 
                timeframe_id=timeframe_id,
                config_id=config_id,
                n_pips=n_pips
            )
            
            print(f"✓ Successfully created {technique} recognizer")
            print(f"  - Feature length: {feature_length}")
            print(f"  - Clusters: {len(clusters_df)}")
        except Exception as e:
            print(f"✗ Error creating {technique} recognizer: {e}")

def test_feature_dimension_handling():
    """Test that the recognizer correctly handles different feature dimensions"""
    print("\nTesting feature dimension handling...")
    
    # Connect to database
    db = Database("Data/Storage/data.db")
    
    # Create a config-based recognizer
    multi_recognizer = ConfigBasedRecognizer(db, default_technique='svm')
    
    # Get a recognizer with explicit Python primitive types
    stock_id = 1
    timeframe_id = 6
    config_id = None
    n_pips = 5
    
    try:
        recognizer, feature_length, clusters_df = multi_recognizer.get_or_create_recognizer(
            stock_id=stock_id, 
            timeframe_id=timeframe_id, 
            config_id=config_id, 
            n_pips=n_pips
        )
        
        print(f"Recognizer created with feature_length={feature_length}")
    except Exception as e:
        print(f"✗ Error creating recognizer: {e}")
        return
    
    # Get some price data using direct SQL to avoid pandas Series issues
    try:
        cursor = db.connection.cursor()
        cursor.execute("""
            SELECT close_price 
            FROM stock_data 
            WHERE stock_id = ? AND timeframe_id = ?
            ORDER BY timestamp
            LIMIT 50
        """, (stock_id, timeframe_id))
        
        price_data = cursor.fetchall()
        
        if not price_data:
            print("✗ No price data found for testing")
            return
            
        price_series = np.array([float(p[0]) for p in price_data])
    except Exception as e:
        print(f"✗ Error fetching price data: {e}")
        return
    
    # Test with different lengths of price data
    test_windows = [
        price_series[:20],   # Shorter than typical
        price_series[:30],   # Typical length
        price_series[:40]    # Longer than typical
    ]
    
    for i, window in enumerate(test_windows):
        print(f"\nTest window {i+1}: Length = {len(window)}")
        
        try:
            # Extract PIPs with explicit expected_length
            x, y = extract_pips(window, n_pips=n_pips, dist_type=3, expected_length=feature_length)
            
            if y is None:
                print("  ✗ Could not extract PIPs")
                continue
                
            print(f"  ✓ Extracted PIPs with length {len(y)}")
            
            # Verify the feature length matches what's expected
            if len(y) != feature_length:
                print(f"  ⚠ Feature length mismatch: got {len(y)}, expected {feature_length}")
                # Adjust the feature vector to match expected length
                if len(y) < feature_length:
                    # Pad with zeros
                    y = np.pad(y, (0, feature_length - len(y)), 'constant')
                else:
                    # Truncate
                    y = y[:feature_length]
                print(f"  ✓ Adjusted to length {len(y)}")
            
            # Try to get a prediction
            cluster_id, confidence = recognizer.predict(y)
            print(f"  ✓ Prediction successful: cluster_id={cluster_id}, confidence={confidence:.4f}")
        except Exception as e:
            print(f"  ✗ Error in test: {e}")

def main():
    """Main test function"""
    print("=" * 80)
    print(" TESTING MULTI-CONFIG RECOGNIZER ".center(80, "="))
    print("=" * 80)
    
    # Run all tests
    #test_recognizer_creation()
    test_pattern_matching(60)
    #test_different_recognition_techniques()
    #test_feature_dimension_handling()
    
    print("\n" + "=" * 80)
    print(" TEST COMPLETE ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    main()
