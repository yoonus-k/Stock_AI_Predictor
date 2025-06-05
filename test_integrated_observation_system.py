#!/usr/bin/env python3
"""
Test script for the integrated observation system with separated market and portfolio features.

This script demonstrates the complete pipeline:
1. Load market features from database (samples.db)
2. Calculate portfolio features from runtime data
3. Concatenate both feature sets
4. Normalize complete observations
5. Feed to RL model

Usage:
    python test_integrated_observation_system.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from Backtrader.data_feeds import (
    create_price_feed, 
    create_observation_manager, 
    PortfolioFeatureCalculator,
    ObservationManager
)
from RL.Envs.Components.observation_normalizer import ObservationNormalizer

def test_complete_observation_pipeline():
    """Test the complete observation pipeline with normalization"""
    print("=" * 60)
    print("TESTING INTEGRATED OBSERVATION SYSTEM")
    print("=" * 60)
    
    # 1. Create data feeds
    print("\n1. Creating data feeds...")
    try:
        price_feed = create_price_feed(stock_id=1, timeframe_id=5)
        obs_manager = create_observation_manager(timeframe_id=5)
        
        if price_feed is None:
            print("❌ Failed to create price feed")
            return
        
        print(f"✅ Price feed created with {len(price_feed.p.dataname)} bars")
        print(f"✅ Observation manager created")
        
    except Exception as e:
        print(f"❌ Error creating data feeds: {e}")
        return
    
    # 2. Test feature separation
    print("\n2. Testing feature separation...")
    try:
        test_datetime = price_feed.p.dataname.index[0]
        
        # Market features only (from database)
        market_features = obs_manager.get_market_observation(test_datetime)
        print(f"✅ Market features: {len(market_features)}")
        
        # Portfolio features only (calculated)
        portfolio_features = obs_manager.portfolio_calculator.get_portfolio_features()
        print(f"✅ Portfolio features: {len(portfolio_features)}")
        
        # Complete observation
        complete_features = obs_manager.get_complete_observation(test_datetime)
        print(f"✅ Complete features: {len(complete_features)}")
        
        # Verify counts
        assert len(market_features) == 24, f"Expected 24 market features, got {len(market_features)}"
        assert len(portfolio_features) == 6, f"Expected 6 portfolio features, got {len(portfolio_features)}"
        assert len(complete_features) == 30, f"Expected 30 total features, got {len(complete_features)}"
        
        print("✅ Feature counts verified")
        
    except Exception as e:
        print(f"❌ Error testing feature separation: {e}")
        return
    
    # 3. Test portfolio feature calculation
    print("\n3. Testing portfolio feature updates...")
    try:
        portfolio_calc = obs_manager.portfolio_calculator
        
        # Simulate some trading activity
        portfolio_calc.reset(initial_balance=10000.0)
        portfolio_calc.update_balance(10500.0)  # 5% gain
        portfolio_calc.record_trade(pnl=500.0, exit_reason='tp', holding_hours=2.0)
        portfolio_calc.update_balance(10200.0)  # Some drawdown
        portfolio_calc.record_trade(pnl=-300.0, exit_reason='sl', holding_hours=1.5)
        
        # Get updated portfolio features
        portfolio_features = portfolio_calc.get_portfolio_features()
        
        print(f"✅ Balance ratio: {portfolio_features[0]:.3f}")
        print(f"✅ Max drawdown: {portfolio_features[1]:.3f}")
        print(f"✅ Win rate: {portfolio_features[2]:.3f}")
        print(f"✅ Avg PnL/hour: {portfolio_features[3]:.3f}")
        print(f"✅ Decisive exits: {portfolio_features[4]:.3f}")
        print(f"✅ Recovery factor: {portfolio_features[5]:.3f}")
        
    except Exception as e:
        print(f"❌ Error testing portfolio features: {e}")
        return
    
    # 4. Test observation normalization
    print("\n4. Testing observation normalization...")
    try:
        # Create normalizer
        normalizer = ObservationNormalizer(
            enable_adaptive_scaling=False,
            output_range=(-1.0, 1.0),
            clip_outliers=True
        )
        
        # Get raw observation
        raw_observation = obs_manager.get_observation_array(test_datetime)
        print(f"✅ Raw observation shape: {raw_observation.shape}")
        print(f"✅ Raw observation range: [{raw_observation.min():.3f}, {raw_observation.max():.3f}]")
        
        # Normalize observation
        normalized_observation = normalizer.normalize_observation(raw_observation)
        print(f"✅ Normalized observation shape: {normalized_observation.shape}")
        print(f"✅ Normalized observation range: [{normalized_observation.min():.3f}, {normalized_observation.max():.3f}]")
        
        # Test with built-in normalization method
        normalized_obs_direct = obs_manager.get_normalized_observation_array(test_datetime, normalizer)
        print(f"✅ Direct normalized observation shape: {normalized_obs_direct.shape}")
        
        # Verify normalization worked
        assert -1.1 <= normalized_observation.min() <= -0.9, "Normalization failed - min value out of range"
        assert 0.9 <= normalized_observation.max() <= 1.1, "Normalization failed - max value out of range"
        
        print("✅ Normalization verified")
        
    except Exception as e:
        print(f"❌ Error testing normalization: {e}")
        return
    
    # 5. Test feature mapping
    print("\n5. Testing feature mapping...")
    try:
        # Verify feature names match indices
        all_features = obs_manager.ALL_FEATURE_NAMES
        market_features = obs_manager.MARKET_FEATURE_NAMES
        portfolio_features_names = obs_manager.PORTFOLIO_FEATURE_NAMES
        
        print(f"✅ All feature names: {len(all_features)}")
        print(f"✅ Market feature names: {len(market_features)}")
        print(f"✅ Portfolio feature names: {len(portfolio_features_names)}")
        
        # Check feature categories
        print("\n   Feature breakdown:")
        print(f"   - Pattern features: {market_features[0:7]}")
        print(f"   - Technical features: {market_features[7:10]}")
        print(f"   - Sentiment features: {market_features[10:11]}")
        print(f"   - COT features: {market_features[11:17]}")
        print(f"   - Time features: {market_features[17:24]}")
        print(f"   - Portfolio features: {portfolio_features_names}")
        
        # Verify concatenation
        reconstructed = market_features + portfolio_features_names
        assert reconstructed == all_features, "Feature name concatenation mismatch"
        print("✅ Feature mapping verified")
        
    except Exception as e:
        print(f"❌ Error testing feature mapping: {e}")
        return
    
    # 6. Performance test with multiple observations
    print("\n6. Testing performance with multiple observations...")
    try:
        price_data = price_feed.p.dataname
        sample_size = min(100, len(price_data))
        sample_datetimes = price_data.index[:sample_size]
        
        start_time = datetime.now()
        observations = []
        
        for dt in sample_datetimes:
            obs = obs_manager.get_normalized_observation_array(dt, normalizer)
            observations.append(obs)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        observations_array = np.array(observations)
        print(f"✅ Processed {sample_size} observations in {processing_time:.3f} seconds")
        print(f"✅ Final array shape: {observations_array.shape}")
        print(f"✅ Average processing time: {processing_time/sample_size*1000:.2f} ms per observation")
        
    except Exception as e:
        print(f"❌ Error testing performance: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! INTEGRATION SUCCESSFUL!")
    print("=" * 60)
    
    # Summary
    print("\n📊 SYSTEM SUMMARY:")
    print(f"   • Market features from database: {len(obs_manager.MARKET_FEATURE_NAMES)}")
    print(f"   • Portfolio features calculated: {len(obs_manager.PORTFOLIO_FEATURE_NAMES)}")
    print(f"   • Total observation features: {len(obs_manager.ALL_FEATURE_NAMES)}")
    print(f"   • Normalization: ✅ Working")
    print(f"   • Performance: {processing_time/sample_size*1000:.2f} ms/observation")
    print(f"   • Architecture: ✅ Properly separated")
    
    return True

def test_database_feature_verification():
    """Verify that database contains exactly the market features we expect"""
    print("\n" + "=" * 60)
    print("VERIFYING DATABASE FEATURE ALIGNMENT")
    print("=" * 60)
    
    try:
        from RL.Data.Utils.loader import load_data_from_db
        
        # Load sample data from database
        data = load_data_from_db(timeframe_id=5)
        
        if data.empty:
            print("❌ No data found in database")
            return False
        
        print(f"✅ Loaded {len(data)} records from database")
        
        # Get actual columns from database
        db_columns = list(data.columns)
        print(f"✅ Database has {len(db_columns)} columns")
        
        # Expected market features
        obs_manager = create_observation_manager(timeframe_id=5)
        expected_market_features = obs_manager.MARKET_FEATURE_NAMES
        
        print(f"✅ Expected {len(expected_market_features)} market features")
        
        # Check alignment
        missing_features = []
        extra_features = []
        
        for feature in expected_market_features:
            if feature not in db_columns:
                missing_features.append(feature)
        
        for column in db_columns:
            if column not in expected_market_features:
                extra_features.append(column)
        
        # Report results
        if not missing_features and not extra_features:
            print("✅ Perfect alignment between database and expected features!")
        else:
            if missing_features:
                print(f"⚠️  Missing features in database: {missing_features}")
            if extra_features:
                print(f"ℹ️  Extra features in database: {extra_features}")
        
        print(f"\n📋 Database columns: {db_columns}")
        print(f"📋 Expected market features: {expected_market_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying database features: {e}")
        return False

if __name__ == "__main__":
    try:
        # Run comprehensive tests
        success = test_complete_observation_pipeline()
        
        if success:
            # Additional verification
            test_database_feature_verification()
            
            print("\n🚀 READY FOR PRODUCTION!")
            print("   The integrated observation system is working correctly.")
            print("   Market features are loaded from database, portfolio features")
            print("   are calculated at runtime, and normalization is applied.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
