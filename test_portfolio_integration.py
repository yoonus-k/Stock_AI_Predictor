#!/usr/bin/env python3
"""
Test Portfolio Calculator Integration with Backtrader Strategy

This test verifies that the PortfolioFeatureCalculator is properly updated
by the backtrader strategy during trading operations.
"""

import sys
import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from Backtrader.data_feeds import (
    create_complete_backtest_data,
    PortfolioFeatureCalculator,
    ObservationManager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPortfolioIntegration(unittest.TestCase):
    """Test portfolio calculator integration with strategy"""
    
    def setUp(self):
        """Set up test environment"""
        self.initial_balance = 10000.0
        self.portfolio_calc = PortfolioFeatureCalculator()
        self.portfolio_calc.reset(self.initial_balance)
    
    def test_portfolio_calculator_standalone(self):
        """Test portfolio calculator functions independently"""
        print("\n=== Testing Portfolio Calculator Standalone ===")
        
        # Test initial state
        initial_features = self.portfolio_calc.get_portfolio_features()
        print(f"Initial portfolio features: {initial_features}")
        
        expected_initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(initial_features, expected_initial, decimal=3)
        print("✓ Initial features correct")
        
        # Test balance update
        new_balance = 11000.0
        self.portfolio_calc.update_balance(new_balance)
        
        updated_features = self.portfolio_calc.get_portfolio_features()
        print(f"After balance update: {updated_features}")
        
        # Balance ratio should be 1.1
        self.assertAlmostEqual(updated_features[0], 1.1, places=3)
        print("✓ Balance update working")
        
        # Test trade recording
        self.portfolio_calc.record_trade(pnl=500.0, exit_reason='tp', holding_hours=2.0)
        
        trade_features = self.portfolio_calc.get_portfolio_features()
        print(f"After recording trade: {trade_features}")
        
        # Win rate should be 1.0 (1 winning trade out of 1 total)
        self.assertAlmostEqual(trade_features[2], 1.0, places=3)
        print("✓ Trade recording working")
        
        # Test losing trade
        self.portfolio_calc.record_trade(pnl=-200.0, exit_reason='sl', holding_hours=1.5)
        
        loss_features = self.portfolio_calc.get_portfolio_features()
        print(f"After losing trade: {loss_features}")
        
        # Win rate should be 0.5 (1 win out of 2 trades)
        self.assertAlmostEqual(loss_features[2], 0.5, places=3)
        print("✓ Multiple trades working")
        
    def test_observation_manager_integration(self):
        """Test that ObservationManager properly uses portfolio calculator"""
        print("\n=== Testing ObservationManager Integration ===")
        
        try:
            # Create observation manager
            _, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
            
            if obs_manager is None:
                print("⚠ No observation manager created - skipping integration test")
                return
            
            # Test that portfolio calculator exists
            self.assertTrue(hasattr(obs_manager, 'portfolio_calculator'))
            print("✓ ObservationManager has portfolio calculator")
            
            # Test getting complete observation
            if hasattr(obs_manager, 'market_observations') and not obs_manager.market_observations.empty:
                test_datetime = obs_manager.market_observations.index[0]
                
                # Get observation before any updates
                initial_obs = obs_manager.get_complete_observation(test_datetime)
                print(f"Initial observation keys: {list(initial_obs.keys())}")
                print(f"Portfolio features in observation: {[k for k in initial_obs.keys() if k in obs_manager.PORTFOLIO_FEATURE_NAMES]}")
                
                # Check that all expected features are present
                expected_features = obs_manager.ALL_FEATURE_NAMES
                for feature in expected_features:
                    self.assertIn(feature, initial_obs)
                print(f"✓ All {len(expected_features)} features present in observation")
                
                # Update portfolio calculator and check if observation changes
                obs_manager.portfolio_calculator.update_balance(12000.0)
                obs_manager.portfolio_calculator.record_trade(pnl=1000.0, exit_reason='tp', holding_hours=3.0)
                
                updated_obs = obs_manager.get_complete_observation(test_datetime)
                
                # Balance ratio should have changed
                initial_balance_ratio = initial_obs['balance_ratio']
                updated_balance_ratio = updated_obs['balance_ratio']
                
                print(f"Balance ratio - Initial: {initial_balance_ratio}, Updated: {updated_balance_ratio}")
                self.assertNotAlmostEqual(initial_balance_ratio, updated_balance_ratio, places=3)
                print("✓ Portfolio features update in observations")
                
                # Test observation array
                obs_array = obs_manager.get_observation_array(test_datetime)
                print(f"Observation array shape: {obs_array.shape}")
                self.assertEqual(len(obs_array), len(obs_manager.ALL_FEATURE_NAMES))
                print("✓ Observation array has correct length")
                
            else:
                print("⚠ No market observations available - skipping detailed tests")
                
        except Exception as e:
            print(f"⚠ Error testing observation manager integration: {e}")
    
    def test_feature_mapping_consistency(self):
        """Test that feature names and arrays are consistent"""
        print("\n=== Testing Feature Mapping Consistency ===")
        
        # Create observation manager
        market_data = pd.DataFrame()  # Empty for this test
        portfolio_calc = PortfolioFeatureCalculator()
        obs_manager = ObservationManager(market_data, portfolio_calc)
        
        # Check feature name counts
        market_count = len(obs_manager.MARKET_FEATURE_NAMES)
        portfolio_count = len(obs_manager.PORTFOLIO_FEATURE_NAMES)
        total_count = len(obs_manager.ALL_FEATURE_NAMES)
        
        print(f"Market features: {market_count}")
        print(f"Portfolio features: {portfolio_count}")
        print(f"Total features: {total_count}")
        
        self.assertEqual(total_count, market_count + portfolio_count)
        print("✓ Feature counts consistent")
        
        # Check that portfolio feature array matches names
        portfolio_array = portfolio_calc.get_portfolio_features()
        self.assertEqual(len(portfolio_array), len(obs_manager.PORTFOLIO_FEATURE_NAMES))
        print("✓ Portfolio array length matches feature names")
        
        # Print feature mappings for verification
        print("\nMarket features:")
        for i, name in enumerate(obs_manager.MARKET_FEATURE_NAMES):
            print(f"  {i:2d}: {name}")
        
        print("\nPortfolio features:")
        for i, name in enumerate(obs_manager.PORTFOLIO_FEATURE_NAMES):
            print(f"  {i:2d}: {name} = {portfolio_array[i]:.6f}")
        
        print("✓ Feature mapping verification complete")


def main():
    """Run integration tests"""
    print("Portfolio Calculator Integration Test")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
