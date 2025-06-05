#!/usr/bin/env python3
"""
Test Backtrader Strategy Integration with Portfolio Calculator

This test simulates actual backtrader trading to verify that the strategy
properly updates the PortfolioFeatureCalculator during trading operations.
"""

import sys
import os
from pathlib import Path
import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from Backtrader.data_feeds import create_complete_backtest_data
from Backtrader.rl_strategy import RLTradingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRLModel:
    """Mock RL model for testing purposes"""
    
    def __init__(self, actions_sequence=None):
        """Initialize with predefined actions sequence"""
        self.actions_sequence = actions_sequence if actions_sequence else [
            [1, 10, 5, 5],  # BUY action
            [0, 0, 0, 0],   # HOLD 
            [0, 0, 0, 0],   # HOLD
            [2, 8, 3, 4],   # SELL action
            [0, 0, 0, 0],   # HOLD
        ]
        self.call_count = 0
    
    def predict(self, observation, deterministic=True):
        """Return predefined action"""
        if self.call_count < len(self.actions_sequence):
            action = self.actions_sequence[self.call_count]
        else:
            action = [0, 0, 0, 0]  # HOLD as default
        
        self.call_count += 1
        return np.array(action), None


def test_strategy_portfolio_integration():
    """Test that strategy properly updates portfolio calculator"""
    print("\n" + "="*60)
    print("TESTING BACKTRADER STRATEGY PORTFOLIO INTEGRATION")
    print("="*60)
    
    try:
        # 1. Create data feeds
        print("1. Creating data feeds...")
        price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
        
        if price_feed is None or obs_manager is None:
            print("❌ Failed to create data feeds")
            return False
        
        print(f"✅ Price feed: {len(price_feed.p.dataname)} bars")
        print(f"✅ Observation manager: {len(obs_manager.market_observations)} observations")
        
        # 2. Check initial portfolio state
        print("\n2. Checking initial portfolio state...")
        initial_features = obs_manager.portfolio_calculator.get_portfolio_features()
        print(f"Initial portfolio features: {initial_features}")
        
        expected_initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if not np.allclose(initial_features, expected_initial, atol=0.001):
            print(f"❌ Initial features incorrect. Expected: {expected_initial}")
            return False
        print("✅ Initial portfolio state correct")
        
        # 3. Create mock model and strategy
        print("\n3. Creating strategy with mock model...")
        mock_model = MockRLModel()
        
        # Create cerebro engine
        cerebro = bt.Cerebro()
        
        # Add strategy with portfolio calculator integration
        cerebro.addstrategy(
            RLTradingStrategy,
            model=mock_model,
            observation_manager=obs_manager,
            initial_cash=10000,
            verbose=True
        )
        
        # Add data feed (limit to first 100 bars for quick test)
        limited_data = price_feed.p.dataname.head(100)
        limited_feed = bt.feeds.PandasData(dataname=limited_data)
        cerebro.adddata(limited_feed)
        
        # Set broker parameters
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.001)
        
        print("✅ Strategy and cerebro configured")
        
        # 4. Run backtest
        print("\n4. Running backtest...")
        initial_value = cerebro.broker.getvalue()
        print(f"Initial portfolio value: ${initial_value:.2f}")
        
        # Capture portfolio features before
        features_before = obs_manager.portfolio_calculator.get_portfolio_features()
        print(f"Portfolio features before backtest: {features_before}")
        
        # Run the backtest
        result = cerebro.run()
        strategy = result[0]
        
        final_value = cerebro.broker.getvalue()
        print(f"Final portfolio value: ${final_value:.2f}")
        
        # 5. Check portfolio features after trading
        print("\n5. Checking portfolio features after trading...")
        features_after = obs_manager.portfolio_calculator.get_portfolio_features()
        print(f"Portfolio features after backtest: {features_after}")
        
        # Check if features actually changed (indicating integration works)
        features_changed = not np.allclose(features_before, features_after, atol=0.001)
        
        if features_changed:
            print("✅ Portfolio features updated during trading!")
            
            # Analyze specific changes
            print("\nFeature changes:")
            feature_names = obs_manager.PORTFOLIO_FEATURE_NAMES
            for i, name in enumerate(feature_names):
                before_val = features_before[i]
                after_val = features_after[i]
                change = after_val - before_val
                print(f"  {name}: {before_val:.6f} → {after_val:.6f} (Δ{change:+.6f})")
        else:
            print("⚠️ Portfolio features did not change - may indicate no trades executed")
        
        # 6. Check strategy trade log
        print("\n6. Checking strategy trade log...")
        trade_log = strategy.get_trade_log()
        print(f"Strategy recorded {len(trade_log)} trades")
        
        if len(trade_log) > 0:
            print("✅ Trades were executed")
            for i, trade in enumerate(trade_log):
                print(f"  Trade {i+1}: {trade['action_type']} P&L: {trade['pnl_pct']:.2%} Reason: {trade['exit_reason']}")
        else:
            print("⚠️ No trades executed - may need longer backtest or different parameters")
        
        # 7. Check portfolio calculator trade count
        print("\n7. Checking portfolio calculator trade count...")
        portfolio_trade_count = obs_manager.portfolio_calculator.total_trades
        print(f"Portfolio calculator recorded {portfolio_trade_count} trades")
        
        if portfolio_trade_count > 0:
            print("✅ Portfolio calculator received trade updates")
            print(f"  Winning trades: {obs_manager.portfolio_calculator.winning_trades}")
            print(f"  Total P&L: {obs_manager.portfolio_calculator.total_pnl:.2f}")
            print(f"  Win rate: {obs_manager.portfolio_calculator.winning_trades / obs_manager.portfolio_calculator.total_trades:.2%}")
        else:
            print("⚠️ Portfolio calculator did not receive trade updates")
        
        # 8. Test observation generation during strategy
        print("\n8. Testing observation generation...")
        test_datetime = limited_data.index[50]  # Mid-point of test data
        
        # Get complete observation
        complete_obs = obs_manager.get_complete_observation(test_datetime)
        obs_array = obs_manager.get_observation_array(test_datetime)
        
        print(f"Complete observation keys: {len(complete_obs)}")
        print(f"Observation array shape: {obs_array.shape}")
        print(f"Current portfolio features in observation:")
        
        for name in obs_manager.PORTFOLIO_FEATURE_NAMES:
            if name in complete_obs:
                print(f"  {name}: {complete_obs[name]:.6f}")
        
        # 9. Final verification
        print("\n9. Final integration verification...")
        
        integration_success = True
        
        # Check 1: Portfolio calculator was updated
        if obs_manager.portfolio_calculator.current_balance != obs_manager.portfolio_calculator.initial_balance:
            print("✅ Portfolio balance was updated")
        else:
            print("⚠️ Portfolio balance not updated")
            integration_success = False
        
        # Check 2: Strategy executed model predictions
        if mock_model.call_count > 0:
            print(f"✅ Model was called {mock_model.call_count} times")
        else:
            print("❌ Model was never called")
            integration_success = False
        
        # Check 3: Observations contain both market and portfolio features
        expected_features = len(obs_manager.ALL_FEATURE_NAMES)
        if len(obs_array) == expected_features:
            print(f"✅ Observations contain all {expected_features} features")
        else:
            print(f"❌ Observation missing features: {len(obs_array)} vs {expected_features}")
            integration_success = False
        
        print("\n" + "="*60)
        if integration_success:
            print("🎉 INTEGRATION TEST SUCCESSFUL!")
            print("The strategy properly updates the portfolio calculator.")
        else:
            print("❌ INTEGRATION TEST FAILED!")
            print("Issues detected in strategy-portfolio integration.")
        print("="*60)
        
        return integration_success
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the integration test"""
    success = test_strategy_portfolio_integration()
    
    if success:
        print("\n🚀 READY FOR PRODUCTION!")
        print("The backtrader strategy properly integrates with PortfolioFeatureCalculator.")
    else:
        print("\n🔧 INTEGRATION NEEDS FIXING!")
        print("Review the integration between strategy and portfolio calculator.")


if __name__ == "__main__":
    main()
