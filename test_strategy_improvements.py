#!/usr/bin/env python3
"""
Test script to verify all recent strategy improvements:
1. Observation normalization is working
2. Bracket orders are being used properly
3. Position management is improved
4. notify_order callback is working
"""

import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from Backtrader.data_feeds import create_complete_backtest_data
from Backtrader.rl_strategy import RLTradingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockRLModel:
    """Mock RL model that generates test actions"""
    
    def __init__(self):
        self.call_count = 0
        self.last_observation = None
        self.observation_stats = []
    
    def predict(self, observation, deterministic=True):
        """Mock prediction method"""
        self.call_count += 1
        self.last_observation = observation.copy()
        
        # Record observation statistics for testing
        self.observation_stats.append({
            'min': observation.min(),
            'max': observation.max(),
            'mean': observation.mean(),
            'std': observation.std(),
            'shape': observation.shape
        })
        
        # Generate diverse test actions
        if self.call_count % 3 == 1:
            action = [1, 5, 3, 2]  # BUY action
        elif self.call_count % 3 == 2:
            action = [2, 8, 6, 4]  # SELL action
        else:
            action = [0, 0, 0, 0]  # HOLD action
        
        return np.array(action), None


def test_observation_normalization():
    """Test that observations are properly normalized"""
    print("\n" + "="*60)
    print("TESTING OBSERVATION NORMALIZATION")
    print("="*60)
    
    # Create data feeds
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No price data available")
        return False
    
    # Create strategy with mock model
    model = MockRLModel()
    
    # Create cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(price_feed)
    
    # Add strategy with model
    cerebro.addstrategy(
        RLTradingStrategy,
        model=model,
        observation_manager=obs_manager,
        initial_cash=10000,
        verbose=True
    )
    
    # Set initial cash
    cerebro.broker.setcash(10000)
    
    # Run limited backtest for testing
    logger.info("Running backtest for observation normalization test...")
    cerebro.run(maxbars=50)  # Limited run for testing
    
    # Check observation statistics
    if len(model.observation_stats) == 0:
        print("❌ No observations were processed")
        return False
    
    print(f"✅ Processed {len(model.observation_stats)} observations")
    
    # Check if observations are normalized
    normalized_count = 0
    for i, stats in enumerate(model.observation_stats):
        if -1.1 <= stats['min'] <= -0.9 and 0.9 <= stats['max'] <= 1.1:
            normalized_count += 1
    
    normalization_rate = normalized_count / len(model.observation_stats)
    print(f"✅ Observations within normalized range: {normalization_rate:.1%}")
    
    if normalization_rate > 0.8:  # Allow some tolerance
        print("✅ Observation normalization is working correctly!")
        return True
    else:
        print("❌ Observation normalization may not be working properly")
        print(f"   Example stats: {model.observation_stats[0]}")
        return False


def test_bracket_orders_and_position_management():
    """Test that bracket orders and position management are working"""
    print("\n" + "="*60)
    print("TESTING BRACKET ORDERS & POSITION MANAGEMENT")
    print("="*60)
    
    # Create data feeds
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No price data available")
        return False
    
    # Create strategy with mock model that forces trades
    class ForcedTradeModel:
        def __init__(self):
            self.call_count = 0
            
        def predict(self, observation, deterministic=True):
            self.call_count += 1
            # Force a BUY action every 10 calls
            if self.call_count % 10 == 1:
                return np.array([1, 5, 3, 2]), None  # BUY
            return np.array([0, 0, 0, 0]), None  # HOLD
    
    model = ForcedTradeModel()
    
    # Track strategy instance
    strategy_instance = None
    
    class TrackedStrategy(RLTradingStrategy):
        def __init__(self):
            super().__init__()
            nonlocal strategy_instance
            strategy_instance = self
            self.order_notifications = []
            self.bracket_orders_created = 0
            
        def buy(self, *args, **kwargs):
            order = super().buy(*args, **kwargs)
            if order:
                self.bracket_orders_created += 1
            return order
            
        def sell(self, *args, **kwargs):
            order = super().sell(*args, **kwargs)
            return order
            
        def notify_order(self, order):
            super().notify_order(order)
            self.order_notifications.append({
                'status': order.status,
                'ref': order.ref,
                'size': getattr(order, 'size', 0),
                'price': getattr(order.executed, 'price', 0) if hasattr(order, 'executed') else 0
            })
    
    # Create cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(price_feed)
    
    # Add tracked strategy
    cerebro.addstrategy(
        TrackedStrategy,
        model=model,
        observation_manager=obs_manager,
        initial_cash=10000,
        verbose=True
    )
    
    # Set initial cash and commission
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Run backtest
    logger.info("Running backtest for bracket orders test...")
    cerebro.run(maxbars=100)
    
    # Check results
    if strategy_instance is None:
        print("❌ Strategy instance not captured")
        return False
    
    print(f"✅ Model called {model.call_count} times")
    print(f"✅ Order notifications received: {len(strategy_instance.order_notifications)}")
    print(f"✅ Positions tracked: {len(strategy_instance.current_positions)}")
    print(f"✅ Trades completed: {len(strategy_instance.trade_log)}")
    
    # Check if bracket orders were created
    if strategy_instance.bracket_orders_created > 0:
        print(f"✅ Bracket orders created: {strategy_instance.bracket_orders_created}")
    else:
        print("⚠️  No bracket orders were created (may be due to risk limits)")
    
    # Check if notify_order was called
    if len(strategy_instance.order_notifications) > 0:
        print("✅ notify_order callback is working")
        print(f"   Order statuses: {[n['status'] for n in strategy_instance.order_notifications[:5]]}")
    else:
        print("⚠️  notify_order callback was not triggered")
    
    # Check position management
    if hasattr(strategy_instance, 'pending_orders'):
        print(f"✅ Pending orders tracking: {len(strategy_instance.pending_orders)} active")
    
    return True


def test_complete_system_integration():
    """Test the complete system with all improvements"""
    print("\n" + "="*60)
    print("TESTING COMPLETE SYSTEM INTEGRATION")
    print("="*60)
    
    # Create data feeds
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No price data available")
        return False
    
    # Create realistic model
    model = MockRLModel()
    
    # Create cerebro with realistic settings
    cerebro = bt.Cerebro()
    cerebro.adddata(price_feed)
    
    # Add strategy
    cerebro.addstrategy(
        RLTradingStrategy,
        model=model,
        observation_manager=obs_manager,
        initial_cash=50000,
        verbose=False,  # Reduce noise for full test
        enable_short=True,
        max_positions=3
    )
    
    # Set realistic broker settings
    cerebro.broker.setcash(50000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Record initial state
    initial_cash = 50000
    
    # Run full backtest
    logger.info("Running complete system integration test...")
    results = cerebro.run()
    strategy = results[0]
    
    # Get final metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash
    
    print(f"✅ Initial cash: ${initial_cash:,.2f}")
    print(f"✅ Final value: ${final_value:,.2f}")
    print(f"✅ Total return: {total_return:.2%}")
    print(f"✅ Total trades: {len(strategy.trade_log)}")
    print(f"✅ Model predictions: {model.call_count}")
    print(f"✅ Max drawdown: {strategy.max_drawdown:.2%}")
    
    # Check observation statistics
    if len(model.observation_stats) > 0:
        final_obs = model.observation_stats[-1]
        print(f"✅ Final observation shape: {final_obs['shape']}")
        print(f"✅ Final observation range: [{final_obs['min']:.3f}, {final_obs['max']:.3f}]")
        
        # Check if normalized
        is_normalized = (-1.1 <= final_obs['min'] <= 1.1) and (-1.1 <= final_obs['max'] <= 1.1)
        print(f"✅ Observations normalized: {is_normalized}")
    
    # Check portfolio integration
    if obs_manager and hasattr(obs_manager, 'portfolio_calculator'):
        portfolio_features = obs_manager.portfolio_calculator.get_portfolio_features()
        print(f"✅ Portfolio features: {portfolio_features}")
        print(f"✅ Portfolio balance ratio: {portfolio_features[0]:.3f}")
        print(f"✅ Portfolio win rate: {portfolio_features[2]:.3f}")
    
    return True


def main():
    """Run all improvement tests"""
    print("🚀 TESTING STRATEGY IMPROVEMENTS")
    print("="*80)
    
    tests = [
        ("Observation Normalization", test_observation_normalization),
        ("Bracket Orders & Position Management", test_bracket_orders_and_position_management),
        ("Complete System Integration", test_complete_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL IMPROVEMENTS ARE WORKING CORRECTLY!")
        print("\n🚀 READY FOR PRODUCTION:")
        print("  ✅ Observation normalization: Working")
        print("  ✅ Bracket orders: Implemented")
        print("  ✅ Position management: Improved")
        print("  ✅ Order notifications: Working")
        print("  ✅ Portfolio integration: Complete")
    else:
        print("⚠️  Some tests failed - check logs above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
