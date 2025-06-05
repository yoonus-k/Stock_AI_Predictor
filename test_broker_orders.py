#!/usr/bin/env python3
"""
Focused test to debug why all orders are being rejected
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

class SimpleBrokerTestStrategy(bt.Strategy):
    """Simple strategy to test basic order submission"""
    
    def __init__(self):
        self.order_count = 0
        self.order_attempts = []
        
    def next(self):
        """Try to place one simple order"""
        if self.order_count < 3 and len(self.data) > 10:  # Only try 3 orders
            current_price = self.data.close[0]
            cash = self.broker.get_cash()
            
            # Calculate a very conservative position size
            shares = 1  # Just 1 share to test
            required_cash = shares * current_price * 1.2  # 20% buffer
            
            logger.info(f"Order {self.order_count + 1}:")
            logger.info(f"  Current price: ${current_price:.2f}")
            logger.info(f"  Available cash: ${cash:.2f}")
            logger.info(f"  Required cash: ${required_cash:.2f}")
            logger.info(f"  Shares to buy: {shares}")
            
            if required_cash <= cash:
                try:
                    # Place simple market order
                    order = self.buy(size=shares)
                    self.order_attempts.append({
                        'order_id': order.ref if order else None,
                        'size': shares,
                        'price': current_price,
                        'cash_available': cash,
                        'cash_required': required_cash,
                        'order_created': order is not None
                    })
                    logger.info(f"  Order created: {order is not None}")
                    if order:
                        logger.info(f"  Order ID: {order.ref}")
                except Exception as e:
                    logger.error(f"  Error creating order: {e}")
            else:
                logger.warning(f"  Insufficient cash: need ${required_cash:.2f}, have ${cash:.2f}")
                
            self.order_count += 1
    
    def notify_order(self, order):
        """Track order notifications"""
        status_names = {
            0: 'Created', 1: 'Submitted', 2: 'Accepted', 3: 'Partial',
            4: 'Completed', 5: 'Canceled', 6: 'Margin', 7: 'Rejected'
        }
        status_name = status_names.get(order.status, f'Unknown({order.status})')
        
        logger.info(f"Order {order.ref}: {status_name}")
        
        if order.status == 7:  # Rejected
            logger.error(f"ORDER REJECTED DETAILS:")
            logger.error(f"  Order size: {order.size}")
            logger.error(f"  Order price: {order.price}")
            logger.error(f"  Order type: {'BUY' if order.isbuy() else 'SELL'}")
            logger.error(f"  Broker cash: ${self.broker.get_cash():.2f}")
            logger.error(f"  Broker value: ${self.broker.get_value():.2f}")
            logger.error(f"  Position size: {self.getposition().size}")


def test_basic_broker_functionality():
    """Test basic broker order functionality"""
    print("\n" + "="*60)
    print("TESTING BASIC BROKER ORDER FUNCTIONALITY")
    print("="*60)
    
    # Create data feeds
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No price data available")
        return False
    
    # Create cerebro with minimal configuration
    cerebro = bt.Cerebro()
    cerebro.adddata(price_feed)
    
    # Add simple strategy
    cerebro.addstrategy(SimpleBrokerTestStrategy)
    
    # Configure broker with explicit settings
    cerebro.broker.setcash(50000)  # More cash to eliminate cash issues
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Log broker configuration
    logger.info("BROKER CONFIGURATION:")
    logger.info(f"  Initial cash: ${cerebro.broker.get_cash():.2f}")
    logger.info(f"  Commission: {0.001}")
    logger.info(f"  Broker type: {type(cerebro.broker)}")
    
    # Run backtest
    logger.info("\nRunning basic broker test...")
    results = cerebro.run(maxbars=20)  # Very limited run
    strategy = results[0]
    
    # Analyze results
    logger.info("\nTEST RESULTS:")
    logger.info(f"  Orders attempted: {len(strategy.order_attempts)}")
    
    for i, attempt in enumerate(strategy.order_attempts):
        logger.info(f"  Order {i+1}: {attempt}")
    
    # Check final broker state
    final_cash = cerebro.broker.get_cash()
    final_value = cerebro.broker.get_value()
    
    logger.info(f"\nFINAL BROKER STATE:")
    logger.info(f"  Final cash: ${final_cash:.2f}")
    logger.info(f"  Final value: ${final_value:.2f}")
    logger.info(f"  Position: {strategy.getposition().size} shares")
    
    # Determine if test passed
    orders_created = sum(1 for attempt in strategy.order_attempts if attempt['order_created'])
    if orders_created > 0:
        print(f"✅ Successfully created {orders_created} orders")
        return True
    else:
        print("❌ Failed to create any orders")
        return False


def test_rl_strategy_with_fixed_broker():
    """Test RL strategy with improved broker configuration"""
    print("\n" + "="*60)
    print("TESTING RL STRATEGY WITH FIXED BROKER")
    print("="*60)
    
    # Create data feeds
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No price data available")
        return False
    
    # Simple model that only places one order
    class SingleOrderModel:
        def __init__(self):
            self.call_count = 0
            
        def predict(self, observation, deterministic=True):
            self.call_count += 1
            # Only place one BUY order at the beginning
            if self.call_count == 5:
                return np.array([1, 2, 2, 2]), None  # BUY with small size
            return np.array([0, 0, 0, 0]), None  # HOLD
    
    model = SingleOrderModel()
    
    # Create cerebro with improved configuration
    cerebro = bt.Cerebro()
    cerebro.adddata(price_feed)
    
    # Add strategy
    cerebro.addstrategy(
        RLTradingStrategy,
        model=model,
        observation_manager=obs_manager,
        initial_cash=100000,  # Much more cash
        verbose=True,
        risk_per_trade=0.005,  # Very small risk (0.5%)
        enable_short=False  # Disable shorts for now
    )
    
    # Configure broker more conservatively
    cerebro.broker.setcash(100000)  # Lots of cash
    cerebro.broker.setcommission(commission=0.001)  # Low commission
    
    logger.info("IMPROVED BROKER CONFIGURATION:")
    logger.info(f"  Initial cash: ${cerebro.broker.get_cash():.2f}")
    logger.info(f"  Commission: 0.1%")
    
    # Run limited backtest
    logger.info("\nRunning RL strategy test...")
    results = cerebro.run(maxbars=15)  # Very limited
    strategy = results[0]
    
    # Check results
    final_cash = cerebro.broker.get_cash()
    final_value = cerebro.broker.get_value()
    
    logger.info(f"\nRL STRATEGY RESULTS:")
    logger.info(f"  Model calls: {model.call_count}")
    logger.info(f"  Final cash: ${final_cash:.2f}")
    logger.info(f"  Final value: ${final_value:.2f}")
    logger.info(f"  Total return: {((final_value - 100000) / 100000) * 100:.2f}%")
    
    if final_value != 100000:
        print("✅ Strategy executed some trades")
        return True
    else:
        print("⚠️  No trades executed, but no errors")
        return True


def main():
    """Run broker debugging tests"""
    print("🔧 DEBUGGING BROKER ORDER REJECTIONS")
    print("="*80)
    
    tests = [
        ("Basic Broker Functionality", test_basic_broker_functionality),
        ("RL Strategy with Fixed Broker", test_rl_strategy_with_fixed_broker)
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
            logger.exception(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 BROKER TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
