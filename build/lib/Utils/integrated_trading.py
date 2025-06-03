#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Trading System

This script integrates pattern mining, sentiment analysis, and reinforcement learning
into a unified trading system that can:
1. Detect patterns in price data
2. Analyze sentiment from news and social media
3. Train and execute RL trading agents
4. Backtest strategies
5. Generate trading signals

Usage:
    python integrated_trading.py --mode [train|backtest|live] --stock_id [ID] --timeframe_id [ID]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent  # Navigate up to project root
sys.path.append(str(project_root))

# Import custom modules
from Pattern.pip_pattern_miner import Pattern_Miner
from Data.Database.db_cloud import Database
from RL.Agents.dqn_agent import DQNAgent  # Adjust based on your actual RL implementation

class IntegratedTradingSystem:
    """Integrated system combining pattern mining, sentiment, and RL."""
    
    def __init__(self, stock_id, timeframe_id, config_id=None):
        """Initialize the integrated trading system.
        
        Args:
            stock_id: ID of the stock to trade
            timeframe_id: ID of the timeframe to use
            config_id: Optional ID of the parameter configuration to use
        """
        # Connect to database
        self.db = Database()
        
        # Store parameters
        self.stock_id = stock_id
        self.timeframe_id = timeframe_id
        
        # Get stock and timeframe info
        self.stock_info = self.get_stock_info(stock_id)
        self.timeframe_info = self.get_timeframe_info(timeframe_id)
        
        print(f"Initializing system for {self.stock_info['symbol']} on "
              f"{self.timeframe_info['name']} timeframe")
        
        # Get or set config_id
        if config_id is None:
            # Use the best performing config for this stock and timeframe
            config_id = self.get_best_config()
        
        self.config_id = config_id
        self.config_info = self.get_config_info(config_id)
        
        # Initialize components
        self.pattern_miner = None
        self.rl_agent = None
    
    def get_stock_info(self, stock_id):
        """Get stock information from the database."""
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT stock_id, symbol, name FROM stocks WHERE stock_id = ?", (stock_id,))
        result = cursor.fetchone()
        
        if result:
            return {
                'stock_id': result[0],
                'symbol': result[1],
                'name': result[2]
            }
        else:
            raise ValueError(f"Stock with ID {stock_id} not found in database")
    
    def get_timeframe_info(self, timeframe_id):
        """Get timeframe information from the database."""
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT timeframe_id, minutes, name FROM timeframes WHERE timeframe_id = ?", (timeframe_id,))
        result = cursor.fetchone()
        
        if result:
            return {
                'timeframe_id': result[0],
                'minutes': result[1],
                'name': result[2]
            }
        else:
            raise ValueError(f"Timeframe with ID {timeframe_id} not found in database")
    
    def get_config_info(self, config_id):
        """Get configuration information from the database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """SELECT config_id, name, n_pips, lookback, hold_period, 
                     returns_hold_period, distance_measure, description
               FROM experiment_configs WHERE config_id = ?""", 
            (config_id,)
        )
        result = cursor.fetchone()
        
        if result:
            return {
                'config_id': result[0],
                'name': result[1],
                'n_pips': result[2],
                'lookback': result[3],
                'hold_period': result[4],
                'returns_hold_period': result[5],
                'distance_measure': result[6],
                'description': result[7]
            }
        else:
            raise ValueError(f"Configuration with ID {config_id} not found in database")
    
    def get_best_config(self):
        """Get the best performing configuration for this stock and timeframe."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """SELECT config_id FROM performance_metrics
               WHERE stock_id = ? AND timeframe_id = ?
               ORDER BY win_rate * profit_factor DESC
               LIMIT 1""",
            (self.stock_id, self.timeframe_id)
        )
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            # Return default config if no performance metrics are available
            cursor.execute("SELECT config_id FROM experiment_configs LIMIT 1")
            return cursor.fetchone()[0]
    
    def get_stock_data(self, start_date=None, end_date=None):
        """Get stock data for the specified period."""
        query = """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM stock_data
            WHERE stock_id = ? AND timeframe_id = ?
        """
        
        params = [self.stock_id, self.timeframe_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        # Execute query
        cursor = self.db.connection.cursor()
        cursor.execute(query, params)
        data = cursor.fetchall()
        
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def initialize_pattern_miner(self, data=None):
        """Initialize and train the pattern miner."""
        # Create pattern miner with configuration parameters
        self.pattern_miner = Pattern_Miner(
            n_pips=self.config_info['n_pips'],
            lookback=self.config_info['lookback'],
            hold_period=self.config_info['hold_period'],
            returns_hold_period=self.config_info['returns_hold_period'],
            distance_measure=self.config_info['distance_measure']
        )
        
        # If data is not provided, retrieve it from the database and train
        if data is None:
            # Get all available data for training
            df = self.get_stock_data()
            if df is None or len(df) < 100:
                raise ValueError("Insufficient data for pattern mining")
            
            print(f"Training pattern miner on {len(df)} data points")
            self.pattern_miner.train(df['close'].values)
        else:
            # Train on provided data
            print(f"Training pattern miner on provided data ({len(data)} points)")
            self.pattern_miner.train(data)
        
        print(f"Pattern miner initialized with {len(self.pattern_miner._unique_pip_patterns)} patterns "
              f"grouped into {len(self.pattern_miner._cluster_centers)} clusters")
    
    def get_sentiment_data(self, start_date=None, end_date=None):
        """Get sentiment data for the specified period."""
        # Check if the sentiment_data table exists
        cursor = self.db.connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'"
        )
        
        if not cursor.fetchone():
            print("Warning: sentiment_data table not found in database")
            return None
        
        # Query sentiment data
        query = """
            SELECT timestamp, sentiment_score, sentiment_magnitude
            FROM sentiment_data
            WHERE stock_id = ?
        """
        
        params = [self.stock_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        # Execute query
        cursor.execute(query, params)
        data = cursor.fetchall()
        
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'sentiment_score', 'sentiment_magnitude'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def initialize_rl_agent(self, state_size, action_size):
        """Initialize the RL agent."""
        # Check if RL modules are available in the project
        try:
            # This should be adjusted based on your actual RL implementation
            self.rl_agent = DQNAgent(state_size, action_size)
            print(f"RL agent initialized with state size {state_size} and action size {action_size}")
        except Exception as e:
            print(f"Warning: Could not initialize RL agent: {e}")
            self.rl_agent = None
    
    def prepare_state(self, price_window, pattern_id=None, sentiment=None):
        """Prepare a state representation for the RL agent.
        
        This combines price data, pattern information, and sentiment into a state vector.
        """
        # Base state includes normalized price changes
        price_changes = np.diff(price_window) / price_window[:-1]
        normalized_changes = (price_changes - np.mean(price_changes)) / (np.std(price_changes) + 1e-8)
        
        # If a pattern was detected, include its features
        pattern_features = []
        if pattern_id is not None and self.pattern_miner is not None:
            # Get the pattern's cluster
            cluster_id = self.pattern_miner.get_pattern_cluster(pattern_id)
            if cluster_id is not None:
                # Include expected outcome and probability
                outcome = self.pattern_miner.get_cluster_outcome(cluster_id)
                probability = self.pattern_miner.get_cluster_probability(cluster_id)
                pattern_features = [outcome, probability]
        
        # Include sentiment if available
        sentiment_features = []
        if sentiment is not None:
            sentiment_features = [sentiment['sentiment_score'], sentiment['sentiment_magnitude']]
        
        # Combine all features into a state vector
        state = np.concatenate([
            normalized_changes,
            pattern_features if pattern_features else [0, 0],
            sentiment_features if sentiment_features else [0, 0]
        ])
        
        return state
    
    def train_rl_agent(self, episodes=1000, start_date=None, end_date=None):
        """Train the RL agent on historical data."""
        if self.rl_agent is None:
            print("RL agent not initialized")
            return
        
        # Get training data
        df = self.get_stock_data(start_date, end_date)
        if df is None or len(df) < 100:
            raise ValueError("Insufficient data for RL training")
        
        # Initialize pattern miner if not already done
        if self.pattern_miner is None:
            self.initialize_pattern_miner(df['close'].values)
        
        # Get sentiment data if available
        sentiment_df = self.get_sentiment_data(start_date, end_date)
        
        print(f"Training RL agent on {len(df)} data points for {episodes} episodes")
        
        # Training loop
        rewards_history = []
        
        for episode in range(episodes):
            print(f"Episode {episode+1}/{episodes}")
            
            # Reset environment state
            episode_reward = 0
            lookback = self.config_info['lookback']
            
            # Start from a random point with enough history
            start_idx = lookback
            current_idx = start_idx
            
            # Initial state
            price_window = df['close'].values[current_idx-lookback:current_idx]
            
            # Detect pattern
            pattern_id = self.pattern_miner.identify_pattern(price_window)
            
            # Get sentiment if available
            sentiment = None
            if sentiment_df is not None:
                # Find the closest sentiment data point
                timestamp = df.index[current_idx]
                closest_idx = sentiment_df.index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx >= 0:
                    sentiment = {
                        'sentiment_score': sentiment_df['sentiment_score'].iloc[closest_idx],
                        'sentiment_magnitude': sentiment_df['sentiment_magnitude'].iloc[closest_idx]
                    }
            
            # Prepare initial state
            state = self.prepare_state(price_window, pattern_id, sentiment)
            
            done = False
            while not done:
                # Agent selects action
                action = self.rl_agent.act(state)
                
                # Take action and observe new state
                current_idx += 1
                if current_idx >= len(df) - 1:
                    done = True
                    continue
                
                # Calculate reward based on price movement and action
                # Action: 0=Hold, 1=Buy, 2=Sell
                price_change = df['close'].values[current_idx] / df['close'].values[current_idx-1] - 1
                
                if action == 1:  # Buy
                    reward = price_change
                elif action == 2:  # Sell
                    reward = -price_change
                else:  # Hold
                    reward = 0
                
                # Update state
                price_window = df['close'].values[current_idx-lookback:current_idx]
                pattern_id = self.pattern_miner.identify_pattern(price_window)
                
                # Update sentiment if available
                if sentiment_df is not None:
                    timestamp = df.index[current_idx]
                    closest_idx = sentiment_df.index.get_indexer([timestamp], method='nearest')[0]
                    if closest_idx >= 0:
                        sentiment = {
                            'sentiment_score': sentiment_df['sentiment_score'].iloc[closest_idx],
                            'sentiment_magnitude': sentiment_df['sentiment_magnitude'].iloc[closest_idx]
                        }
                
                next_state = self.prepare_state(price_window, pattern_id, sentiment)
                
                # Store experience and learn
                self.rl_agent.remember(state, action, reward, next_state, done)
                self.rl_agent.replay()
                
                state = next_state
                episode_reward += reward
                
                # End episode if loss exceeds threshold
                if episode_reward < -0.5:  # -50% stop loss
                    done = True
            
            rewards_history.append(episode_reward)
            print(f"Episode {episode+1} reward: {episode_reward:.4f}")
            
            # Save agent periodically
            if (episode + 1) % 10 == 0:
                self.save_rl_agent(f"episode_{episode+1}")
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history)
        plt.title('RL Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(f"RL_training_{self.stock_info['symbol']}_{self.timeframe_info['name']}.png")
        plt.close()
        
        print("RL agent training completed")
    
    def save_rl_agent(self, version_name):
        """Save the RL agent to the database and file system."""
        if self.rl_agent is None:
            return
        
        # Save agent to file
        model_dir = project_root / "RL" / "Models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = model_dir / f"{self.stock_info['symbol']}_{self.timeframe_info['name']}_{version_name}.h5"
        self.rl_agent.save(str(model_path))
        
        # Record in database if rl_models table exists
        cursor = self.db.connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rl_models'"
        )
        
        if cursor.fetchone():
            # Check if model already exists
            cursor.execute(
                "SELECT model_id FROM rl_models WHERE name = ?",
                (f"{self.stock_info['symbol']}_{self.timeframe_info['name']}_agent",)
            )
            result = cursor.fetchone()
            
            if result:
                model_id = result[0]
                
                # Insert new version
                cursor.execute(
                    "SELECT MAX(version_number) FROM rl_model_versions WHERE model_id = ?",
                    (model_id,)
                )
                max_version = cursor.fetchone()[0] or 0
                
                cursor.execute(
                    """INSERT INTO rl_model_versions 
                       (model_id, version_number, model_file_path, training_episodes)
                       VALUES (?, ?, ?, ?)""",
                    (model_id, max_version + 1, str(model_path), 1000)
                )
            else:
                # Insert new model
                cursor.execute(
                    """INSERT INTO rl_models 
                       (name, description, model_type, input_features)
                       VALUES (?, ?, ?, ?)""",
                    (
                        f"{self.stock_info['symbol']}_{self.timeframe_info['name']}_agent",
                        f"RL agent for {self.stock_info['symbol']} on {self.timeframe_info['name']} timeframe",
                        "DQN",
                        json.dumps(["price_changes", "pattern_features", "sentiment"])
                    )
                )
                model_id = cursor.lastrowid
                
                # Insert first version
                cursor.execute(
                    """INSERT INTO rl_model_versions 
                       (model_id, version_number, model_file_path, training_episodes)
                       VALUES (?, ?, ?, ?)""",
                    (model_id, 1, str(model_path), 1000)
                )
            
            self.db.connection.commit()
    
    def backtest(self, start_date=None, end_date=None, use_rl=True, use_patterns=True, use_sentiment=True):
        """Backtest the trading strategy."""
        # Get data for backtesting
        df = self.get_stock_data(start_date, end_date)
        if df is None or len(df) < 100:
            raise ValueError("Insufficient data for backtesting")
        
        # Initialize components if needed
        if use_patterns and self.pattern_miner is None:
            # Use the first 70% of data for training
            train_size = int(len(df) * 0.7)
            train_data = df['close'].values[:train_size]
            self.initialize_pattern_miner(train_data)
            
            # Use the remaining 30% for testing
            df = df.iloc[train_size:]
        
        # Get sentiment data if available and requested
        sentiment_df = None
        if use_sentiment:
            sentiment_df = self.get_sentiment_data(
                df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')
            )
        
        # Initialize RL agent if needed
        if use_rl and self.rl_agent is None:
            # State size: price changes (lookback-1) + pattern features (2) + sentiment (2)
            state_size = (self.config_info['lookback'] - 1) + 2 + 2
            action_size = 3  # Hold, Buy, Sell
            self.initialize_rl_agent(state_size, action_size)
            
            # Load the agent if available
            # This is just a placeholder - actual loading depends on your implementation
            model_path = project_root / "RL" / "Models" / f"{self.stock_info['symbol']}_{self.timeframe_info['name']}_latest.h5"
            if os.path.exists(model_path):
                self.rl_agent.load(str(model_path))
        
        print(f"Backtesting on {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
        # Prepare results storage
        lookback = self.config_info['lookback']
        hold_period = self.config_info['hold_period']
        
        results = []
        trades = []
        
        # Initial capital
        capital = 10000.0
        position = 0  # 0=None, 1=Long, -1=Short
        entry_price = 0
        entry_date = None
        
        for i in range(lookback, len(df) - hold_period):
            timestamp = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Skip if we're still in the hold period of a trade
            if position != 0 and (timestamp - entry_date).days < hold_period:
                continue
            
            # Detect pattern
            price_window = df['close'].values[i-lookback:i]
            pattern_id = None
            pattern_prediction = None
            
            if use_patterns:
                pattern_id = self.pattern_miner.identify_pattern(price_window)
                if pattern_id is not None:
                    pattern_prediction = self.pattern_miner.predict_outcome(pattern_id)
            
            # Get sentiment
            sentiment = None
            if sentiment_df is not None and use_sentiment:
                closest_idx = sentiment_df.index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx >= 0:
                    sentiment = {
                        'sentiment_score': sentiment_df['sentiment_score'].iloc[closest_idx],
                        'sentiment_magnitude': sentiment_df['sentiment_magnitude'].iloc[closest_idx]
                    }
            
            # Determine action
            action = 0  # Default: Hold
            
            if use_rl and self.rl_agent is not None:
                # Prepare state for RL agent
                state = self.prepare_state(price_window, pattern_id, sentiment)
                # Get action from RL agent
                action = self.rl_agent.act(state, explore=False)
            else:
                # Use pattern-based strategy
                if pattern_prediction is not None:
                    if pattern_prediction > 0.01:  # 1% threshold for long
                        action = 1  # Buy
                    elif pattern_prediction < -0.01:  # -1% threshold for short
                        action = 2  # Sell
            
            # Execute action
            if position == 0:  # No position
                if action == 1:  # Buy
                    position = 1
                    entry_price = current_price
                    entry_date = timestamp
                    trades.append({
                        'date': timestamp,
                        'action': 'BUY',
                        'price': current_price,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
                elif action == 2:  # Sell
                    position = -1
                    entry_price = current_price
                    entry_date = timestamp
                    trades.append({
                        'date': timestamp,
                        'action': 'SELL',
                        'price': current_price,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
            elif position == 1:  # Long position
                if action == 2:  # Close long and go short
                    # Calculate profit/loss
                    pnl = (current_price / entry_price - 1) * capital
                    capital += pnl
                    
                    trades.append({
                        'date': timestamp,
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
                    
                    # Go short
                    position = -1
                    entry_price = current_price
                    entry_date = timestamp
                    
                    trades.append({
                        'date': timestamp,
                        'action': 'SELL',
                        'price': current_price,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
                elif i == len(df) - hold_period - 1:  # Last data point, close position
                    # Calculate profit/loss
                    pnl = (current_price / entry_price - 1) * capital
                    capital += pnl
                    position = 0
                    
                    trades.append({
                        'date': timestamp,
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
            elif position == -1:  # Short position
                if action == 1:  # Close short and go long
                    # Calculate profit/loss
                    pnl = (entry_price / current_price - 1) * capital
                    capital += pnl
                    
                    trades.append({
                        'date': timestamp,
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
                    
                    # Go long
                    position = 1
                    entry_price = current_price
                    entry_date = timestamp
                    
                    trades.append({
                        'date': timestamp,
                        'action': 'BUY',
                        'price': current_price,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
                elif i == len(df) - hold_period - 1:  # Last data point, close position
                    # Calculate profit/loss
                    pnl = (entry_price / current_price - 1) * capital
                    capital += pnl
                    position = 0
                    
                    trades.append({
                        'date': timestamp,
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital,
                        'pattern_id': pattern_id,
                        'pattern_prediction': pattern_prediction
                    })
            
            # Record daily results
            results.append({
                'date': timestamp,
                'price': current_price,
                'position': position,
                'capital': capital
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        trades_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        initial_capital = 10000.0
        final_capital = capital
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Calculate daily returns
        results_df['daily_return'] = results_df['capital'].pct_change().fillna(0)
        
        # Calculate metrics
        sharpe_ratio = results_df['daily_return'].mean() / results_df['daily_return'].std() * np.sqrt(252)
        max_drawdown = (results_df['capital'].cummax() - results_df['capital']) / results_df['capital'].cummax()
        max_drawdown = max_drawdown.max() * 100
        
        # Calculate win rate
        if len(trades_df) > 0:
            wins = trades_df[trades_df['pnl'] > 0].shape[0]
            win_rate = wins / trades_df[trades_df['pnl'].notna()].shape[0] * 100
        else:
            win_rate = 0
        
        # Print results
        print("\nBacktesting Results:")
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Final Capital: ${final_capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Number of Trades: {trades_df[trades_df['pnl'].notna()].shape[0]}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['date'], results_df['capital'])
        plt.title(f"Equity Curve: {self.stock_info['symbol']} ({self.timeframe_info['name']})")
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        plt.tight_layout()
        
        # Plot entry/exit points
        buy_signals = trades_df[trades_df['action'] == 'BUY']
        sell_signals = trades_df[trades_df['action'] == 'SELL']
        close_long = trades_df[trades_df['action'] == 'CLOSE_LONG']
        close_short = trades_df[trades_df['action'] == 'CLOSE_SHORT']
        
        plt.scatter(buy_signals['date'], buy_signals['price'], marker='^', color='green', label='Buy')
        plt.scatter(sell_signals['date'], sell_signals['price'], marker='v', color='red', label='Sell')
        plt.scatter(close_long['date'], close_long['price'], marker='o', color='blue', label='Close Long')
        plt.scatter(close_short['date'], close_short['price'], marker='o', color='purple', label='Close Short')
        
        plt.legend()
        
        # Save plot
        output_dir = project_root / "Images" / "Backtesting"
        os.makedirs(output_dir, exist_ok=True)
        
        strategy_name = []
        if use_patterns:
            strategy_name.append("Patterns")
        if use_sentiment:
            strategy_name.append("Sentiment")
        if use_rl:
            strategy_name.append("RL")
        
        strategy_str = "_".join(strategy_name) if strategy_name else "Combined"
        plt.savefig(output_dir / f"Backtest_{self.stock_info['symbol']}_{self.timeframe_info['name']}_{strategy_str}.png")
        plt.close()
        
        # Save detailed results
        trades_df.to_csv(output_dir / f"Trades_{self.stock_info['symbol']}_{self.timeframe_info['name']}_{strategy_str}.csv", index=False)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': trades_df[trades_df['pnl'].notna()].shape[0],
            'trades_df': trades_df,
            'results_df': results_df
        }
    
    def generate_signals(self, days_back=30):
        """Generate trading signals for the most recent data."""
        # Get recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        df = self.get_stock_data(start_date, end_date)
        if df is None or len(df) < self.config_info['lookback']:
            raise ValueError("Insufficient recent data for signal generation")
        
        # Make sure pattern miner is initialized
        if self.pattern_miner is None:
            self.initialize_pattern_miner(df['close'].values)
        
        # Get sentiment data if available
        sentiment_df = self.get_sentiment_data(start_date, end_date)
        
        # Get most recent data point
        latest_idx = len(df) - 1
        latest_timestamp = df.index[latest_idx]
        latest_price = df['close'].iloc[latest_idx]
        
        # Extract window for pattern detection
        lookback = self.config_info['lookback']
        price_window = df['close'].values[latest_idx-lookback+1:latest_idx+1]
        
        # Detect pattern
        pattern_id = self.pattern_miner.identify_pattern(price_window)
        pattern_prediction = None
        cluster_id = None
        
        if pattern_id is not None:
            pattern_prediction = self.pattern_miner.predict_outcome(pattern_id)
            cluster_id = self.pattern_miner.get_pattern_cluster(pattern_id)
        
        # Get sentiment
        sentiment = None
        if sentiment_df is not None:
            closest_idx = sentiment_df.index.get_indexer([latest_timestamp], method='nearest')[0]
            if closest_idx >= 0:
                sentiment = {
                    'sentiment_score': sentiment_df['sentiment_score'].iloc[closest_idx],
                    'sentiment_magnitude': sentiment_df['sentiment_magnitude'].iloc[closest_idx]
                }
        
        # Determine signal
        signal = "HOLD"
        confidence = 0.5
        
        if pattern_prediction is not None:
            if pattern_prediction > 0.01:  # 1% threshold for buy
                signal = "BUY"
                confidence = abs(pattern_prediction) * 10  # Scale to 0-1 range
            elif pattern_prediction < -0.01:  # -1% threshold for sell
                signal = "SELL"
                confidence = abs(pattern_prediction) * 10  # Scale to 0-1 range
        
        # Adjust confidence based on sentiment
        if sentiment is not None:
            # Align sentiment direction with signal
            if signal == "BUY" and sentiment['sentiment_score'] > 0:
                confidence = min(0.95, confidence + 0.1 * sentiment['sentiment_score'])
            elif signal == "SELL" and sentiment['sentiment_score'] < 0:
                confidence = min(0.95, confidence + 0.1 * abs(sentiment['sentiment_score']))
            elif signal != "HOLD":
                confidence = max(0.05, confidence - 0.1 * abs(sentiment['sentiment_score']))
        
        # Use RL agent if available
        if self.rl_agent is not None:
            state = self.prepare_state(price_window, pattern_id, sentiment)
            rl_action = self.rl_agent.act(state, explore=False)
            
            if rl_action == 1 and signal != "BUY":  # RL suggests buy
                confidence = max(confidence, 0.6)  # Increase confidence
                if signal == "HOLD":
                    signal = "BUY"
            elif rl_action == 2 and signal != "SELL":  # RL suggests sell
                confidence = max(confidence, 0.6)  # Increase confidence
                if signal == "HOLD":
                    signal = "SELL"
        
        # Store signal in database
        try:
            cursor = self.db.connection.cursor()
            
            # Check if predictions table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            )
            
            if cursor.fetchone():
                # Insert prediction
                cursor.execute(
                    """INSERT INTO predictions 
                       (stock_id, pattern_id, timeframe_id, config_id, 
                        prediction_date, predicted_outcome, confidence_level)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self.stock_id,
                        pattern_id if pattern_id is not None else None,
                        self.timeframe_id,
                        self.config_id,
                        latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        json.dumps({
                            'signal': signal,
                            'price': float(latest_price),
                            'pattern_prediction': float(pattern_prediction) if pattern_prediction is not None else None,
                            'cluster_id': int(cluster_id) if cluster_id is not None else None,
                            'sentiment': sentiment
                        }),
                        float(confidence)
                    )
                )
                self.db.connection.commit()
        except Exception as e:
            print(f"Warning: Could not store prediction in database: {e}")
        
        # Print signal
        print("\nTrading Signal:")
        print(f"Date: {latest_timestamp}")
        print(f"Symbol: {self.stock_info['symbol']}")
        print(f"Timeframe: {self.timeframe_info['name']}")
        print(f"Price: ${latest_price:.2f}")
        print(f"Signal: {signal}")
        print(f"Confidence: {confidence:.2f}")
        
        if pattern_id is not None:
            print(f"Pattern ID: {pattern_id}")
            print(f"Cluster ID: {cluster_id}")
            print(f"Expected Outcome: {pattern_prediction:.4f}")
        
        if sentiment is not None:
            print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}")
            print(f"Sentiment Magnitude: {sentiment['sentiment_magnitude']:.2f}")
        
        return {
            'date': latest_timestamp,
            'symbol': self.stock_info['symbol'],
            'timeframe': self.timeframe_info['name'],
            'price': latest_price,
            'signal': signal,
            'confidence': confidence,
            'pattern_id': pattern_id,
            'cluster_id': cluster_id,
            'pattern_prediction': pattern_prediction,
            'sentiment': sentiment
        }
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'db') and self.db is not None:
            self.db.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Integrated Trading System')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'live'], 
                        default='backtest', help='Operating mode')
    parser.add_argument('--stock_id', type=int, required=True, help='Stock ID to trade')
    parser.add_argument('--timeframe_id', type=int, required=True, help='Timeframe ID to use')
    parser.add_argument('--config_id', type=int, help='Configuration ID to use')
    parser.add_argument('--start_date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--use_rl', action='store_true', help='Use reinforcement learning')
    parser.add_argument('--use_patterns', action='store_true', help='Use pattern recognition')
    parser.add_argument('--use_sentiment', action='store_true', help='Use sentiment analysis')
    
    args = parser.parse_args()
    
    try:
        # Create integrated system
        system = IntegratedTradingSystem(args.stock_id, args.timeframe_id, args.config_id)
        
        if args.mode == 'train':
            # Train the RL agent
            system.train_rl_agent(episodes=1000, start_date=args.start_date, end_date=args.end_date)
        
        elif args.mode == 'backtest':
            # Backtest the strategy
            system.backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                use_rl=args.use_rl,
                use_patterns=args.use_patterns,
                use_sentiment=args.use_sentiment
            )
        
        elif args.mode == 'live':
            # Generate trading signals
            system.generate_signals()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up
        if 'system' in locals():
            system.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
