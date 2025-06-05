#!/usr/bin/env python
"""
Portfolio Evaluator Test Script

This script tests the BacktraderPortfolioEvaluator functionality independently
using the existing trained RL model to evaluate its performance metrics.

Usage:
    python test_portfolio_evaluator.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import required modules
from stable_baselines3 import PPO
from RL.Envs.trading_env import TradingEnv
from RL.Data.Utils.loader import load_data_from_db
from Backtrader.portfolio_evaluator import BacktraderPortfolioEvaluator
import mlflow
import mlflow.tracking

def load_latest_model_from_mlflow(timeframe="1H"):
    """
    Load the latest trained model for the specified timeframe from MLflow
    
    Args:
        timeframe: Model timeframe (e.g., "1H", "4H", "1D")
    
    Returns:
        tuple: (model, run_id) or (None, None) if not found
    """
    try:
        # Set MLflow tracking URI to local mlruns folder
        mlflow.set_tracking_uri("file:///D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/mlruns")
        
        # Set experiment name based on timeframe
        experiment_name = f"stock_trading_rl_{timeframe}"
        
        print(f"  Looking for experiment: {experiment_name}")
        
        # Search for runs in the experiment
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["start_time DESC"],  # Get latest runs first
            max_results=5
        )
        
        if runs.empty:
            print(f"  ❌ No runs found in experiment {experiment_name}")
            return None, None
        
        print(f"  ✅ Found {len(runs)} runs in experiment")
        
        # Try to find the best model artifact from the latest runs
        for _, run in runs.iterrows():
            run_id = run['run_id']
            print(f"  Checking run: {run_id[:8]}...")
            
            try:
                # List artifacts for this run
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run_id, "models")
                
                if not artifacts:
                    print(f"    No models folder found")
                    continue
                
                # Look for timeframe-specific model files
                model_artifacts = client.list_artifacts(run_id, f"models/{timeframe}")
                
                if not model_artifacts:
                    print(f"    No {timeframe} models found")
                    continue
                
                # Find the best model file (prefer best_model, then final_model, then base model)
                model_files = [a.path for a in model_artifacts if a.path.endswith('.zip')]
                
                if not model_files:
                    print(f"    No .zip model files found")
                    continue
                
                # Priority order for model selection
                model_priorities = [
                    f"models/{timeframe}/best_model_{timeframe}",
                    f"models/{timeframe}/{timeframe}_model.zip",
                    f"models/{timeframe}/final_model_{timeframe}"
                ]
                
                selected_model_path = None
                for priority_path in model_priorities:
                    matching_files = [f for f in model_files if priority_path in f]
                    if matching_files:
                        selected_model_path = matching_files[0]
                        break
                
                if not selected_model_path:
                    selected_model_path = model_files[0]  # Fallback to first available
                
                print(f"    Found model: {selected_model_path}")
                
                # Download and load the model
                model_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=selected_model_path
                )
                
                print(f"    Downloaded model to: {model_path}")
                return model_path, run_id
                
            except Exception as e:
                print(f"    Error checking run {run_id[:8]}: {e}")
                continue
        
        print(f"  ❌ No suitable model found in any run")
        return None, None
        
    except Exception as e:
        print(f"  ❌ Error loading model from MLflow: {e}")
        return None, None


def test_portfolio_evaluator():
    """
    Test the BacktraderPortfolioEvaluator with the existing trained model
    """
    print("=" * 60)
    print("Portfolio Evaluator Functionality Test")
    print("=" * 60)
      # Step 1: Load the trained model
    print("\n1. Loading trained RL model...")
    try:
        # First try to load from MLflow
        model_path, run_id = load_latest_model_from_mlflow("1H")
        
        if model_path is None:
            print("  Falling back to static model file...")
            model_path = "RL/Models/Experiments/best_model.zip"
            if not os.path.exists(model_path):
                print(f"❌ Model not found at {model_path}")
                return False
          # Load actual data for temporary environment
        print("  Loading actual data for model environment...")
        temp_data = load_data_from_db()  # 1H = 60 minutes
        if temp_data.empty:
            print("❌ No data found in database for temp environment")
            return False
            
        # Use a small sample for temp environment (just need structure)
        temp_sample = temp_data.head(10).copy()
        print(f"  Using {len(temp_sample)} records for temp environment")
        print(f"  Temp data columns: {list(temp_sample.columns)}")
        
        temp_env = TradingEnv(temp_sample, normalize_observations=True)
        model = PPO.load(model_path, env=temp_env)
        
        if run_id:
            print(f"✅ Successfully loaded model from MLflow run: {run_id[:8]}")
        else:
            print(f"✅ Successfully loaded model from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False
      # Step 2: Load evaluation data
    print("\n2. Loading evaluation data...")
    try:
        # Load data from database
        data = load_data_from_db()  # 1H = 60 minutes
        if data.empty:
            print("❌ No data found in database")
            return False
            
        # Use a subset for testing (last 500 records for recent data)
        eval_data = data.head(100).copy()
        print(f"✅ Loaded {len(eval_data)} records for evaluation")
        print(f"Data date range: {eval_data.index.min()} to {eval_data.index.max()}")
        
    except Exception as e:
        print(f"❌ Error loading evaluation data: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: Create the portfolio evaluator
    print("\n3. Initializing Portfolio Evaluator...")
    try:
        portfolio_evaluator = BacktraderPortfolioEvaluator(
            initial_cash=100000,
            commission=0.001,      # 0.1% commission
            slippage=0.0005,       # 0.05% slippage
            enable_short=True,
            enable_hedging=True,
            max_positions=5,
            risk_per_trade=0.02,   # 2% risk per trade
            position_sizing='fixed',
            verbose=True
        )
        print("✅ Portfolio evaluator initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing portfolio evaluator: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Create trading environment for generating actions
    print("\n4. Creating trading environment...")
    try:
        trading_env = TradingEnv(
            eval_data,
            initial_balance=100000,
            normalize_observations=True,
            reward_type='combined'
        )
        print("✅ Trading environment created successfully")
        
    except Exception as e:
        print(f"❌ Error creating trading environment: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Generate model predictions/actions for the evaluation data
    print("\n5. Generating model predictions...")
    try:
        actions_data = []
        rewards_data = []
        observations_data = []
        
        obs, _ = trading_env.reset()
        done = False
        step_count = 0
        max_steps = min(len(eval_data) - 1, 200)  # Limit to 200 steps for testing
        
        print(f"Running simulation for {max_steps} steps...")
        
        while not done and step_count < max_steps:
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            
            # Store data
            actions_data.append(action.copy())
            observations_data.append(obs.copy())
            
            # Execute step in environment
            obs, reward, done, truncated, info = trading_env.step(action)
            rewards_data.append(reward)
            
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"  Completed {step_count}/{max_steps} steps...")
            
            if done or truncated:
                break
        
        print(f"✅ Generated {len(actions_data)} action predictions")
        print(f"Final portfolio balance: ${info.get('portfolio_balance', 0):,.2f}")
        
    except Exception as e:
        print(f"❌ Error generating model predictions: {e}")
        traceback.print_exc()
        return False
    
    # Step 6: Test the portfolio evaluator
    print("\n6. Testing Portfolio Evaluator...")
    try:
        # Prepare environment data for portfolio evaluation
        environment_data = {
            'observations': np.array(observations_data),
            'actions': np.array(actions_data),
            'rewards': np.array(rewards_data),
            'market_data': eval_data.iloc[:len(actions_data)].copy()
        }
        
        print("Running portfolio evaluation...")
        start_time = datetime.now()
        
        # Run the portfolio evaluation
        metrics = portfolio_evaluator.evaluate_portfolio(
            rl_model=model,
            environment_data=environment_data,
            episode_length=len(actions_data),
            timeframe='1H'
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Portfolio evaluation completed in {duration:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during portfolio evaluation: {e}")
        traceback.print_exc()
        return False
    
    # Step 7: Display results
    print("\n7. Portfolio Evaluation Results:")
    print("=" * 50)
    
    if metrics:
        # Core performance metrics
        core_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades',
            'win_rate', 'profit_factor', 'final_portfolio_value', 'volatility'
        ]
        
        print("\n📊 Core Performance Metrics:")
        for metric in core_metrics:
            if metric in metrics:
                value = metrics[metric]
                if metric == 'total_return' or metric == 'max_drawdown' or metric == 'volatility':
                    print(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                elif metric == 'final_portfolio_value':
                    print(f"  {metric.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Additional metrics
        print(f"\n📈 Additional Metrics:")
        other_metrics = {k: v for k, v in metrics.items() if k not in core_metrics}
        for metric, value in other_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if abs(value) < 1 and abs(value) > 0.001:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                elif abs(value) >= 1000:
                    print(f"  {metric.replace('_', ' ').title()}: {value:,.2f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  {metric.replace('_', ' ').title()}: {value}")
    else:
        print("❌ No metrics returned from portfolio evaluation")
        return False
    
    # Step 8: Generate summary assessment
    print("\n8. Assessment Summary:")
    print("=" * 50)
    
    try:
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        
        print(f"🎯 Strategy Performance Assessment:")
        print(f"  Return: {total_return:.2%} ({'Positive' if total_return > 0 else 'Negative'})")
        print(f"  Risk-Adjusted Return (Sharpe): {sharpe_ratio:.2f} ({'Good' if sharpe_ratio > 1 else 'Poor' if sharpe_ratio < 0 else 'Fair'})")
        print(f"  Maximum Drawdown: {max_drawdown:.2%} ({'Acceptable' if abs(max_drawdown) < 0.1 else 'High'})")
        print(f"  Win Rate: {win_rate:.1%} ({'Good' if win_rate > 0.5 else 'Poor'})")
        print(f"  Trading Activity: {total_trades} trades ({'Active' if total_trades > 10 else 'Conservative'})")
        
        if 'error' in metrics:
            print(f"⚠️  Note: {metrics['error']}")
        
    except Exception as e:
        print(f"⚠️  Error generating assessment: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Portfolio Evaluator Test Completed Successfully!")
    print("=" * 60)
    
    return True

def main():
    """Main execution function"""
    try:
        success = test_portfolio_evaluator()
        if success:
            print("\n🎉 All tests passed! Portfolio evaluator is working correctly.")
        else:
            print("\n❌ Some tests failed. Please review the errors above.")
            
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
