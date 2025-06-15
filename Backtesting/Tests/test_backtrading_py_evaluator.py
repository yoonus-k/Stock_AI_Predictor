#!/usr/bin/env python
"""
Backtesting.py Portfolio Evaluator Test Script

This script tests the BacktestingPyEvaluator functionality independently
using the existing trained RL model to evaluate its performance metrics.
Includes comprehensive trade history display functionality.

Usage:
    python test_backtrading_py_evaluator.py
"""
import os
import sys
import traceback
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime
#test removed
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import required modules
from stable_baselines3 import PPO
from RL.Envs.trading_env import TradingEnv
from RL.Data.Utils.loader import load_data_from_db
from Backtesting.evaluator import BacktestingPyEvaluator


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
                    f"models/{timeframe}/best_model_{timeframe}_10000.zip",
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
    Test the BacktestingPyEvaluator with the existing trained model
    """
    print("=" * 60)
    print("Backtesting.py Portfolio Evaluator Functionality Test")
    print("=" * 60)
    
    # Step 1: Load the trained model
    print("\n1. Loading trained RL model...")
    try:
        # First try to load from MLflow
        model_path, run_id = load_latest_model_from_mlflow("1H")
        
        if model_path is None:
            print("  ❌ Could not load model from MLflow, using placeholder")
            model_path = "model_placeholder"  # This will be handled later
            run_id = None

        model = PPO.load(model_path, env=None)
        
        if run_id:
            print(f"  ✅ Model loaded from MLflow run {run_id}")
        else:
            print(f"  ✅ Model loaded from path")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False
    
 
    
    # Step 3: Create the portfolio evaluator
    print("\n3. Initializing Backtesting.py Portfolio Evaluator...")
    try:
        portfolio_evaluator = BacktestingPyEvaluator(
            initial_cash=100000,
            # 3 USD / Lot , # 1 Lot = 100,000 USD , so 3 USD / 100,000 USD = 0.00003 , 
            # since we apply commission on open and close, we need to divide by 2, 
            # so final commission = 0.00003 / 2 = 0.000015
            commission=0.000015, 
            spread=0.0001, # formula : price * (1+x) = spreaded_price , where x is the spread
            margin=0.01,
            trade_on_close=True,
            hedging=True,
            exclusive_orders=False,
            finalize_trade_on_close=True,
            max_positions=10,
            risk_per_trade=0.02,
            enable_short=True,
            enable_hedging=True,
            position_sizing='fixed',
            verbose=True,
        )
        print("✅ Portfolio evaluator initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing portfolio evaluator: {e}")
        traceback.print_exc()
        return False
    
    # Step 6: Test the portfolio evaluator
    print("\n6. Testing Backtesting.py Portfolio Evaluator...")
    try:
        print("Running portfolio evaluation...")
        start_time = datetime.now()
        
        # Run the portfolio evaluation
        metrics = portfolio_evaluator.evaluate_portfolio(
            rl_model=model,
            timeframe='1H'
        )
        
        # print metrics in table format
        if metrics is None:
            print("❌ No metrics returned from evaluation")
            return False
        print("\n✅ Portfolio evaluation completed successfully")
        print("Metrics:")
        metrics_dict = metrics[0]
        # for key, value in metrics_dict.items():
        #     print(f"  {key}: {value}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Portfolio evaluation completed in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error during portfolio evaluation: {e}")
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    try:
        success = test_portfolio_evaluator()
        if success:
            print("\n🚀 Test completed successfully!")
        else:
            print("\n⚠️ Test completed with errors")
            
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
