"""
Complete RL training pipeline with MLflow integration
Trains individual timeframe models and a meta-model that aggregates them
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import training scripts
from RL.Scripts.Train.train_timeframe_model import train_timeframe_model
from RL.Scripts.Train.train_meta_model import train_meta_model
from RL.Analysis.experiment_analyzer import ExperimentAnalyzer


def run_training_pipeline(
    timeframes: List[str] = ["daily", "weekly", "monthly", "meta"],
    timesteps_per_timeframe: Dict[str, int] = None,
    experiment_name: str = "stock_trading_rl",
    models_dir: Optional[str] = None,
    generate_report: bool = True
):
    """
    Run the complete RL training pipeline
    
    Args:
        timeframes: List of timeframes to train models for
        timesteps_per_timeframe: Dict mapping timeframes to training timesteps
        experiment_name: MLflow experiment name
        models_dir: Directory to save models
        generate_report: Whether to generate a report after training
    
    Returns:
        Dictionary mapping timeframes to trained model paths
    """
    if timesteps_per_timeframe is None:
        timesteps_per_timeframe = {
            "daily": 100000,
            "weekly": 80000,
            "monthly": 60000,
            "meta": 50000
        }
    
    # Define models directory
    if models_dir is None:
        models_dir = project_root / "RL" / "Models"
    else:
        models_dir = Path(models_dir)
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Dictionary to store model paths
    model_paths = {}
    
    # Track pipeline start time
    pipeline_start = time.time()
    print(f"\n===== STARTING RL TRAINING PIPELINE: {experiment_name} =====\n")
    
    # Train timeframe-specific models
    timeframe_models = [tf for tf in timeframes if tf != "meta"]
    
    for timeframe in timeframe_models:
        print(f"\n===== TRAINING {timeframe.upper()} MODEL =====\n")
        
        # Define model path
        model_path = str(models_dir / timeframe / f"{timeframe}_model.zip")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Configure training
        config = {
            "timesteps": timesteps_per_timeframe.get(timeframe, 100000),
            "eval_freq": 5000,
            "reward_type": "combined"
        }
        
        # Train the model
        model, run_id = train_timeframe_model(
            timeframe=timeframe,
            config=config,
            experiment_name=experiment_name,
            model_save_path=model_path
        )
        
        if model is not None:
            model_paths[timeframe] = model_path
            print(f"✅ {timeframe.capitalize()} model trained successfully")
            print(f"   - Saved to: {model_path}")
            print(f"   - Run ID: {run_id}")
        else:
            print(f"❌ {timeframe.capitalize()} model training failed")
    
    # Train meta-model if configured
    if "meta" in timeframes:
        print("\n===== TRAINING META-MODEL =====\n")
        
        # Define meta-model path
        meta_model_path = str(models_dir / "meta" / "meta_model.zip")
        os.makedirs(os.path.dirname(meta_model_path), exist_ok=True)
        
        # Configure meta-model training
        meta_config = {
            "timesteps": timesteps_per_timeframe.get("meta", 50000),
            "eval_freq": 2000,
            "include_model_predictions": True
        }
        
        # Get paths for available timeframe models
        daily_model_path = model_paths.get("daily", None)
        weekly_model_path = model_paths.get("weekly", None)
        monthly_model_path = model_paths.get("monthly", None)
        
        # Train meta-model
        meta_model, meta_run_id = train_meta_model(
            daily_model_path=daily_model_path,
            weekly_model_path=weekly_model_path,
            monthly_model_path=monthly_model_path,
            config=meta_config,
            experiment_name=experiment_name,
            model_save_path=meta_model_path
        )
        
        if meta_model is not None:
            model_paths["meta"] = meta_model_path
            print(f"✅ Meta-model trained successfully")
            print(f"   - Saved to: {meta_model_path}")
            print(f"   - Run ID: {meta_run_id}")
        else:
            print(f"❌ Meta-model training failed")
    
    # Calculate total training time
    total_time = time.time() - pipeline_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n===== TRAINING PIPELINE COMPLETED =====")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Generate report if requested
    if generate_report:
        print("\n===== GENERATING EXPERIMENT REPORT =====\n")
        analyzer = ExperimentAnalyzer(experiment_name=experiment_name)
        dashboard_path = analyzer.generate_experiment_dashboard()
        print(f"✅ Experiment dashboard generated: {dashboard_path}")
    
    return model_paths


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run RL training pipeline")
    
    parser.add_argument("--timeframes", type=str, default="daily,weekly,monthly,meta",
                      help="Comma-separated list of timeframes to train models for")
    parser.add_argument("--experiment", type=str, default="stock_trading_rl",
                      help="MLflow experiment name")
    parser.add_argument("--models-dir", type=str, default=None,
                      help="Directory to save trained models")
    parser.add_argument("--no-report", action="store_true",
                      help="Skip generating report after training")
    
    # Add timesteps arguments for each timeframe
    parser.add_argument("--daily-timesteps", type=int, default=100000,
                      help="Timesteps for training daily model")
    parser.add_argument("--weekly-timesteps", type=int, default=80000,
                      help="Timesteps for training weekly model")
    parser.add_argument("--monthly-timesteps", type=int, default=60000,
                      help="Timesteps for training monthly model")
    parser.add_argument("--meta-timesteps", type=int, default=50000,
                      help="Timesteps for training meta-model")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse timeframes
    timeframes = args.timeframes.split(",")
    
    # Configure timesteps
    timesteps = {
        "daily": args.daily_timesteps,
        "weekly": args.weekly_timesteps,
        "monthly": args.monthly_timesteps,
        "meta": args.meta_timesteps
    }
    
    # Run pipeline
    run_training_pipeline(
        timeframes=timeframes,
        timesteps_per_timeframe=timesteps,
        experiment_name=args.experiment,
        models_dir=args.models_dir,
        generate_report=not args.no_report
    )
