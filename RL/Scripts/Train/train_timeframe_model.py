"""
Timeframe-specific RL model training with unified MLflow tracking
Supports daily, weekly, and monthly prediction models with comprehensive monitoring
"""

import os
import sys
import traceback
import time
import json
import argparse
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Stable Baselines imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Custom imports
from RL.Utils.mlflow_manager import MLflowManager
from RL.Utils.mlflow_callback import MLflowLoggingCallback
from RL.Envs.trading_env import TradingEnv
from RL.Data.Utils.loader import load_data_from_db


def train_timeframe_model(
    timeframe: str = "1H",
    config: Dict[str, Any] = None,
    data = None,
    experiment_name: str = "stock_trading_rl",
    run_name: Optional[str] = None,
    model_save_path: Optional[str] = None,
    enhance_model_version: Optional[str] = None,
    enhancement_type: str = "continued",
    auto_find_model: bool = True,
    non_destructive: bool = True,
):
    """
    Train RL model for specific timeframe with comprehensive MLflow tracking
    
    Args:
        timeframe: Time period for model (e.g., "1H", "D", "4H")
        config: Training configuration dict
        data: Optional data to use (if None, will load from DB)
        experiment_name: MLflow experiment name
        run_name: MLflow run name (generated if None)
        model_save_path: Path to save the final model
        enhance_model_path: Path to existing model to enhance or "latest" to find latest model
        enhancement_type: Type of enhancement to apply (continued, replay, curriculum, adaptive)
        auto_find_model: Whether to automatically find latest model of specified type
        non_destructive: Whether to avoid overwriting existing models by creating unique paths
        
    Returns:
        Tuple of (trained model, run_id, enhancement_metrics)
    """
    # Default configuration
    default_config = {
        "timesteps": 100000,
        "eval_freq": 10000,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "n_steps": 1024,
        "n_epochs": 10,
        "ent_coef": 0.5,
        "reward_type": "combined",
        "normalize_observations": True,
    }
    
    timeframe_id_map = {
        "D": 7,  # Daily
        "4H": 6,  # 4-Hour
        "1H": 5,  # 1-Hour
        "30M": 4,  # 30-Minute
        "15M": 3,  # 15-Minute
        "5M": 2,  # 5-Minute
        "1M": 1   # 1-Minute
    }
    timeframe_id = timeframe_id_map.get(timeframe, 5)  # Default to 1H if not found
    
    # Override defaults with provided config
    if config is None:
        config = {}
    
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
      # Track enhancement metrics
    enhancement_metrics = {}
    base_model_run_id = None
    
    # Handle auto-find model option or when "latest" is specified
    if (auto_find_model and enhance_model_version is None) or enhance_model_version == "latest":
        # Initialize MLflow manager temporarily just to find the model
        temp_mlflow_manager = MLflowManager(
            experiment_name=experiment_name,
            timeframe=timeframe,
            model_type="base"
        )
        
        print(f"Looking for latest base model for timeframe {timeframe}...")
        # Use the updated find_latest_model with version_type parameter
        base_run_id, model_path = temp_mlflow_manager.find_latest_model(
            model_type="base", 
            version_type="latest"
        )
        
        if base_run_id and model_path:
            enhance_model_version = model_path
            base_model_run_id = base_run_id
            print(f"Found latest model: {enhance_model_version} (run_id: {base_run_id})")
            
            # Mark the previous latest model as "old" since we'll be creating a newer model
            temp_mlflow_manager.update_model_tags(
                run_id=base_run_id,
                new_tags={"version_type": "old"}
            )
            print(f"Updated previous model (run_id: {base_run_id}) to version_type='old'")
        else:
            enhance_model_version = None
            print(f"No model found for enhancement. Creating a new base model.")
    
    # Define model_type based on enhancement status
    model_type = "base"
    if enhance_model_version:
        model_type = enhancement_type if enhancement_type else "enhanced"
        
    # Define save paths with version info for non-destructive mode
    if model_save_path is None:
        if non_destructive and enhance_model_version:
            # Create versioned filename for enhanced models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_save_path = str(project_root / "RL" / "Models" / timeframe / 
                                 f"{timeframe}_{model_type}_{timestamp}.zip")
        else:
            model_save_path = str(project_root / "RL" / "Models" / timeframe / 
                                 f"{timeframe}_model.zip")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Set up logs directory
    logs_dir = project_root / "RL" / "Logs" / timeframe
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize MLflow tracking
    mlflow_manager = MLflowManager(
        experiment_name=experiment_name,
        timeframe=timeframe,
        model_type=model_type
    )
      # Prepare tags with enhanced versioning using the updated tagging system
    run_tags = {
        # Primary tags (standardized)
        "timeframe": timeframe,                   # Time period (e.g., "1H", "D", "4H")
        "model_type": model_type,                 # Either "base" or "enhanced"
        "version_type": "latest",                 # "latest" for current model, will be changed to "old" when a newer model is created
    }
    
    # Add enhancement-specific tags
    if enhance_model_version:
        run_tags.update({
            "enhancement_type": enhancement_type,
            "base_model_path": enhance_model_version
        })
        
        # Add base model run ID if we have it
        if base_model_run_id:
            run_tags["base_model_run_id"] = base_model_run_id
    
    # Start MLflow run
    run = mlflow_manager.start_run(run_name=run_name, tags=run_tags)
      # Log config parameters
    mlflow_manager.log_params(config)
    mlflow_manager.log_params({
        "timeframe": timeframe,
        "model_save_path": model_save_path,
        "model_type": model_type,
        "version_type": "latest",  # Using "latest" instead of "base"/"enhanced"
        "non_destructive": non_destructive,
        "base_model": enhance_model_version or "none"
    })
    
    try:
        # Load or use provided data
        print("Loading or preparing data...")
        
        if data is None:
            # Load data from database
            data = load_data_from_db(timeframe_id=timeframe_id)
        
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError(f"No data available for timeframe: {timeframe}")
        
        # Log dataset info
        mlflow_manager.log_dataset_info(data, "full_dataset")
        
        # Split data for training/evaluation
        print("Splitting data for training and evaluation...")
        split_idx = int(len(data) * 0.8)
        training_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        mlflow_manager.log_metrics({
            "training_samples": len(training_data),
            "evaluation_samples": len(eval_data)
        })
        
        # Create training environment
        print(f"Creating {timeframe} training environment...")
        train_env = TradingEnv(
            training_data,
            normalize_observations=config["normalize_observations"],
            reward_type=config["reward_type"],
            timeframe_id=timeframe_id
        )
        
        # Create evaluation environment
        eval_env_base = TradingEnv(
            eval_data,
            normalize_observations=config["normalize_observations"],
            reward_type=config["reward_type"],
            timeframe_id=timeframe_id,
        )
        eval_env = Monitor(eval_env_base)
        
        # Either load existing model or create new one
        if enhance_model_version:
            print(f"Loading existing model from {enhance_model_version}...")
            try:
                model = PPO.load(
                    enhance_model_version, 
                    env=train_env,
                    custom_objects={'learning_rate': config["learning_rate"]}
                )
                # Update learning rate if enhancing
                if enhancement_type == "continued":
                    model.learning_rate = config["learning_rate"]
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead.")
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=config["learning_rate"],
                    n_steps=config["n_steps"],
                    batch_size=config["batch_size"],
                    n_epochs=config["n_epochs"],
                    ent_coef=config["ent_coef"],
                    verbose=1
                )
        else:
            print("Creating new model...")
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                ent_coef=config["ent_coef"],
                verbose=1
            )
        
        # Set up TensorBoard logging
        tensorboard_path = logs_dir / "tensorboard"
        os.makedirs(tensorboard_path, exist_ok=True)
        logger = configure(str(tensorboard_path), ["tensorboard", "stdout"])
        model.set_logger(logger)
        
        # Set up unified MLflow callback
        print("Setting up unified MLflow callback with comprehensive monitoring...")
        
        unified_callback = MLflowLoggingCallback(
            mlflow_manager=mlflow_manager,
            eval_env=eval_env,
            eval_freq=config["eval_freq"],
            log_freq=min(config["eval_freq"] // 10, 500),  # Log training metrics frequently
            feature_importance_freq=config["eval_freq"] * 2,  # Calculate feature importance less frequently
            portfolio_eval_freq=config["eval_freq"],  # Portfolio evaluation at same frequency as general eval
            n_eval_episodes=3,  # Multiple episodes for better statistics
            max_eval_steps=500,  # Prevent infinite loops
            risk_free_rate=0.02,  # 2% annual risk-free rate for Sharpe calculation
            save_plots=True,  # Enable comprehensive plot generation
            save_model_checkpoints=True,  # Enable automatic model checkpointing
            verbose=1,  # Normal verbosity
            timeframe=timeframe
        )
        
        # Use only the unified callback (includes all functionality)
        callbacks = CallbackList([unified_callback])
        
        # Log training start
        start_time = time.time()
        mlflow_manager.log_metrics({"training_start_time": start_time})
        
        # Train model
        print(f"\n🚀 Starting {timeframe} model training for {config['timesteps']} timesteps...")
        print(f"📊 Unified callback will monitor:")
        print(f"   • Training metrics every {min(config['eval_freq'] // 10, 500)} steps")
        print(f"   • Model evaluation every {config['eval_freq']} steps") 
        print(f"   • Portfolio performance every {config['eval_freq']} steps")
        print(f"   • Feature importance every {config['eval_freq'] * 2} steps")
        print(f"   • Comprehensive plots and checkpoints")
        
        model.learn(
            total_timesteps=config["timesteps"],
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(enhancement_type != "continued")
        )
        
        # Log training end and duration
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        mlflow_manager.log_metrics({
            "training_end_time": end_time,
            "training_duration_minutes": duration_minutes
        })
          # Save final model
        print(f"💾 Saving final {timeframe} model...")
        model.save(model_save_path)
        
        # Log model artifact
        mlflow_manager.log_sb3_model(model, f"{timeframe}_model")
        
        # Log additional artifacts
        mlflow_manager.log_directory_artifacts(str(logs_dir), f"{timeframe}_logs")
        
        # Handle model enhancement comparison and registration
        if enhance_model_version and base_model_run_id:
            print(f"📊 Generating enhancement metrics and comparisons...")
            
            # Log enhancement metrics comparing to base model
            enhancement_metrics = mlflow_manager.log_enhancement_metrics(base_model_run_id)
            
            # Generate enhancement report
            enhancement_summary = {
                "base_model_run_id": base_model_run_id,
                "enhancement_type": enhancement_type,
                "metrics": enhancement_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Write enhancement summary to temp file and log
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(enhancement_summary, f, indent=2)
                enhancement_path = f.name
                
            mlflow_manager.log_file_artifact(enhancement_path, f"{timeframe}_enhancement_report.json")
            os.unlink(enhancement_path)
            
            # Register the enhanced model with proper versioning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"{timeframe}_trading_model"
            version_tags = {
                "base_model_run_id": base_model_run_id,
                "enhancement_type": enhancement_type,
                "timeframe": timeframe,
                "enhancement_timestamp": timestamp
            }
            
            # Register model in MLflow Registry
            model_version = mlflow_manager.register_model_version(
                model_path=model_save_path,
                model_name=model_name,
                base_run_id=base_model_run_id,
                stage="Development",
                tags=version_tags
            )
            
            print(f"📝 Registered enhanced model as {model_name} version {model_version}")
            print(f"📈 Enhancement metrics: {', '.join([f'{k}: {v:.2f}%' for k, v in enhancement_metrics.items()][:3])}")
          # Create and log performance summary
        performance_summary = {
            "timeframe": timeframe,
            "training_samples": len(training_data),
            "evaluation_samples": len(eval_data),
            "training_duration_minutes": duration_minutes,
            "model_path": model_save_path,
            "run_id": run.info.run_id,
            "unified_callback_features": [
                "training_metrics_logging",
                "model_evaluation",
                "portfolio_performance_tracking",
                "feature_importance_analysis",
                "comprehensive_plotting",
                "automatic_checkpointing"
            ]
        }
        
        # Add enhancement information to summary if applicable
        if enhance_model_version and base_model_run_id:
            performance_summary.update({
                "enhancement": {
                    "base_model_run_id": base_model_run_id,
                    "enhancement_type": enhancement_type,
                    "has_comparison_metrics": len(enhancement_metrics) > 0,
                    "key_improvements": dict(list(enhancement_metrics.items())[:5]) if enhancement_metrics else {}
                }
            })
        
        # Write summary to temp file and log
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(performance_summary, f, indent=2)
            temp_path = f.name
            
        mlflow_manager.log_file_artifact(temp_path, f"{timeframe}_summary.json")
        os.unlink(temp_path)
        
        print(f"✅ {timeframe.capitalize()} model training completed successfully!")
        print(f"📈 MLflow Run ID: {run.info.run_id}")
        print(f"💾 Model saved to: {model_save_path}")
        print(f"📊 Comprehensive monitoring data logged to MLflow")
        
        # Display enhancement information if applicable
        if enhance_model_version and base_model_run_id and enhancement_metrics:
            print("\n🚀 Model Enhancement Results:")
            # Get the top 3 improvement metrics
            top_metrics = sorted(
                [(k.replace('improvement_', ''), v) for k, v in enhancement_metrics.items()],
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:3]
            
            for metric, value in top_metrics:
                color = '🟢' if value > 0 else '🔴'
                print(f"{color} {metric}: {value:.2f}% change")
                
            print("\nView the full enhancement report and comparison plots in MLflow artifacts.")
        
        return model, run.info.run_id, enhancement_metrics if enhance_model_version else {}
        
    except Exception as e:
        print(f"❌ Error during {timeframe} model training: {e}")
        traceback.print_exc()
        
        # Log error
        mlflow_manager.log_params({
            "error": str(e),
            "error_type": type(e).__name__
        })
        
        raise
        
    finally:
        # Always end the MLflow run
        mlflow_manager.end_run()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Train timeframe-specific RL trading models with unified MLflow monitoring")
    
    parser.add_argument("--timeframe", type=str, default="1H", 
                       choices=["D", "4H", "1H", "30M", "15M", "5M", "1M"],
                       help="Timeframe for the model")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Number of training timesteps")
    parser.add_argument("--experiment", type=str, default="stock_trading_rl",
                       help="MLflow experiment name")
    parser.add_argument("--enhance", type=str,
                       help="Path to existing model to enhance, or 'latest' to find latest model")
    parser.add_argument("--enhance-type", type=str, default="continued",
                       choices=["continued", "replay", "curriculum", "adaptive"],
                       help="Type of enhancement to apply")
    parser.add_argument("--save-path", type=str,
                       help="Path to save the trained model")
    parser.add_argument("--auto-find", action="store_true",
                       help="Automatically find the latest model to enhance")
    parser.add_argument("--non-destructive", action="store_true", default=True,
                       help="Create unique model files instead of overwriting")
    parser.add_argument("--register-model", action="store_true",
                       help="Register the model in MLflow model registry")
    
    args = parser.parse_args()
    
    # Prepare config
    config = {
        "timesteps": args.timesteps,
        "eval_freq": min(10000, args.timesteps // 20),
        "reward_type": "combined"
    }
    
    # Run training
    try:
        print("🎯 Starting training with unified MLflow callback system...")
        print("This includes comprehensive monitoring of:")
        print("  • Training progress and metrics")
        print("  • Model evaluation and performance")
        print("  • Portfolio tracking and financial metrics")
        print("  • Feature importance analysis")
        print("  • Automatic plotting and visualization")
        print("  • Model checkpointing and artifact management")
        model, run_id, enhancement_metrics = train_timeframe_model(
        timeframe=args.timeframe,
        config=config,
        experiment_name=args.experiment,
        model_save_path=args.save_path,
        enhance_model_version=args.enhance,
        enhancement_type=args.enhance_type,
        auto_find_model=True if args.enhance == "latest" else False,
        non_destructive=True
    )
        
        print("\n📈 To view comprehensive training results:")
        print("1. Run: mlflow ui")
        print("2. Open: http://localhost:5000")
        print(f"3. Navigate to experiment: {args.experiment}")
        print(f"4. Find run ID: {run_id}")
        print("5. Explore artifacts: plots, models, histories, and summaries")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
