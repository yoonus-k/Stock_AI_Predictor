"""
Timeframe-specific RL model training with unified MLflow tracking
Supports daily, weekly, and monthly prediction models with comprehensive monitoring
"""
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import os
import sys
import traceback
import time
import json
import argparse
import tempfile
import shutil
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
from stable_baselines3 import DQN, A2C, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Custom imports
import mlflow.artifacts
from RL.Mlflow.mlflow_manager import MLflowManager
from RL.Mlflow.mlflow_callback import MLflowLoggingCallback
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
    model_type: str = "base",  # Added parameter to specify which model type to enhance
    auto_find_model: bool = True,
    non_destructive: bool = True,
    stock_id: Optional[str] = 1,
    start_date: Optional[str] ="2024-01-01",
    end_date: Optional[str] = "2025-01-01"
):
    """
    Train RL model for specific timeframe with comprehensive MLflow tracking
    #test
    Args:
        timeframe: Time period for model (e.g., "1H", "D", "4H")
        config: Training configuration dict
        data: Optional data to use (if None, will load from DB)
        experiment_name: MLflow experiment name
        run_name: MLflow run name (generated if None)
        model_save_path: Path to save the final model (if None, will use temporary storage)
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
        "eval_freq": 20000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 256,
        "n_epochs": 10,
        "ent_coef": 0.1,
        "reward_type": "combined",
        "normalize_observations": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,  # Use default value if not specified
        "normalize_advantage": True,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,  # Resample noise every 8 steps (tune this)
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
    model_path = None
    
    # Handle auto-find model option or when "latest" is specified
    if auto_find_model and enhance_model_version is None or enhance_model_version in ["latest", "best"]:
        # Initialize MLflow manager
        temp_mlflow_manager = MLflowManager(
            experiment_name=experiment_name,
            timeframe=timeframe,
            model_type=model_type  # Use the model type specified in parameters
        )
        
        # Determine which finding method to use
        if enhance_model_version == "best":
            print(f"Looking for best performing {model_type} model for timeframe {timeframe}...")
            metric = config['enhance_metric'] if 'enhance_metric' in config else "evaluation/best_mean_reward"
            print(f"Using metric '{metric}' to find the best model")
            base_run_id, model_path = temp_mlflow_manager.find_best_model(
                model_type=model_type,  # Use the model type specified in parameters
                metric=metric
            )
        else:
            print(f"Looking for latest {model_type} model for timeframe {timeframe}...")
            base_run_id, model_path = temp_mlflow_manager.find_latest_model(
                model_type=model_type,  # Use the model type specified in parameters
                version_type="latest"
            )
        
        # Process the found model (applies to both "best" and "latest")
        if base_run_id and model_path:
            # Ensure model path uses forward slashes
            model_path = model_path.replace("\\", "/")
            base_model_run_id = base_run_id
            
            # Print appropriate message
            if enhance_model_version == "best":
                print(f"Found best model using metric '{metric}': {model_path} (run_id: {base_run_id})")
            else:
                print(f"Found latest model: {model_path} (run_id: {base_run_id})")
            
            # Mark the found model as "old" since we'll create a newer one
            temp_mlflow_manager.update_model_tags(
                run_id=base_run_id,
                new_tags={"version_type": "old"}
            )
            print(f"Updated previous model (run_id: {base_run_id}) to version_type='old'")
        else:
            # No model of specified type found, try base model as fallback if not already trying base
            if model_type != "base":
                print(f"No model of type '{model_type}' found. Trying to find base model as fallback...")
                temp_mlflow_manager = MLflowManager(
                    experiment_name=experiment_name,
                    timeframe=timeframe,
                    model_type="base"
                )
                
                # Determine which finding method to use
                if enhance_model_version == "best":
                    print(f"Looking for best performing base model for timeframe {timeframe}...")
                    metric = config.get("enhance_metric", "evaluation/best_mean_reward")
                    print(f"Using metric '{metric}' to find the best model")
                    base_run_id, model_path = temp_mlflow_manager.find_best_model(
                        model_type="base",
                        metric=metric
                    )
                else:
                    print(f"Looking for latest base model for timeframe {timeframe}...")
                    base_run_id, model_path = temp_mlflow_manager.find_latest_model(
                        model_type="base", 
                        version_type="latest"
                    )
                    
                # Process fallback model if found
                if base_run_id and model_path:
                    print(f"Found base model to use as fallback for enhancement")
                    model_path = model_path.replace("\\", "/")
                    base_model_run_id = base_run_id
                    
                    # Mark the found model as "old" since we'll create a newer one
                    temp_mlflow_manager.update_model_tags(
                        run_id=base_run_id,
                        new_tags={"version_type": "old"}
                    )
                    print(f"Updated previous model (run_id: {base_run_id}) to version_type='old'")
                else:
                    # No model found through automatic methods
                    model_path = None
                    print(f"No model found for enhancement. Creating a new base model.")
            else:
                # No model found through automatic methods
                model_path = None
                print(f"No model found for enhancement. Creating a new base model.")
     
    # Use a temporary directory for any intermediate files instead of local storage
    # Model will be saved directly to MLflow, not to a local path
    temp_dir = tempfile.mkdtemp(prefix=f"stock_ai_{timeframe}_")
    
    # If model_save_path is provided, use it (for backward compatibility)
    # Otherwise use a temporary path that will be deleted after MLflow logging
    if model_save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_filename = f"{timeframe}_{model_type}_{timestamp}.zip"
        model_save_path = os.path.join(temp_dir, model_filename)
        
    # Set up temporary logs directory within the temp directory
    logs_dir = Path(os.path.join(temp_dir, "logs", timeframe))
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
    if model_path is not None:
        run_tags.update({
            "enhancement_type": enhancement_type,
            "base_model_path": model_path
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
        "base_model": model_path or "none"
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
        max_eval_episode_length = len(eval_data) - 1  # Ensure at least one step for evaluation
        
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
            timeframe_id=timeframe_id ,
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date        
            )
        
        # Create evaluation environment
        eval_env = TradingEnv(
            eval_data,
            normalize_observations=config["normalize_observations"],
            reward_type=config["reward_type"],
            timeframe_id=timeframe_id,
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        
        # Either load existing model or create new one
        if model_path and model_path.startswith("runs:"):
        
            print(f"Using MLflow model: {model_path}")
            try:
                # Extract run_id and artifact path from the runs:/ URI
                # Format is typically: runs:/<run_id>/<artifact_path>
                parts = model_path.replace("\\", "/").split('/', 2)
                if len(parts) >= 3:
                    run_id = parts[1]
                    artifact_path = parts[2]
                    
                    print(f"Downloading artifact from run {run_id}, path: {artifact_path}")
                    
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()
                    temp_model_path = os.path.join(temp_dir, os.path.basename(artifact_path))
                    
                    # Download the artifact
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=artifact_path,
                        dst_path=temp_dir
                    )
                    print(f"Downloaded model to: {local_path}")
                    
                    # Update the path to use the local file
                    model_path = local_path
            except Exception as e:
                print(f"Error downloading artifact: {e}")
                print("Will attempt to use the original path.")
        
            print(f"Loading existing model from {model_path}...")
            
            try:
                # Keep track of temp directory if we created one
                temp_dir_to_cleanup = None
                if 'temp_dir' in locals():
                    temp_dir_to_cleanup = temp_dir
                
                model = PPO.load(
                    model_path, 
                    env=train_env,
                    #custom_objects={'learning_rate': config["learning_rate"]}
                )
                
                 # Then update model parameters based on config
                print("Updating model parameters based on config...")
                
                # Define mapping between config keys and model attributes
                param_mapping = {
                                "n_epochs": "n_epochs",
                                "n_steps": "n_steps",
                                "batch_size": "batch_size",
                                "learning_rate": "learning_rate",
                                "ent_coef": "ent_coef",
                                "gamma": "gamma",
                                "gae_lambda": "gae_lambda",
                                "vf_coef": "vf_coef",
                                "max_grad_norm": "max_grad_norm",
                                "normalize_advantage": "normalize_advantage"
                            }
                
                
                 # Apply each parameter from config to the model
                for config_key, model_attr in param_mapping.items():
                    if config_key in config:
                        try:
                            original_value = getattr(model, model_attr, None)
                            setattr(model, model_attr, config[config_key])
                            new_value = getattr(model, model_attr)
                            
                            # Also log to MLflow for tracking
                            if original_value != new_value:
                                 # Log the parameter change
                                print(f"  • Updated {model_attr}: {original_value} → {new_value}")
                                mlflow_manager.log_params({
                                    f"updated_{model_attr}": f"{original_value} -> {new_value}"
                                })
                        except Exception as e:
                            print(f"Warning: Could not update {model_attr}: {e}")
                
                # Track previous timesteps when enhancing
                previous_timesteps = 0
                if enhancement_type == "continued":
                    # Try to get previous timesteps from base model run
                    if base_model_run_id:
                        try:
                            # Query MLflow for the previous run's metrics
                            client = mlflow.tracking.MlflowClient()
                            previous_run = client.get_run(base_model_run_id)
                            
                            # Check if total_timesteps was logged
                            if "total_timesteps" in previous_run.data.metrics:
                                previous_timesteps = int(previous_run.data.metrics["total_timesteps"])
                                print(f"Continuing training from previous {previous_timesteps} timesteps")
                            else:
                                # Try to get num_timesteps which might be available
                                metrics_dict = previous_run.data.metrics
                                timestep_keys = [k for k in metrics_dict.keys() if 'timesteps' in k.lower()]
                                if timestep_keys:
                                    # Use the highest timestep value found
                                    highest_timestep = max([int(metrics_dict[k]) for k in timestep_keys])
                                    previous_timesteps = highest_timestep
                                    print(f"Continuing training from estimated {previous_timesteps} timesteps")
                        except Exception as e:
                            print(f"Could not retrieve previous timesteps: {e}")
                            print("Will track timesteps from this run only.")
                    
                    # Log the starting point
                    mlflow_manager.log_metrics({"previous_timesteps": previous_timesteps})
                    
                # Cleanup temp directory if we created one
                if temp_dir_to_cleanup:
                    try:
                        shutil.rmtree(temp_dir_to_cleanup)
                        print(f"Cleaned up temporary directory: {temp_dir_to_cleanup}")
                    except Exception as cleanup_err:
                        print(f"Note: Failed to clean up temp directory: {cleanup_err}")
                
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
            # set previous_timesteps to 0 for new models
            previous_timesteps = 0
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
        # Set up TensorBoard logging in temp directory (will be captured by MLflow)
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
            log_freq=min(config["eval_freq"] // 10, 1000),  # Log training metrics frequently
            feature_importance_freq=config["eval_freq"] * 2,  # Calculate feature importance less frequently
            model_eval_freq=config["eval_freq"],  # Model evaluation at same frequency as general eval
            portfolio_eval_freq=config["eval_freq"],  # Portfolio evaluation at same frequency as general eval
            n_eval_episodes=3,  # Multiple episodes for better statistics
            max_eval_steps=max_eval_episode_length,  # Prevent infinite loops
            risk_free_rate=0.02,  # 2% annual risk-free rate for Sharpe calculation
            save_plots=True,  # Enable comprehensive plot generation
            save_model_checkpoints=True,  # Enable automatic model checkpointing
            verbose=1,  # Normal verbosity
            timeframe=timeframe,
            previous_timesteps=previous_timesteps  # Add previous timesteps for continuous tracking
        )
        
        # Use only the unified callback (includes all functionality)
        callbacks = CallbackList([unified_callback])
        
        # Log training start
        start_time = time.time()
        mlflow_manager.log_metrics({"training_start_time": start_time})        # Track total cumulative timesteps for proper tracking
       
        # Train model
        print(f"\n🚀 Starting {timeframe} model training for {config['timesteps']} timesteps...")
        if previous_timesteps > 0:
            print(f"💡 Continuing from previous {previous_timesteps} timesteps (total will be {previous_timesteps + config['timesteps']})")
        
        print(f"📊 Unified callback will monitor:")
        print(f"   • Training metrics every {min(config['eval_freq'] // 10, 1000)} steps")
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
        
        # Calculate total timesteps (original + new)
        total_timesteps = previous_timesteps + config["timesteps"]
        
        mlflow_manager.log_metrics({
            "training_end_time": end_time,
            "training_duration_minutes": duration_minutes,
            "current_run_timesteps": config["timesteps"],
            "total_timesteps": total_timesteps
        })
        
        # Modified code:
        print(f"🔄 Using MLflow callback to save model checkpoints...")
        # Note: Model saving is now handled entirely by the MLflow callback's _save_model_checkpoint method
        # No explicit model saving here anymore
                
        # Log additional artifacts from temp directory
        mlflow_manager.log_directory_artifacts(str(logs_dir), f"{timeframe}_logs")
        
        # Handle model enhancement comparison and registration
        if model_path and base_model_run_id:
            print(f"📊 Generating enhancement metrics and comparisons...")
            
            # Log enhancement metrics comparing to base model
            enhancement_metrics = mlflow_manager.log_enhancement_metrics(base_model_run_id)
              # Generate enhancement report
            enhancement_summary = {
                "base_model_run_id": base_model_run_id,
                "enhancement_type": enhancement_type,
                "metrics": enhancement_metrics,
                "timestamp": datetime.now().isoformat(),
                "previous_timesteps": previous_timesteps,
                "new_timesteps": config["timesteps"],
                "total_timesteps": total_timesteps
            }
            
            # Write enhancement summary to temp file and log
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(enhancement_summary, f, indent=2)
                enhancement_path = f.name
                
            mlflow_manager.log_file_artifact(enhancement_path, f"{timeframe}_enhancement_report.json")
            os.unlink(enhancement_path)              # Store model reference information
          
            print(f"📝 Model saved and logged to MLflow (use model management CLI to register)")
            print(f"📈 Enhancement metrics: {', '.join([f'{k}: {v:.2f}%' for k, v in enhancement_metrics.items()][:3])}")
        
        
        print(f"✅ {timeframe.capitalize()} model training completed successfully!")
        print(f"📈 MLflow Run ID: {run.info.run_id}")
        print(f"🔄 Model stored exclusively in MLflow (not saved locally)")
        print(f"📊 Comprehensive monitoring data logged to MLflow")
        
        # Display enhancement information if applicable
        if model_path and base_model_run_id and enhancement_metrics:
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
        
        return model, run.info.run_id, enhancement_metrics if model_path else {}
        
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
        
        # Clean up temporary directory
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_err:
            print(f"Note: Failed to clean up temp directory: {cleanup_err}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Train timeframe-specific RL trading models with unified MLflow monitoring")
    parser.add_argument("--timeframe", type=str, default="1H", 
                       choices=["D", "4H", "1H", "30M", "15M", "5M", "1M"],
                       help="Timeframe for the model")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Number of training timesteps")
    parser.add_argument("--experiment", type=str, default="stock_trading_rl_new_env",
                        help="MLflow experiment name")
    parser.add_argument("--enhance", type=str, default="latest",
                        help="Path to existing model to enhance, 'latest' to find latest model, or 'best' to find best model")
    parser.add_argument("--enhance-metric", type=str, default="evaluation/best_mean_reward",
                        help="Metric to use for finding the best model (only used when --enhance=best)")
    parser.add_argument("--enhance-type", type=str, default="continued",
                        choices=["continued", "replay", "curriculum", "adaptive"],
                        help="Type of enhancement to apply")
    parser.add_argument("---model-type", type=str, default="continued",
                        help="Type of model to enhance (e.g., 'base', 'continued', etc.)")

    parser.add_argument("--save-path", type=str,
                        help="Optional temporary path for model (final model is stored only in MLflow)")
    parser.add_argument("--auto-find", action="store_true",
                        help="Automatically find the latest model to enhance")
    parser.add_argument("--non-destructive", action="store_true", default=True,
                        help="Create unique model files instead of overwriting")
    
    args = parser.parse_args()
    
    # Prepare configs
    config = {
        "timesteps": args.timesteps,
        "eval_freq": 20000,
        "reward_type": "combined",
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
        
        # Add enhance metric to config
        if args.enhance == "best":
            config["enhance_metric"] = args.enhance_metric
            
        model, run_id, enhancement_metrics = train_timeframe_model(
            timeframe=args.timeframe,
            config=config,
            experiment_name=args.experiment,
            model_save_path=args.save_path,
            enhance_model_version=args.enhance,
            enhancement_type=args.enhance_type,
            model_type=args.model_type,
            auto_find_model=True if args.enhance in ["latest", "best"] else False,
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