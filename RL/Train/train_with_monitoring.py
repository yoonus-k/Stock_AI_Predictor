# Suppress TensorFlow oneDNN optimization warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import os
import json
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Envs.trading_env import TradingEnv
from RL.Envs.Components.callbacks import FeatureImportanceCallback , PortfolioTrackingCallback,EvalCallback
# TupleActionWrapper is no longer needed as we use Box action space directly
from RL.Data.Utils.loader import load_data_from_db


def train_rl_model(
    db_path="RL/Data/samples.db", 
    save_path="RL/Models/Experiments", 
    timesteps=200000,     # Reduced from 200000 for testing
    eval_freq=10000,     # Increased from 1000 to avoid overlap with other callbacks
    # checkpoint_freq=2500,  # Reduced from 10000 for testing
    log_path=None,
    tensorboard=True,
    progress_bar=True,  # Enable progress bar
):
    """
    Train the RL model with data from database with enhanced monitoring and logging
    
    Parameters:
        db_path: Path to database file. If None, will search in common locations.
        save_path: Where to save the trained model
        timesteps: Number of training timesteps
        eval_freq: How often to evaluate the model (in timesteps)
        checkpoint_freq: How often to save model checkpoints (in timesteps)
        log_path: Directory to save logs and checkpoints
        tensorboard: Whether to use TensorBoard logging
        progress_bar: Whether to show progress bar during training
        use_minimal_callbacks: If True, use only essential callbacks (for debugging)
    """
    print("\n========== TRADING AGENT TRAINING WITH IMPROVED CALLBACKS ==========\n")
    
    # Setup directories
    if log_path is None:
        log_path = Path(__file__).parent.parent.parent / "Logs"
    else:
        log_path = Path(log_path)
    
    model_dir = Path(os.path.dirname(save_path))
    tensorboard_path = log_path / "tensorboard"
    # checkpoint_path = log_path / "checkpoints"
    
    # Create directories
    log_path.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_path.mkdir(exist_ok=True)
    # checkpoint_path.mkdir(exist_ok=True)
    
    print(f"Logs will be saved to: {log_path}")
    # print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Final model will be saved to: {save_path}")
    
    # Load dataset from database
    print("\nLoading data from database...")
    rl_dataset = load_data_from_db()
    
    if rl_dataset.empty:
        print("❌ ERROR: No data found. Please run data_exploration.ipynb to generate samples.db first.")
        return
    
    # Print dataset info for verification
    print(f"✅ Loaded {len(rl_dataset)} records from database")
    print(f"Dataset columns: {rl_dataset.columns.tolist()}")
    print(f"Dataset sample:\n{rl_dataset.head(1).T}")
    
    # Split into training and evaluation sets
    split_idx = int(len(rl_dataset) * 0.8)
    training_data = rl_dataset[:split_idx]
    eval_data = rl_dataset[split_idx:]
    
    # take only 100 records for the training to test functionality
    #training_data = training_data.head(100)
    
    print(f"\nTraining data size: {len(training_data)} records")
    print(f"Evaluation data size: {len(eval_data)} records")
    
    if len(training_data) == 0 or len(eval_data) == 0:
        print("❌ ERROR: Not enough data for training and evaluation.")
        return
    
    # Create environments
    print("\nCreating environments...")
    # Create base training env and wrap it to convert tuple action space to box action space    # Create training env (no wrapper needed as it uses Box action space directly)
    train_env = TradingEnv(
        training_data, 
        normalize_observations=True,
        reward_type='combined'  # Use a basic reward type for initial checks
    )
    
    # Create evaluation env (no action wrapper needed, just Monitor for stats)
    eval_env_base = TradingEnv(
        eval_data,
        normalize_observations=True,
    )
    # Wrap with Monitor for stats tracking
    eval_env = Monitor(eval_env_base)
    
    # Print action space info for debugging
    print(f"Action space: {train_env.action_space}")
    
    # Setup TensorBoard logging
    if tensorboard:
        print("\nSetting up TensorBoard logging...")
        logger = configure(str(tensorboard_path), ["tensorboard", "stdout"])
    else:
        logger = None
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    
    
    
    # Full callbacks setup with improved versions
    print("Using full callbacks setup with safety improvements")
    
    # Create feature importance callback with safety improvements
    feature_callback = FeatureImportanceCallback(
        eval_env=eval_env,
        log_path=log_path,
        max_saves=3,     # Limit disk writes
        verbose=0        # Show progress logs
    )
    
    # Create portfolio tracking callback with safety improvements
    portfolio_callback = PortfolioTrackingCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=eval_freq ,  # Offset by 500 steps to avoid overlap with eval callback
        max_eval_steps=500,  # Reduced from 500 to prevent long loops
        n_eval_episodes=1,   # Just 1 episode for evaluation
        verbose=1           # Show progress logs
    )
    
    # Start with essential callbacks
    active_callbacks = [ feature_callback, portfolio_callback]
    
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(log_path),
        eval_freq=eval_freq,
        deterministic=False,
        render=False,
        n_eval_episodes=3,  # Reduced from 3 for speed
        verbose=1
    )
    active_callbacks.insert(0, eval_callback)
        
    # Combine all callbacks
    callbacks = CallbackList(active_callbacks)    # Initialize the model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=0.0003,  # Increased learning rate for faster adaptation
        ent_coef=0.5,  # Higher entropy encourages more exploration
        n_steps=1024,     
        batch_size=256,   
        n_epochs=10,      
        gae_lambda=0.95,  
        tensorboard_log=str(tensorboard_path),
        clip_range=0.2,         # Increase clip range for more aggressive updates
        vf_coef=0.5,           # Value function coefficient
    )
    
    # Set custom logger if TensorBoard is enabled
    if logger is not None:
        model.set_logger(logger)
    
    # Train the model    
    print("\n" + "="*50)
    print(f"Starting training for {timesteps} timesteps...")
    print("="*50 + "\n")
    
    # Train the model with improved error handling
    try:
        print("\nStarting training with safe callbacks...")
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=progress_bar)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()
        print("\nAttempting to save partial model...")
    
    # Save the final model
    try:
        model.save(os.join(save_path,"rl_model.zip"))
        print(f"\n✅ Final model saved to {save_path}")
        
        # Save training metadata
        metadata = {
            "training_data_size": len(training_data),
            "eval_data_size": len(eval_data),
            "timesteps": timesteps,
            "eval_freq": eval_freq,
            # "checkpoint_freq": checkpoint_freq,
            "feature_count": train_env.observation_space.shape[0],
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(f"{os.path.splitext(save_path)[0]}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("\nTraining complete!")
    print("\nAvailable monitoring tools:")
    print(f"1. Launch unified monitoring dashboard: python Scripts/monitoring_dashboard.py --model={save_path} --log-dir={log_path}")
    print(f"2. View TensorBoard logs: tensorboard --logdir={tensorboard_path}")
    
    return model


def enhance_existing_model(
    model_path,
    new_data=None,
    enhancement_type="continued",
    timesteps=100000,
    save_path=None,
    log_path=None,
    tensorboard=True,
    progress_bar=True,
    **kwargs
):
    """
    Enhance an existing model using various advanced techniques
    
    Parameters:
        model_path: Path to the existing model
        new_data: New training data (if None, will use original env data)
        enhancement_type: Type of enhancement ('continued', 'replay', 'curriculum', 
                          'adaptive', 'distillation')
        timesteps: Training timesteps
        save_path: Where to save the enhanced model
        log_path: Directory to save logs
        tensorboard: Whether to use TensorBoard logging
        progress_bar: Whether to show progress bar during training
        **kwargs: Additional parameters specific to each enhancement type
    
    Returns:
        Enhanced model
    """
    print("\n========== ENHANCING EXISTING TRADING AGENT MODEL ==========\n")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Setup directories
    if log_path is None:
        log_path = Path(__file__).parent.parent.parent / "Logs" / f"enhanced_{enhancement_type}"
    else:
        log_path = Path(log_path)
    
    # Default save path using original model name + enhancement type
    if save_path is None:
        original_name = os.path.splitext(os.path.basename(model_path))[0]
        save_dir = os.path.dirname(model_path)
        save_path = os.path.join(save_dir, f"{original_name}_{enhancement_type}")
    
    model_dir = Path(os.path.dirname(save_path))
    tensorboard_path = log_path / "tensorboard"
    
    # Create directories
    log_path.mkdir(exist_ok=True, parents=True)
    #model_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_path.mkdir(exist_ok=True)
    
    print(f"Enhancement type: {enhancement_type}")
    print(f"Logs will be saved to: {log_path}")
    print(f"Enhanced model will be saved to: {save_path}")
    
    # Load the existing model
    print(f"\nLoading existing model from {model_path}...")
    model = PPO.load(model_path, custom_objects={'learning_rate': 0.0003})
    
    # If no new data provided, try to use data from database
    if new_data is None:
        print("\nNo data provided. Loading data from database...")
        rl_dataset = load_data_from_db()
        if rl_dataset.empty:
            raise ValueError("No data available for model enhancement")
            
        # Split into training and evaluation sets
        split_idx = int(len(rl_dataset) * 0.8)
        new_data = rl_dataset[:split_idx]
        eval_data = rl_dataset[split_idx:]
    else:
        # Split provided data for evaluation
        split_idx = int(len(new_data) * 0.8)
        train_part = new_data[:split_idx]
        eval_data = new_data[split_idx:]
        new_data = train_part
    
    print(f"Training data size: {len(new_data)} records")
    print(f"Evaluation data size: {len(eval_data)} records")
    
    # Create environments for the new data
    train_env = TradingEnv(
        new_data, 
        normalize_observations=True,
        reward_type=kwargs.get("reward_type", "combined")  # Default to combined reward type
    )
    
    # Create evaluation environment
    eval_env_base = TradingEnv(
        eval_data,
        normalize_observations=True,
    )
    eval_env = Monitor(eval_env_base)
    
    # Update model's environment
    model.set_env(train_env)
    
    # Setup TensorBoard logging
    if tensorboard:
        print("\nSetting up TensorBoard logging...")
        logger = configure(str(tensorboard_path), ["tensorboard", "stdout"])
        model.set_logger(logger)
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    
    # Feature importance callback
    feature_callback = FeatureImportanceCallback(
        eval_env=eval_env,
        log_path=log_path,
        max_saves=3,
        verbose=1
    )
    
    # Portfolio tracking callback
    portfolio_callback = PortfolioTrackingCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=10000,
        max_eval_steps=500,
        n_eval_episodes=1,
        verbose=1
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(log_path),
        eval_freq=10000,
        deterministic=False,
        render=False,
        n_eval_episodes=2,
        verbose=1
    )
    
    # Combine all callbacks
    callbacks = CallbackList([eval_callback, feature_callback, portfolio_callback])
    
    # Apply the selected enhancement technique
    if enhancement_type == "continued":
        print("\nApplying continued training enhancement...")
        
        # Optionally adjust learning rate for fine-tuning (typically lower)
        learning_rate = kwargs.get("learning_rate", model.learning_rate )  # Default: 5e-4
        print(f"Adjusting learning rate from {model.learning_rate} to {learning_rate}")
        model.learning_rate = learning_rate
        
    elif enhancement_type == "replay":
        print("\nApplying experience replay enhancement...")
        
        # Get replay buffer data
        replay_buffer_path = kwargs.get("replay_buffer_path")
        if replay_buffer_path is None:
            raise ValueError("replay_buffer_path must be provided for replay enhancement")
            
        try:
            print(f"Loading replay buffer from {replay_buffer_path}...")
            replay_buffer = pd.read_pickle(replay_buffer_path)
            
            # Create combined dataset with replay ratio
            replay_ratio = kwargs.get("replay_ratio", 0.3)
            replay_size = int(len(new_data) * replay_ratio / (1 - replay_ratio))
            replay_sample = replay_buffer.sample(min(replay_size, len(replay_buffer)))
            
            print(f"Adding {len(replay_sample)} replay samples to {len(new_data)} new samples")
            combined_data = pd.concat([new_data, replay_sample]).reset_index(drop=True)
            
            # Create new environment with combined data
            train_env = TradingEnv(
                combined_data, 
                normalize_observations=True,
            )
            
            # Update model's environment
            model.set_env(train_env)
        except Exception as e:
            print(f"Error setting up replay buffer: {e}")
            print("Falling back to continued training")
        
    elif enhancement_type == "curriculum":
        print("\nApplying curriculum learning enhancement...")
        
        # Get difficulty stages
        data_stages = kwargs.get("data_stages")
        if data_stages is None:
            # Create default difficulty stages based on volatility
            print("No data stages provided, creating default stages based on volatility...")
            
            # Measure volatility using ATR ratio or similar metric
            if 'atr_ratio' in new_data.columns:
                volatility_metric = 'atr_ratio'
            else:
                # Calculate approximate volatility if not available
                new_data['temp_volatility'] = np.abs(new_data['max_gain'] - new_data['max_drawdown'])
                volatility_metric = 'temp_volatility'
            
            # Sort by volatility and split into stages
            sorted_data = new_data.sort_values(by=volatility_metric).reset_index(drop=True)
            stage_size = len(sorted_data) // 3
            
            data_stages = [
                sorted_data[:stage_size],                      # Low volatility
                sorted_data[stage_size:2*stage_size],          # Medium volatility
                sorted_data[2*stage_size:]                     # High volatility
            ]
            print(f"Created {len(data_stages)} difficulty stages with {stage_size} samples each")
        
        # Calculate training time per stage
        timesteps_per_stage = timesteps // len(data_stages)
        
        # Train progressively through stages
        print("\n" + "="*50)
        print(f"Starting curriculum training through {len(data_stages)} difficulty stages...")
        print("="*50 + "\n")
        
        for i, stage_data in enumerate(data_stages):
            print(f"\nTraining on difficulty level {i+1}/{len(data_stages)}...")
            print(f"Stage data size: {len(stage_data)} records")
            
            # Create environment for this difficulty stage
            stage_env = TradingEnv(
                stage_data,
                normalize_observations=True,
            )
            
            # Update model's environment
            model.set_env(stage_env)
            
            # Train on this stage
            model.learn(
                total_timesteps=timesteps_per_stage,
                callback=callbacks,
                progress_bar=progress_bar,
                reset_num_timesteps=False
            )
            
            print(f"Completed stage {i+1}/{len(data_stages)}")
        
        print("\nCurriculum training complete!")
        
    elif enhancement_type == "adaptive":
        print("\nApplying adaptive hyperparameter enhancement...")
        
        # First check if optuna is available
        have_optuna = False
        try:
            import optuna
            from optuna.pruners import MedianPruner
            have_optuna = True
        except ImportError:
            print("Optuna not installed. Falling back to continued training with default parameters.")
            print("To use adaptive enhancement, install optuna: pip install optuna")
        
        if have_optuna:
            print("Setting up hyperparameter optimization study...")
            
            # Create evaluation function
            def evaluate_model(test_env, model, n_eval_episodes=5):
                """Evaluate a model on test environment"""
                episode_rewards = []
                for _ in range(n_eval_episodes):
                    obs, _ = test_env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, _ = model.predict(obs, deterministic=False)
                        obs, reward, done, _, _ = test_env.step(action)
                        episode_reward += reward
                    episode_rewards.append(episode_reward)
                return np.mean(episode_rewards)
            
            # Define objective function for optimization
            def objective(trial):
                # Sample hyperparameters
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
                batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
                
                # Create a copy of the model with new hyperparameters
                trial_model = PPO.load(model_path)
                trial_model.learning_rate = learning_rate
                trial_model.ent_coef = ent_coef
                trial_model.batch_size = batch_size
                trial_model.set_env(train_env)
                
                # Short training run
                trial_model.learn(total_timesteps=20000, reset_num_timesteps=False)
                
                # Evaluate performance
                mean_reward = evaluate_model(eval_env, trial_model)
                return mean_reward
            
            try:
                # Create and run study
                iterations = kwargs.get("iterations", 5)
                print(f"Running hyperparameter search with {iterations} iterations...")
                
                study = optuna.create_study(direction="maximize", pruner=MedianPruner())
                study.optimize(objective, n_trials=iterations)
                
                # Get best parameters
                best_params = study.best_params
                print(f"Best hyperparameters found: {best_params}")
                
                # Update model with best parameters
                model.learning_rate = best_params.get("learning_rate", model.learning_rate)
                model.ent_coef = best_params.get("ent_coef", model.ent_coef)
                if "batch_size" in best_params:
                    model.batch_size = best_params["batch_size"]
            except Exception as e:
                print(f"Error during hyperparameter optimization: {e}")
                print("Falling back to continued training")
        
    elif enhancement_type == "distillation":
        print("\nApplying policy distillation enhancement...")
        
        # Get teacher model paths
        teacher_paths = kwargs.get("teacher_model_paths")
        if not teacher_paths:
            print("No teacher models provided. Falling back to continued training.")
        else:
            try:
                # Load teacher models
                print(f"Loading {len(teacher_paths)} teacher models...")
                teachers = []
                for i, path in enumerate(teacher_paths):
                    try:
                        teachers.append(PPO.load(path))
                        print(f"✓ Loaded teacher model {i+1}: {path}")
                    except Exception as e:
                        print(f"✗ Failed to load teacher model {path}: {e}")
                
                if not teachers:
                    print("No teacher models loaded successfully. Falling back to continued training.")
                else:
                    print(f"Successfully loaded {len(teachers)} teacher models")
                    # Note: We're doing simple knowledge distillation by averaging predictions
                    # Full implementation would require custom training loop with KL divergence
                    print("Advanced distillation would require custom training loop")
            except Exception as e:
                print(f"Error in distillation setup: {e}")
                print("Falling back to continued training")
    
    else:
        raise ValueError(f"Unknown enhancement type: {enhancement_type}")
    
    # Train the model unless curriculum learning was used (which already did training)
    if enhancement_type != "curriculum":
        print("\n" + "="*50)
        print(f"Starting enhancement training for {timesteps} timesteps...")
        print("="*50 + "\n")
        
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                progress_bar=progress_bar,
                reset_num_timesteps=False  # Continue counting from previous training
            )
            print("\nEnhancement training completed successfully!")
        except Exception as e:
            print(f"\n❌ Error during enhancement training: {e}")
            traceback.print_exc()
    
    # Save the enhanced model
    try:
        model.save(save_path)
        print(f"\n✅ Enhanced model saved to {save_path}")
        
        # # Save enhancement metadata
        # metadata = {
        #     "original_model": model_path,
        #     "enhancement_type": enhancement_type,
        #     "training_data_size": len(new_data),
        #     "eval_data_size": len(eval_data),
        #     "timesteps": timesteps,
        #     "enhancement_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        #     "hyperparameters": {
        #         "learning_rate": float(model.learning_rate),
        #         "ent_coef": float(model.ent_coef),
        #         "n_steps": int(model.n_steps),
        #         "batch_size": int(model.batch_size),
        #         "gae_lambda": float(model.gae_lambda),
        #     }
        # }
        
        # # Add enhancement-specific metadata
        # if enhancement_type == "adaptive" and "best_params" in locals():
        #     metadata["best_hyperparameters"] = best_params
        
        # with open(f"{os.path.splitext(save_path)[0]}_metadata.json", "w") as f:
        #     json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error saving enhanced model: {e}")
    
    print("\nEnhancement complete!")
    print("\nAvailable monitoring tools:")
    print(f"1. Launch monitoring dashboard: python Scripts/monitoring_dashboard.py --model={save_path} --log-dir={log_path}")
    print(f"2. View TensorBoard logs: tensorboard --logdir={tensorboard_path}")
    
    return model


if __name__ == "__main__":
    import time
    import argparse
    
    # Add command line arguments for easier control
    parser = argparse.ArgumentParser(description="Train or enhance RL trading model")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--check-env", action="store_true", help="Run environment checker before training")
    
    # Model enhancement arguments
    parser.add_argument("--enhance", default=True, help="whether to enhance an existing model", action="store_true")
    parser.add_argument("--enhance-path", type=str,default="RL/Models/Experiments/best_model.zip", help="Path to existing model to enhance")
    parser.add_argument("--enhance-type", type=str, default="continued", 
                       choices=["continued", "replay", "curriculum", "adaptive", "distillation"],
                       help="Type of enhancement to apply")
    parser.add_argument("--replay-buffer", type=str, help="Path to replay buffer for replay enhancement")
    parser.add_argument("--learning-rate", type=float, help="Custom learning rate for enhancement")
    parser.add_argument("--teacher-models", type=str, nargs="+", help="Teacher model paths for distillation")
    parser.add_argument("--save-path", type=str, help="Path to save the enhanced model")
    
    args = parser.parse_args()
    
    # Run environment checker if requested
    if args.check_env:
        try:
            print("\nRunning environment compatibility check...")
            from RL.Envs.Utils.env_visualization import check_trading_environment
            check_trading_environment()
            proceed = input("\nDo you want to proceed with training? (y/n): ")
            if proceed.lower() != 'y':
                print("Training canceled.")
                sys.exit(0)
        except Exception as e:
            print(f"Error running environment check: {e}")
            print("Proceeding with training anyway...")
    
    start_time = time.time()
    
    # Check if enhancing an existing model
    if args.enhance:
        print(f"\nEnhancing existing model: {args.enhance_path}")
        print(f"Enhancement type: {args.enhance_type}")
        
        # Prepare enhancement kwargs
        enhancement_kwargs = {
            "learning_rate": args.learning_rate,
            "reward_type": "sharpe",  # Default reward type, can be overridden
        }
        
        # Add enhancement-specific arguments
        if args.enhance_type == "replay" and args.replay_buffer:
            enhancement_kwargs["replay_buffer_path"] = args.replay_buffer
        
        if args.enhance_type == "distillation" and args.teacher_models:
            enhancement_kwargs["teacher_model_paths"] = args.teacher_models
            
        # Call the enhancement function
        model = enhance_existing_model(
            model_path=args.enhance_path,
            enhancement_type=args.enhance_type,
            timesteps=args.timesteps,
            save_path=args.save_path,
            progress_bar=not args.no_progress,
            **enhancement_kwargs
        )
        
        end_time = time.time()
        print(f"\nModel enhancement completed in {(end_time - start_time)/60:.2f} minutes")
    else:
        # Original training code
        model = train_rl_model(
            timesteps=args.timesteps,
            progress_bar=not args.no_progress,
        )
        end_time = time.time()
        print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")
