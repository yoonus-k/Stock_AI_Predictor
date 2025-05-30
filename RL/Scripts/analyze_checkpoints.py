import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Data.loader import load_data_from_db
from RL.Envs.trading_env import PatternSentimentEnv


def plot_training_progress(log_dir, save_path=None, show=True):
    """
    Plot training progress from Monitor logs
    
    Parameters:
        log_dir: Directory containing Monitor logs
        save_path: Where to save the plot image
        show: Whether to display the plot
    """
    # Load monitor data
    monitor_data = load_results(log_dir)
    
    # Skip if no data available
    if len(monitor_data) == 0:
        print("No monitor data found.")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot rewards
    axes[0].plot(monitor_data.l.values, monitor_data.r.values, label='Reward')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Progress')
    axes[0].grid(True)
    
    # Calculate and plot rolling average
    window_size = min(50, len(monitor_data) // 10) if len(monitor_data) > 100 else 5
    if len(monitor_data) > window_size:
        rolling_avg = pd.Series(monitor_data.r.values).rolling(window_size).mean()
        axes[0].plot(monitor_data.l.values, rolling_avg, 'r', label=f'{window_size}-episode Rolling Avg')
        axes[0].legend()
    
    # Plot episode lengths
    axes[1].plot(monitor_data.l.values, monitor_data.l.values)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot if requested
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Training progress plot saved to: {save_path}")
    
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()


def analyze_checkpoints(checkpoint_dir, data=None, num_episodes=3):
    """
    Analyze performance of model checkpoints
    
    Parameters:
        checkpoint_dir: Directory containing model checkpoints
        data: Evaluation data. If None, loads from database.
        num_episodes: Number of episodes to evaluate each checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all checkpoint files
    checkpoint_files = sorted([
        f for f in checkpoint_dir.glob("*.zip") 
        if "trading_model" in str(f)
    ], key=lambda f: int(str(f).split("_")[-1].split(".")[0]))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Load evaluation data if not provided
    if data is None:
        data = load_data_from_db()
        # Use only validation portion (20%)
        split_idx = int(len(data) * 0.8)
        data = data[split_idx:]
    
    # Create evaluation environment
    env = PatternSentimentEnv(data)
    
    # Track results
    results = []
    
    # Evaluate each checkpoint
    for checkpoint in checkpoint_files:
        try:
            print(f"Evaluating {checkpoint.name}...")
            
            # Load the model
            model = PPO.load(checkpoint)
            
            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(
                model, 
                env,
                n_eval_episodes=num_episodes,
                deterministic=True
            )
            
            # Extract the timestep from filename
            timestep = int(checkpoint.stem.split("_")[-1])
            
            # Record results
            results.append({
                'timestep': timestep,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'checkpoint': checkpoint.name
            })
            
        except Exception as e:
            print(f"Error evaluating {checkpoint.name}: {e}")
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\n===== CHECKPOINT EVALUATION RESULTS =====\n")
    print(results_df)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['timestep'], results_df['mean_reward'], 'o-')
    plt.fill_between(
        results_df['timestep'],
        results_df['mean_reward'] - results_df['std_reward'],
        results_df['mean_reward'] + results_df['std_reward'],
        alpha=0.2
    )
    plt.xlabel('Training Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Checkpoint Performance')
    plt.grid(True)
    
    # Add best checkpoint marker
    best_idx = results_df['mean_reward'].argmax()
    plt.scatter(
        results_df.iloc[best_idx]['timestep'], 
        results_df.iloc[best_idx]['mean_reward'],
        s=100, 
        color='red', 
        zorder=5,
        label=f'Best: {results_df.iloc[best_idx]["checkpoint"]}'
    )
    plt.legend()
    
    # Save plot
    plot_path = checkpoint_dir.parent / f"checkpoint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path)
    print(f"\nAnalysis plot saved to: {plot_path}")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Return the best checkpoint
    best_checkpoint = checkpoint_files[best_idx]
    print(f"\nBest checkpoint: {best_checkpoint}")
    print(f"Mean reward: {results_df.iloc[best_idx]['mean_reward']:.4f} ± {results_df.iloc[best_idx]['std_reward']:.4f}")
    
    return best_checkpoint, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model training and checkpoints")
    parser.add_argument('--log-dir', type=str, default=None, help="Directory containing monitor logs")
    parser.add_argument('--checkpoint-dir', type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument('--episodes', type=int, default=3, help="Number of episodes for checkpoint evaluation")
    parser.add_argument('--no-plot', action='store_true', help="Don't display plots")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.log_dir is None:
        args.log_dir = Path(__file__).resolve().parent.parent / "Logs"
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.log_dir) / "checkpoints"
    
    # Plot training progress
    plot_training_progress(args.log_dir, show=not args.no_plot)
    
    # Analyze checkpoints
    analyze_checkpoints(args.checkpoint_dir, num_episodes=args.episodes)
