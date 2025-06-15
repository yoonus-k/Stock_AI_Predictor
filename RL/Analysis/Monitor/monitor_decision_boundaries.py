"""
Decision Boundary and Pattern Recognition Analysis for RL Trading Model

This script provides post-training analysis tools for RL trading models:
1. Decision boundary visualization - Visualize how the model makes decisions based on key features
2. Pattern recognition analysis - Identify patterns in the data that trigger specific trading actions
3. State-value visualization - Visualize the learned value function across different states

These visualizations help understand what the model has learned and how it makes trading decisions.
"""

# Suppress TensorFlow oneDNN optimization warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Data.Utils.loader import load_data_from_db
from RL.Envs.trading_env import TradingEnv


def generate_decision_boundary(model, feature1_idx, feature2_idx, feature_names, 
                              n_points=50, output_dir=None):
    """
    Visualize decision boundary for two selected features
    
    Parameters:
        model: Trained RL model
        feature1_idx: Index of first feature for visualization
        feature2_idx: Index of second feature for visualization
        feature_names: List of feature names
        n_points: Number of points in each dimension for the grid
        output_dir: Directory to save output
    """
    print(f"Generating decision boundary for {feature_names[feature1_idx]} vs {feature_names[feature2_idx]}...")
    
    # Get observation space from model
    observation_space = model.observation_space
    
    # Create grid of points
    f1_range = np.linspace(observation_space.low[feature1_idx], 
                           observation_space.high[feature1_idx], n_points)
    f2_range = np.linspace(observation_space.low[feature2_idx], 
                           observation_space.high[feature2_idx], n_points)
    
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)
    
    # Store actions for each grid point
    actions = np.zeros_like(f1_grid, dtype=int)
    values = np.zeros_like(f1_grid, dtype=float)
    
    # Default observation (neutral state)
    default_observation = np.zeros(observation_space.shape[0])
    
    # Fill in values and actions for each grid point
    for i in range(n_points):
        for j in range(n_points):
            # Create observation with current grid point values
            observation = default_observation.copy()
            observation[feature1_idx] = f1_grid[i, j]
            observation[feature2_idx] = f2_grid[i, j]
              # Get action and value from model
            action, _ = model.predict(observation, deterministic=False)
            
            # Handle MultiDiscrete action space - extract action type only for boundary visualization
            if hasattr(action, '__len__') and len(action) > 1:
                action_type = int(action[0])  # Extract action type (0=Hold, 1=Buy, 2=Sell)
            else:
                action_type = int(action)
                
            actions[i, j] = action_type
            
            # Extract value estimate
            obs_tensor = model.policy.obs_to_tensor(observation)[0]
            values[i, j] = model.policy.predict_values(obs_tensor).detach().numpy()
    
    # Plot decision boundaries
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot action boundaries
    action_cmap = plt.cm.get_cmap('viridis', 3)  # 3 possible actions
    action_mesh = ax1.pcolormesh(f1_grid, f2_grid, actions, cmap=action_cmap, shading='auto')
    ax1.set_xlabel(feature_names[feature1_idx])
    ax1.set_ylabel(feature_names[feature2_idx])
    ax1.set_title(f"Decision Boundary - Actions")
    
    # Add colorbar
    cbar = fig.colorbar(action_mesh, ax=ax1, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Hold', 'Buy', 'Sell'])
    
    # Plot value function
    value_mesh = ax2.pcolormesh(f1_grid, f2_grid, values, cmap='viridis', shading='auto')
    ax2.set_xlabel(feature_names[feature1_idx])
    ax2.set_ylabel(feature_names[feature2_idx])
    ax2.set_title("Value Function")
    fig.colorbar(value_mesh, ax=ax2)
    
    plt.tight_layout()
    
    # Save plot if output directory is specified
    if output_dir:
        save_path = Path(output_dir) / f"decision_boundary_{feature1_idx}_{feature2_idx}.png"
        plt.savefig(save_path)
        print(f"Decision boundary saved to {save_path}")
    
    plt.show()


def plot_key_feature_pairs(model, eval_env, output_dir=None):
    """
    Plot decision boundaries for key feature pairs
    
    Parameters:
        model: Trained RL model
        eval_env: Evaluation environment
        output_dir: Directory to save outputs
    """    # Define feature names for better visualization (31 total features)
    feature_names = [
        # Base pattern features (7 features)
        "probability",
        "action", 
        "reward_risk_ratio",
        "max_gain",
        "max_drawdown_pattern",
        "mse",
        "expected_value",
        # Technical indicators (3 features)
        "rsi",
        "atr",
        "atr_ratio",
        # Sentiment features (2 features)
        "unified_sentiment",
        "sentiment_count",
        # COT data (6 features)
        "net_noncommercial",
        "net_nonreportable",
        "change_nonrept_long",
        "change_nonrept_short",
        "change_noncommercial_long",
        "change_noncommercial_short",
        # Time features (7 features)
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "asian_session",
        "london_session",
        "ny_session",
        # Portfolio features (6 features)
        "balance_ratio",
        "position_ratio",
        "position",
        "max_drawdown_portfolio",
        "win_rate",
        "steps_without_action",
    ]    # Define key feature pairs to visualize (updated for 31-feature structure)
    key_pairs = [
        # Pattern recognition and RSI
        (0, 7),   # probability vs RSI
        (3, 7),   # max_gain vs RSI
        (2, 7),   # reward_risk_ratio vs RSI
        
        # Sentiment and pattern features
        (0, 10),  # probability vs unified_sentiment
        (10, 11), # unified_sentiment vs sentiment_count
        (3, 10),  # max_gain vs unified_sentiment
        
        # COT data relationships
        (12, 13), # net_noncommercial vs net_nonreportable
        (16, 17), # change_noncommercial_long vs change_noncommercial_short
        
        # Portfolio features  
        (25, 26), # balance_ratio vs position_ratio
        (26, 27), # position_ratio vs position
        (28, 29), # max_drawdown_portfolio vs win_rate
    ]
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate decision boundaries for each pair
    for idx1, idx2 in key_pairs:
        generate_decision_boundary(
            model, 
            idx1, idx2, 
            feature_names, 
            n_points=30,
            output_dir=output_dir
        )


def visualize_state_space(model, eval_env, num_episodes=5, num_samples=1000, output_dir=None):
    """
    Visualize the state space and learned policy in lower dimensions
    
    Parameters:
        model: Trained RL model
        eval_env: Evaluation environment
        num_episodes: Number of episodes to sample
        num_samples: Maximum number of samples to collect
        output_dir: Directory to save outputs
    """
    print("\nCollecting state samples for visualization...")
    
    # Initialize containers
    states = []
    actions = []
    rewards = []
    values = []
    
    # Sample states from environment
    sample_count = 0
    
    for _ in tqdm(range(num_episodes)):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        while not (done or truncated) and sample_count < num_samples:
            # Get action and value from model
            action, _ = model.predict(obs, deterministic=False)
            
            # Handle MultiDiscrete action space - extract action type for visualization
            if hasattr(action, '__len__') and len(action) > 1:
                action_type = int(action[0])  # Extract action type (0=Hold, 1=Buy, 2=Sell)
            else:
                action_type = int(action)
            
            value = model.policy.predict_values(
                model.policy.obs_to_tensor(obs)[0]
            ).detach().numpy().item()
            
            # Store data
            states.append(obs.copy())
            actions.append(action_type)
            values.append(value)
            
            # Step environment (use full action for environment)
            next_obs, reward, done, truncated, info = eval_env.step(action)
            rewards.append(reward)
            
            # Update observation
            obs = next_obs
            sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    values = np.array(values)
    
    print(f"Collected {len(states)} state samples")
    
    # Apply dimensionality reduction
    print("Applying dimensionality reduction...")
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(states)
    
    # t-SNE for visualization (can be slow for large datasets)
    if len(states) <= 5000:
        try:
            tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
            tsne_results = tsne.fit_transform(states)
        except Exception as e:
            print(f"Error during t-SNE: {e}")
            tsne_results = None
    else:
        print("Too many samples for t-SNE visualization, skipping")
        tsne_results = None
    
    # Plot PCA visualization with actions
    plt.figure(figsize=(12, 10))
    
    # Define colors and labels for actions
    action_colors = {0: 'gray', 1: 'green', 2: 'red'}
    action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    
    # Create scatter plot
    for action in np.unique(actions):
        mask = actions == action
        plt.scatter(
            pca_results[mask, 0], 
            pca_results[mask, 1],
            c=action_colors[action],
            label=action_labels[action],
            alpha=0.7
        )
    
    plt.title('PCA Visualization of State Space with Actions')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if output directory is specified
    if output_dir:
        save_path = Path(output_dir) / "state_space_pca.png"
        plt.savefig(save_path)
        print(f"PCA visualization saved to {save_path}")
    
    plt.show()
    
    # Plot t-SNE visualization if available
    if tsne_results is not None:
        plt.figure(figsize=(12, 10))
        
        # Plot with actions
        for action in np.unique(actions):
            mask = actions == action
            plt.scatter(
                tsne_results[mask, 0], 
                tsne_results[mask, 1],
                c=action_colors[action],
                label=action_labels[action],
                alpha=0.7
            )
        
        plt.title('t-SNE Visualization of State Space with Actions')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot if output directory is specified
        if output_dir:
            save_path = Path(output_dir) / "state_space_tsne.png"
            plt.savefig(save_path)
            print(f"t-SNE visualization saved to {save_path}")
        
        plt.show()
    
    # Plot PCA visualization with values
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        pca_results[:, 0], 
        pca_results[:, 1],
        c=values,
        cmap='viridis',
        alpha=0.7
    )
    
    plt.title('PCA Visualization of State Space with Value Function')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter, label='State Value')
    plt.grid(True, alpha=0.3)
    
    # Save plot if output directory is specified
    if output_dir:
        save_path = Path(output_dir) / "state_space_values_pca.png"
        plt.savefig(save_path)
        print(f"Value function visualization saved to {save_path}")
    
    plt.show()
    
    return states, actions, rewards, values


def analyze_pattern_recognition(model, eval_env, output_dir=None):
    """
    Analyze what patterns the model recognizes for trading decisions
    
    Parameters:
        model: Trained RL model
        eval_env: Evaluation environment
        output_dir: Directory to save outputs
    """
    print("\nCollecting data for pattern recognition analysis...")
    
    # Reset environment
    obs, _ = eval_env.reset()
    
    # Collect data for pattern analysis
    observations = []
    actions = []
    portfolio_states = []
    rewards = []
    
    # Run through data
    done = False
    truncated = False
    while not (done or truncated):
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)
        
        # Handle MultiDiscrete action space - extract action type for analysis
        if hasattr(action, '__len__') and len(action) > 1:
            action_type = int(action[0])  # Extract action type (0=Hold, 1=Buy, 2=Sell)
        else:
            action_type = int(action)
        
        # Store data
        observations.append(obs.copy())
        actions.append(action_type)
        
        # Extract portfolio state if available (updated for new environment structure)
        if hasattr(eval_env, 'position'):
            portfolio_state = {
                'position': eval_env.position,
                'balance': getattr(eval_env, 'balance', 0),
                'position_value': getattr(eval_env, 'position_value', 0),
            }
            portfolio_states.append(portfolio_state)
        
        # Step environment (use full action for environment)
        next_obs, reward, done, truncated, info = eval_env.step(action)
        rewards.append(reward)
        
        # Update observation
        obs = next_obs
    
    # Convert to arrays
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    print(f"Collected {len(observations)} observations")
    
    # Group by action
    action_observations = {
        0: observations[actions == 0],  # Hold
        1: observations[actions == 1],  # Buy
        2: observations[actions == 2]   # Sell
    }
      # Analyze price patterns leading to different actions
    print("Analyzing action patterns...")
    
    # Analyze feature distributions for different actions
    feature_names = [
        # Base pattern features (7 features)
        "probability", "action", "reward_risk_ratio", "max_gain", "max_drawdown_pattern", 
        "mse", "expected_value",
        # Technical indicators (3 features)  
        "rsi", "atr", "atr_ratio",
        # Sentiment features (2 features)
        "unified_sentiment", "sentiment_count",
        # COT data (6 features)
        "net_noncommercial", "net_nonreportable", "change_nonrept_long", 
        "change_nonrept_short", "change_noncommercial_long", "change_noncommercial_short",
        # Time features (7 features)
        "hour_sin", "hour_cos", "day_sin", "day_cos", 
        "asian_session", "london_session", "ny_session",
        # Portfolio features (6 features)
        "balance_ratio", "position_ratio", "position", 
        "max_drawdown_portfolio", "win_rate", "steps_without_action"
    ]
      # Select key features to analyze (updated indices for 31-feature structure)
    key_features = [0, 2, 3, 7, 10, 28, 29, 30]  # probability, reward_risk_ratio, max_gain, rsi, sentiment, balance_ratio, position_ratio, position
    
    # Plot distributions for key features
    for feature_idx in key_features:
        plt.figure(figsize=(10, 6))
        
        feature_name = feature_names[feature_idx]
        
        # Plot distribution for each action
        for action, name, color in [(0, 'Hold', 'gray'), (1, 'Buy', 'green'), (2, 'Sell', 'red')]:
            if action in action_observations and len(action_observations[action]) > 0:
                feature_values = action_observations[action][:, feature_idx]
                
                # Skip if all values are the same
                if np.std(feature_values) == 0:
                    continue
                
                # Plot kernel density estimate
                sns.kdeplot(feature_values, label=name, color=color)
        
        plt.title(f'Distribution of {feature_name} by Action')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot if output directory is specified
        if output_dir:
            save_path = Path(output_dir) / f"feature_distribution_{feature_name}.png"
            plt.savefig(save_path)
            print(f"Feature distribution saved to {save_path}")
        
        plt.show()
    
    return observations, actions, rewards


def run_decision_boundary_analysis(model_path, data=None, output_dir=None):
    """
    Run comprehensive decision boundary analysis for a trained model
    
    Parameters:
        model_path: Path to the trained model
        data: Optional evaluation data
        output_dir: Directory to save outputs
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "Analysis" / "decision_boundaries"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, custom_objects={'learning_rate': 0.0003})
    
    # Load evaluation data if not provided
    if data is None:
        print("Loading data from database...")
        data = load_data_from_db()
        # Use only validation portion (20%)
        split_idx = int(len(data) * 0.7)
        data = data[split_idx:]
    
    print(f"Using {len(data)} records for evaluation")
    
    # Create evaluation environment
    env = TradingEnv(
        data,
        normalize_observations=True,
    )
    
    # 1. Visualize decision boundaries
    print("\n=== Analyzing Decision Boundaries ===")
    plot_key_feature_pairs(model, env, output_dir)
    
    # 2. Visualize state space
    print("\n=== Visualizing State Space ===")
    states, actions, rewards, values = visualize_state_space(model, env, output_dir=output_dir)
    
    # 3. Analyze pattern recognition
    print("\n=== Analyzing Pattern Recognition ===")
    observations, actions, rewards = analyze_pattern_recognition(model, env, output_dir)
    
    print(f"\nDecision boundary analysis complete. Results saved to {output_dir}")
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "values": values
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze decision boundaries of a trained RL model")
    parser.add_argument('--model', type=str, default="RL/Models/Experiments/best_model_continued.zip",  help="Path to the trained model")
    parser.add_argument('--output-dir', type=str, default="Images/RL/Analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    run_decision_boundary_analysis(
        model_path=args.model,
        output_dir=args.output_dir
    )
