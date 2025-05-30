"""
Feature Importance Analysis for Reinforcement Learning Trading Model

This script performs feature importance analysis on a trained RL model to understand 
which features have the greatest impact on the model's decisions.

Methods implemented:
1. Permutation Importance - Shuffle individual features and measure impact on performance
2. SHAP Values - Use SHAP (SHapley Additive exPlanations) for feature contribution analysis
3. Feature Ablation - Systematically remove features and measure performance impact
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import shap
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Data.loader import load_data_from_db
from RL.Envs.trading_env import PatternSentimentEnv
from RL.Envs.action_wrapper import TupleActionWrapper
from RL.Envs.observation_normalizer import ObservationNormalizer


def calculate_permutation_importance(model, env, feature_names=None, num_episodes=10, num_shuffles=5):
    """
    Calculate feature importance by permuting (shuffling) features and measuring performance change
    
    Parameters:
        model: Trained RL model
        env: Trading environment
        feature_names: List of feature names corresponding to observation space
        num_episodes: Number of episodes to evaluate each permutation
        num_shuffles: Number of times to shuffle each feature
        
    Returns:
        DataFrame with feature importances
    """
    print("\n== Calculating Permutation Importance ==")
    
    # If feature names not provided, use generic names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(env.observation_space.shape[0])]
    
    # Ensure we have the right number of feature names
    if len(feature_names) != env.observation_space.shape[0]:
        print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match observation space " +
              f"dimension ({env.observation_space.shape[0]}). Using generic names.")
        feature_names = [f"feature_{i}" for i in range(env.observation_space.shape[0])]
    
    # Get baseline performance (no features shuffled)
    baseline_mean_reward, baseline_std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    print(f"Baseline reward: {baseline_mean_reward:.4f} ± {baseline_std_reward:.4f}")
    
    # Store importance results
    importance_results = []
    
    # For each feature, shuffle its values and measure performance
    for feature_idx, feature_name in enumerate(tqdm(feature_names, desc="Features")):
        feature_rewards = []
        
        # Repeat shuffling multiple times to get robust estimates
        for shuffle in range(num_shuffles):
            # Create a modified environment with one feature shuffled
            modified_env = create_environment_with_shuffled_feature(env, feature_idx)
            
            # Evaluate model with the modified environment
            mean_reward, std_reward = evaluate_policy(
                model, 
                modified_env,
                n_eval_episodes=num_episodes,
                deterministic=True
            )
            
            # Calculate importance as the decrease in performance
            importance = baseline_mean_reward - mean_reward
            feature_rewards.append(importance)
        
        # Average across shuffles
        avg_importance = np.mean(feature_rewards)
        std_importance = np.std(feature_rewards)
        
        # Store results
        importance_results.append({
            'feature_name': feature_name,
            'feature_idx': feature_idx,
            'importance': avg_importance,
            'std': std_importance
        })
    
    # Convert to DataFrame and sort by importance
    importance_df = pd.DataFrame(importance_results).sort_values('importance', ascending=False)
    
    return importance_df


def create_environment_with_shuffled_feature(env, feature_idx):
    """
    Create a modified environment where one feature's values are shuffled
    
    Parameters:
        env: Original environment
        feature_idx: Index of feature to shuffle
        
    Returns:
        Modified environment
    """
    # Get the unwrapped environment to access its properties
    unwrapped_env = env.unwrapped
    
    # Create a copy of the environment
    base_modified_env = PatternSentimentEnv(
        unwrapped_env.data.copy(),
        normalize_observations=unwrapped_env.normalize_observations,
        enable_adaptive_scaling=unwrapped_env.normalizer.enable_adaptive_scaling if hasattr(unwrapped_env, 'normalizer') else False
    )
    
    # Wrap with action converter
    modified_env = TupleActionWrapper(base_modified_env)
    
    # Store original step function
    original_step = modified_env.step
    original_get_obs = modified_env._get_observation
    
    # Override the _get_observation method
    def modified_get_observation():
        # Get original observation
        observation = original_get_obs()
        
        if observation is not None:
            # Shuffle the specified feature
            observation[feature_idx] = np.random.uniform(
                low=env.observation_space.low[feature_idx],
                high=env.observation_space.high[feature_idx]
            )
        
        return observation
    
    # Replace the method
    modified_env._get_observation = modified_get_observation
    
    return modified_env


def analyze_feature_ablation(model, env, feature_names=None, num_episodes=5, group_size=3):
    """
    Analyze model performance when groups of features are removed
    
    Parameters:
        model: Trained RL model
        env: Trading environment
        feature_names: List of feature names corresponding to observation space
        num_episodes: Number of episodes for evaluation
        group_size: Number of features to remove in each ablation test
        
    Returns:
        DataFrame with feature group importance
    """
    print("\n== Analyzing Feature Ablation ==")
    
    # If feature names not provided, use generic names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(env.observation_space.shape[0])]
    
    # Get baseline performance (all features)
    baseline_mean_reward, baseline_std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    print(f"Baseline reward (all features): {baseline_mean_reward:.4f} ± {baseline_std_reward:.4f}")
    
    # Store results
    ablation_results = []
    
    # Group features
    feature_indices = list(range(env.observation_space.shape[0]))
    num_groups = env.observation_space.shape[0] // group_size
    
    # Evaluate each feature group
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        
        # Indices to remove
        indices_to_remove = feature_indices[start_idx:end_idx]
        
        # Feature names in this group
        group_feature_names = [feature_names[idx] for idx in indices_to_remove]
        group_name = f"Group {i+1}: {', '.join(group_feature_names)}"
        
        print(f"Testing without {group_name}...")
        
        # Create environment without these features
        masked_env = create_environment_with_masked_features(env, indices_to_remove)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model,
            masked_env,
            n_eval_episodes=num_episodes,
            deterministic=True
        )
        
        # Calculate importance
        importance = baseline_mean_reward - mean_reward
        
        # Store results
        ablation_results.append({
            'group': f"Group {i+1}",
            'features': group_feature_names,
            'reward': mean_reward,
            'std': std_reward,
            'importance': importance
        })
    
    # Convert to DataFrame and sort by importance
    ablation_df = pd.DataFrame(ablation_results).sort_values('importance', ascending=False)
    
    return ablation_df


def create_environment_with_masked_features(env, feature_indices):
    """
    Create a modified environment where specified features are masked (set to zero)
    
    Parameters:
        env: Original environment
        feature_indices: Indices of features to mask
        
    Returns:
        Modified environment
    """
    # Get the unwrapped environment to access its properties
    unwrapped_env = env.unwrapped
    
    # Create a copy of the environment
    base_masked_env = PatternSentimentEnv(
        unwrapped_env.data.copy(),
        normalize_observations=unwrapped_env.normalize_observations,
        enable_adaptive_scaling=unwrapped_env.normalizer.enable_adaptive_scaling if hasattr(unwrapped_env, 'normalizer') else False
    )
    
    # Wrap with action converter
    masked_env = TupleActionWrapper(base_masked_env)
    
    # Store original get observation function
    original_get_obs = masked_env._get_observation
    
    # Override the _get_observation method
    def modified_get_observation():
        # Get original observation
        observation = original_get_obs()
        
        if observation is not None:
            # Mask specified features by setting them to zero
            for idx in feature_indices:
                if idx < len(observation):
                    observation[idx] = 0.0
        
        return observation
    
    # Replace the method
    masked_env._get_observation = modified_get_observation
    
    return masked_env


def calculate_shap_values(model, env, num_samples=100, feature_names=None):
    """
    Calculate SHAP values to explain model predictions
    
    Parameters:
        model: Trained RL model
        env: Trading environment
        num_samples: Number of observations to sample for SHAP analysis
        feature_names: List of feature names corresponding to observation space
        
    Returns:
        SHAP values and corresponding observations
    """
    print("\n== Calculating SHAP Values ==")
    
    # If feature names not provided, use generic names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(env.observation_space.shape[0])]
    
    # Collect observations
    observations = []
    
    # Reset environment
    obs, _ = env.reset()
    observations.append(obs)
    
    # Sample observations by stepping through environment
    for _ in tqdm(range(num_samples - 1), desc="Collecting observations"):
        action = env.action_space.sample()  # Random action
        obs, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()
        
        observations.append(obs)
    
    # Convert to numpy array
    observations = np.array(observations)
    
    # Define prediction function for SHAP
    def predict(X):
        return np.array([model.policy.evaluate_actions(x.reshape(1, -1), None)[0].detach().numpy() for x in X])
    
    # Initialize explainer
    explainer = shap.KernelExplainer(predict, observations)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(observations)
    
    return shap_values, observations, feature_names


def plot_feature_importance(importance_df, title="Feature Importance", save_path=None):
    """
    Plot feature importance
    
    Parameters:
        importance_df: DataFrame with feature importance results
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    df_plot = importance_df.copy()
    df_plot = df_plot.sort_values('importance', ascending=True)
    
    # Create horizontal bar chart
    bars = plt.barh(df_plot['feature_name'], df_plot['importance'], xerr=df_plot['std'], 
            color='skyblue', ecolor='black', capsize=5)
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + bar.get_width()*0.05, 
                bar.get_y() + bar.get_height()/2, 
                f"{df_plot['importance'].iloc[i]:.3f}", 
                va='center')
    
    # Add labels and title
    plt.xlabel('Importance (Reduction in Reward)')
    plt.title(title)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_shap_summary(shap_values, observations, feature_names, save_path=None):
    """
    Plot SHAP summary plots
    
    Parameters:
        shap_values: SHAP values
        observations: Observations used for SHAP analysis
        feature_names: Feature names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create DataFrame with feature names
    X_df = pd.DataFrame(observations, columns=feature_names)
    
    # Summary plot
    shap.summary_plot(shap_values, X_df, show=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.show()
    
    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path and save_path.endswith('.png'):
        bar_path = save_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, bbox_inches='tight')
        print(f"SHAP bar plot saved to {bar_path}")
    
    plt.show()


def analyze_feature_importance(model_path, data=None, output_dir=None, num_episodes=5, 
                              feature_groups=None):
    """
    Perform comprehensive feature importance analysis
    
    Parameters:
        model_path: Path to the trained model
        data: Optional evaluation data
        output_dir: Directory to save results
        num_episodes: Number of episodes for evaluation
        feature_groups: Dictionary mapping feature indices to group names
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "Analysis"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Load evaluation data if not provided
    if data is None:
        print("Loading data from database...")
        data = load_data_from_db()
        # Use only validation portion (20%)
        split_idx = int(len(data) * 0.8)
        data = data[split_idx:]
    
    print(f"Using {len(data)} records for evaluation")
    
    # Define feature names based on the actual features in your trading environment
    feature_names = [
        # Market features
        "close_price", "open_price", "high_price", "low_price",
        "volume", "price_change", "price_volatility", "ma_5",
        "ma_10", "ma_20", "ema_5", "ema_10",
        "ema_20", "rsi", "macd", "macd_signal",
        "macd_hist", "bollinger_upper", "bollinger_middle", "bollinger_lower",
        "atr", "adi", "obv", "pattern_bullish",
        "pattern_bearish", "sentiment",
        # Portfolio features
        "position", "entry_price", "stop_loss", "take_profit"
    ]
      # Create evaluation environment
    base_env = PatternSentimentEnv(
        data,
        normalize_observations=True,
        enable_adaptive_scaling=False
    )
    # Wrap with action converter for compatibility with the trained model
    env = TupleActionWrapper(base_env)
    
    # 1. Permutation Importance
    permutation_results = calculate_permutation_importance(
        model, 
        env, 
        feature_names=feature_names,
        num_episodes=num_episodes
    )
    
    print("\nTop 10 most important features (permutation method):")
    print(permutation_results.head(10))
    
    # Save results
    permutation_results.to_csv(output_dir / "permutation_importance.csv", index=False)
    
    # Plot permutation importance
    plot_feature_importance(
        permutation_results, 
        title="Feature Importance (Permutation Method)",
        save_path=output_dir / "permutation_importance.png"
    )
    
    # 2. Feature Ablation
    ablation_results = analyze_feature_ablation(
        model, 
        env, 
        feature_names=feature_names,
        num_episodes=num_episodes
    )
    
    print("\nFeature group importance:")
    print(ablation_results)
    
    # Save results
    ablation_results.to_csv(output_dir / "feature_ablation.csv", index=False)
    
    # 3. SHAP Analysis (if not too many samples)
    try:
        shap_values, observations, feature_names = calculate_shap_values(
            model, 
            env, 
            num_samples=100,
            feature_names=feature_names
        )
        
        # Plot SHAP summary
        plot_shap_summary(
            shap_values, 
            observations, 
            feature_names,
            save_path=output_dir / "shap_summary.png"
        )
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
    
    print(f"\nFeature importance analysis complete. Results saved to {output_dir}")
    
    return {
        "permutation_importance": permutation_results,
        "feature_ablation": ablation_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature importance of a trained RL model")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save analysis results")
    parser.add_argument('--episodes', type=int, default=5, help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    analyze_feature_importance(
        model_path=args.model,
        output_dir=args.output_dir,
        num_episodes=args.episodes
    )
