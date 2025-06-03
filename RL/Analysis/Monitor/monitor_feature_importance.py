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
import shap
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Data.Utils.loader import load_data_from_db
from RL.Envs.trading_env import TradingEnv
from RL.Envs.Components.observation_normalizer import ObservationNormalizer


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
    base_modified_env = TradingEnv(
        unwrapped_env.data.copy(),
        normalize_observations=True,
        enable_adaptive_scaling=False
    )
    
    # Wrap with action converter
  
    
    # Store original step function
    original_step = base_modified_env.step
    original_get_obs = base_modified_env._get_observation
    
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
    base_modified_env._get_observation = modified_get_observation
    
    return base_modified_env


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
    base_masked_env = TradingEnv(
        unwrapped_env.data.copy(),
        normalize_observations=True,
        enable_adaptive_scaling= False
    )
    
    # Wrap with action converter
   
    
    # Store original get observation function
    original_get_obs = base_masked_env._get_observation
    
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
    base_masked_env._get_observation = modified_get_observation
    
    return base_masked_env

def calculate_gradient_importance(model, env, observations, feature_names):
    """
    Alternative feature importance calculation using gradient-based methods
    """
    try:
        import torch
        
        print("Calculating gradient-based feature importance...")
        
        # Get device from model
        device = next(model.policy.parameters()).device
        
        # Calculate gradients for feature importance
        importance_scores = []
        
        for obs in tqdm(observations[:50], desc="Calculating gradients"):  # Limit to 50 samples
            try:
                # Convert to tensor and enable gradients
                obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(device)
                obs_tensor.requires_grad_(True)
                
                # Forward pass through the policy network
                with torch.enable_grad():
                    # Get the policy logits
                    features = model.policy.extract_features(obs_tensor)
                    logits = model.policy.action_net(features)
                    
                    # Calculate the sum of absolute gradients for each feature
                    grad_outputs = torch.ones_like(logits)
                    gradients = torch.autograd.grad(
                        outputs=logits,
                        inputs=obs_tensor,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True
                    )[0]
                    
                    importance_scores.append(torch.abs(gradients).cpu().numpy().flatten())
                    
            except Exception as e:
                print(f"Warning: Error calculating gradients for sample: {e}")
                # Use zeros as fallback
                importance_scores.append(np.zeros(len(feature_names)))
        
        # Average importance across all samples
        avg_importance = np.mean(importance_scores, axis=0)
        
        # Create mock SHAP values format
        shap_values = avg_importance.reshape(1, -1)  # Single row
        eval_observations = observations[:1]  # Single observation
        
        return shap_values, eval_observations, feature_names
        
    except Exception as e:
        print(f"Error in gradient-based importance: {e}")
        return None, None, None


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
    model = PPO.load(model_path, custom_objects={'learning_rate': 0.0003})
    
    # Load evaluation data if not provided
    if data is None:
        print("Loading data from database...")
        data = load_data_from_db()
        # Use only validation portion (20%)
        split_idx = int(len(data) * 0.5)
        data = data[split_idx:]
    
    print(f"Using {len(data)} records for evaluation")
      # Define feature names based on the actual features in your trading environment
    feature_names = [
        # Base pattern features (7 features)
        "probability", "action", "reward_risk_ratio", "max_gain",
        "max_drawdown", "mse", "expected_value",
        # Technical indicators (3 features)
        "rsi", "atr", "atr_ratio",
        # Sentiment features (2 features)
        "unified_sentiment", "sentiment_count",
        # COT data (6 features)
        "net_noncommercial", "net_nonreportable",
        "change_nonrept_long", "change_nonrept_short",
        "change_noncommercial_long", "change_noncommercial_short",
        # Time features (7 features)
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "asian_session", "london_session", "ny_session",
        # Portfolio features (6 features)
        "balance_ratio", "position_ratio", "position", "portfolio_max_drawdown", "win_rate", "steps_without_action"
    ]
      # Create evaluation environment
    base_env = TradingEnv(
        data,
        normalize_observations=True,
        enable_adaptive_scaling=False
    )
    # Wrap with action converter for compatibility with the trained model
    
    
    # # 1. Permutation Importance
    # permutation_results = calculate_permutation_importance(
    #     model, 
    #     base_env, 
    #     feature_names=feature_names,
    #     num_episodes=num_episodes
    # )
    
    # print("\nTop 10 most important features (permutation method):")
    # print(permutation_results.head(10))
    
    # # Save results
    # permutation_results.to_csv(output_dir / "permutation_importance.csv", index=False)
    
    # # Plot permutation importance
    # plot_feature_importance(
    #     permutation_results, 
    #     title="Feature Importance (Permutation Method)",
    #     save_path=output_dir / "permutation_importance.png"
    # )
    
    # 2. Feature Ablation
    ablation_results = analyze_feature_ablation(
        model, 
        base_env, 
        feature_names=feature_names,
        num_episodes=num_episodes
    )
    
    print("\nFeature group importance:")
    print(ablation_results)
    
    # Save results
    ablation_results.to_csv(output_dir / "feature_ablation.csv", index=False)
      # 3. SHAP Analysis (with proper error handling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature importance of a trained RL model")
    parser.add_argument('--model', type=str,default="RL/Models/Experiments/best_model_continued.zip", help="Path to the trained model")
    parser.add_argument('--output-dir', type=str, default="Images/RL/Analysis", help="Directory to save analysis results")
    parser.add_argument('--episodes', type=int, default=5, help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    analyze_feature_importance(
        model_path=args.model, 
        output_dir=args.output_dir,
        num_episodes=args.episodes
    )
