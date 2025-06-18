"""
Environment Checker Script

This script tests your trading environment for compatibility with Stable Baselines 3
and checks for common issues that might cause training to hang.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
import traceback
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from datetime import datetime

# Import environment modules
from RL.Data.Utils.loader import load_data_from_db
from RL.Envs.trading_env import TradingEnv
from RL.Envs.trading_env_v2 import TradingEnvV2



# Import SB3 checker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

def visualize_env_metrics(env, num_steps=1000):
    """
    Run an episode with random actions and visualize the rewards and portfolio metrics
    to analyze the correlation between rewards and performance.
    
    Args:
        env: The trading environment instance
        num_steps: Maximum number of steps to run
    
    Returns:
        DataFrame with all the collected metrics
    """
    print("\n========== VISUALIZING ENVIRONMENT METRICS ==========\n")
    
    # Import required libraries for enhanced visualizations
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
        print("Seaborn not found. Falling back to basic visualizations.")
    
    # Initialize tracking variables with expanded metrics
    metrics = {
        'step': [],
        'equity': [], 
        'balance': [],
        'drawdown': [],
        'reward': [],
        'unrealized_pnl': [],
        'action_type': [],
        'position_size': [],
        'risk_reward': [],
        'win_rate': [],
        'trade_count': [],
        'active_positions': [],
        'tp_exits': [],
        'sl_exits': [],
        'time_exits': [],
        'profit_factor': [],
        # Additional metrics for more detailed analysis
        'position_value': [],
        'margin_used': [],
        'available_margin': [],
        'new_position_opened': [],
        'trade_closed': [],
        # Reward component tracking for combined reward type
        'base_reward': [],
        'sharpe_adjustment': [],
        'drawdown_penalty': [],
        'win_rate_bonus': [],
        'consistency_bonus': [],
        'hold_penalty': [],
        # Component contribution percentages
        'base_reward_pct': [],
        'sharpe_adjustment_pct': [],
        'drawdown_penalty_pct': [],
        'win_rate_bonus_pct': [],
        'consistency_bonus_pct': [],
        'hold_penalty_pct': []
    }
    
    # Reset environment
    obs, _ = env.reset()
    
    # Check if we're using combined reward to track components
    is_combined_reward = hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined'
    
    # Run episode
    print("Running episode with random actions...")
    for step in range(num_steps):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Extract info for plotting
        metrics['step'].append(step)
        metrics['reward'].append(reward)
        
        # Portfolio metrics
        metrics['equity'].append(info.get('equity', 0))
        metrics['balance'].append(info.get('cash_balance', 0))
        metrics['drawdown'].append(info.get('drawdown', 0))
        metrics['unrealized_pnl'].append(info.get('unrealized_pnl', 0))
        
        # Action and position metrics
        metrics['action_type'].append(info.get('action_type', 0))
        metrics['position_size'].append(info.get('position_size', 0))
        metrics['risk_reward'].append(info.get('risk_reward', 0))
        
        # Trade metrics
        metrics['win_rate'].append(info.get('win_rate', 0))
        metrics['trade_count'].append(info.get('trade_count', 0))
        metrics['active_positions'].append(info.get('active_positions', 0))
        metrics['tp_exits'].append(info.get('tp_exits', 0))
        metrics['sl_exits'].append(info.get('sl_exits', 0))
        metrics['time_exits'].append(info.get('time_exits', 0))
        metrics['profit_factor'].append(info.get('profit_factor', 1.0))
        
        # Additional metrics
        metrics['position_value'].append(info.get('position_value', 0))
        metrics['margin_used'].append(info.get('margin_used', 0))
        metrics['available_margin'].append(info.get('available_margin', 0))
        metrics['new_position_opened'].append(info.get('new_position', False))
        metrics['trade_closed'].append(info.get('closed_positions_count', 0) > 0)
        
        # Track reward components if using combined reward
        if is_combined_reward:
            # Calculate estimated components
            # Base reward estimate (from equity change)
            equity_change_pct = 0
            if step > 0:
                equity_change_pct = (metrics['equity'][-1] - metrics['equity'][-2]) / metrics['equity'][-2] if metrics['equity'][-2] != 0 else 0
            
            # Base reward calculation (more accurate estimate)
            base_reward = equity_change_pct * 10  # Scale factor based on environment configuration
            metrics['base_reward'].append(base_reward)
            
            # Component estimates based on RewardCalculator implementation
            # Sharpe adjustment (using reward calculator's formula)
            sharpe_adj = 0
            if hasattr(env.reward_calculator, 'trading_state') and env.reward_calculator.trading_state.returns_history:
                returns = np.array(env.reward_calculator.trading_state.returns_history)
                mean_return = np.mean(returns) if len(returns) > 0 else 0
                std_return = np.std(returns) if len(returns) > 1 else 1e-6
                sharpe = mean_return / max(std_return, 1e-6)
                # Formula from RewardCalculator.calculate_sharpe_reward multiplied by 0.5 (from combined reward)
                sharpe_adj = base_reward * min(1, max(-0.5, sharpe * 0.2)) * 0.5
            metrics['sharpe_adjustment'].append(sharpe_adj)
            
            # Drawdown penalty
            dd_penalty = 0
            if hasattr(env.reward_calculator, 'trading_state'):
                # Formula from RewardCalculator.calculate_drawdown_penalty with 0.1 scaling from combined reward
                dd_penalty = min(0, env.reward_calculator.trading_state.drawdown * 2 * 0.1)
            metrics['drawdown_penalty'].append(dd_penalty)
            
            # Win rate bonus
            wr_bonus = 0
            if hasattr(env.reward_calculator, 'trading_state') and env.reward_calculator.trading_state.trade_count > 0:
                win_rate = env.reward_calculator.trading_state.get_win_rate()
                # Formula from RewardCalculator.calculate_win_rate_bonus
                wr_bonus = (win_rate - 0.5) * 0.1
            metrics['win_rate_bonus'].append(wr_bonus)
            
            # Consistency bonus
            consist_bonus = 0
            if hasattr(env.reward_calculator, 'trading_state') and len(env.reward_calculator.trading_state.returns_history) >= 3:
                returns = np.array(env.reward_calculator.trading_state.returns_history)
                mean_return = np.mean(returns)
                consistency = np.mean(returns > 0)
                # Simplified formula from RewardCalculator.calculate_consistency_bonus
                if mean_return > 0:
                    consist_bonus = consistency * 0.05
                else:
                    # For negative mean returns, calculate trend 
                    trend = np.corrcoef(returns, np.arange(len(returns)))[0, 1]
                    consist_bonus = -abs(trend) * 0.05
            metrics['consistency_bonus'].append(consist_bonus)
            
            # Hold penalty estimate
            hold_penalty = 0
            if info.get('action_type') == 0 and step >= 10:  # HOLD action and after initial period
                steps_without_action = 1
                for s in range(step-1, max(0, step-20), -1):
                    if metrics['action_type'][s] == 0:
                        steps_without_action += 1
                    else:
                        break
                # Formula from RewardCalculator.calculate_hold_penalty
                if steps_without_action >= 10:
                    hold_penalty = -0.0001 * (steps_without_action - 9)
                    hold_penalty = max(-0.005, hold_penalty)
            metrics['hold_penalty'].append(hold_penalty)
            
            # Calculate component contribution percentages to better understand importance
            total_contribution = abs(base_reward) + abs(sharpe_adj) + abs(dd_penalty) + abs(wr_bonus) + abs(consist_bonus) + abs(hold_penalty)
            if total_contribution > 0:
                metrics['base_reward_pct'].append(abs(base_reward) / total_contribution * 100)
                metrics['sharpe_adjustment_pct'].append(abs(sharpe_adj) / total_contribution * 100)
                metrics['drawdown_penalty_pct'].append(abs(dd_penalty) / total_contribution * 100)
                metrics['win_rate_bonus_pct'].append(abs(wr_bonus) / total_contribution * 100)
                metrics['consistency_bonus_pct'].append(abs(consist_bonus) / total_contribution * 100)
                metrics['hold_penalty_pct'].append(abs(hold_penalty) / total_contribution * 100)
            else:
                metrics['base_reward_pct'].append(0)
                metrics['sharpe_adjustment_pct'].append(0)
                metrics['drawdown_penalty_pct'].append(0)
                metrics['win_rate_bonus_pct'].append(0)
                metrics['consistency_bonus_pct'].append(0)
                metrics['hold_penalty_pct'].append(0)
        
            # Verify component sum approximately equals reward (debugging sanity check)
            component_sum = base_reward * (1 + sharpe_adj * 0.5 / base_reward if base_reward != 0 else 0) + dd_penalty + wr_bonus + consist_bonus + hold_penalty
            # Print debugging info if components don't sum approximately to reward (more than 10% difference)
            if abs(component_sum - reward) / (abs(reward) + 1e-6) > 0.1 and step % 50 == 0:
                print(f"Step {step}: Estimated reward components sum ({component_sum:.6f}) differs from actual reward ({reward:.6f})")
            
        else:
            # If not combined reward, add zeros for components
            metrics['base_reward'].append(0)
            metrics['sharpe_adjustment'].append(0)
            metrics['drawdown_penalty'].append(0)
            metrics['win_rate_bonus'].append(0)
            metrics['consistency_bonus'].append(0)
            metrics['hold_penalty'].append(0)
            metrics['base_reward_pct'].append(0)
            metrics['sharpe_adjustment_pct'].append(0)
            metrics['drawdown_penalty_pct'].append(0)
            metrics['win_rate_bonus_pct'].append(0)
            metrics['consistency_bonus_pct'].append(0)
            metrics['hold_penalty_pct'].append(0)
        
        # Break if episode is done
        if done or truncated:
            print(f"Episode ended after {step+1} steps")
            break
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metrics)
    
    # Calculate derived metrics
    df['cumulative_reward'] = df['reward'].cumsum()
    df['reward_rolling_mean'] = df['reward'].rolling(window=20, min_periods=1).mean()
    df['equity_change'] = df['equity'].pct_change().fillna(0)
    df['equity_pct_change'] = df['equity_change'] * 100  # For better visualization
    df['action_type_str'] = df['action_type'].map({0: 'HOLD', 1: 'BUY', 2: 'SELL'})
    
    # Calculate forward returns (1, 5, 10 steps ahead) to analyze reward predictive power
    df['equity_change_fwd_1'] = df['equity'].pct_change(periods=1).shift(-1).fillna(0)
    df['equity_change_fwd_5'] = df['equity'].pct_change(periods=5).shift(-5).fillna(0)
    df['equity_change_fwd_10'] = df['equity'].pct_change(periods=10).shift(-10).fillna(0)
    
    # Create main visualization
    print("Creating basic visualizations...")
    
    plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=plt.gcf())
    
    # 1. Equity and Balance
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(df['step'], df['equity'], label='Equity', linewidth=2)
    ax1.plot(df['step'], df['balance'], label='Cash Balance', linestyle='--')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = plt.subplot(gs[0, 1])
    ax2.fill_between(df['step'], df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
    ax2.set_title('Drawdown Over Time')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(min(df['drawdown'])*1.2, 0.05)  # Leave some space above zero
    ax2.grid(True, alpha=0.3)
    
    # 3. Rewards with colored background for trade events
    ax3 = plt.subplot(gs[1, 0])
    # Highlight background for trade events
    for i, (closed, opened) in enumerate(zip(df['trade_closed'], df['new_position_opened'])):
        if closed:
            ax3.axvspan(i-0.5, i+0.5, alpha=0.2, color='blue')
        elif opened:
            ax3.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
    
    ax3.plot(df['step'], df['reward'], label='Reward', alpha=0.5, linewidth=1, color='gray')
    ax3.plot(df['step'], df['reward_rolling_mean'], label='Reward (20-step MA)', linewidth=2, color='blue')
    ax3.set_title('Reward Signal (Blue bg = Trade Closed, Green bg = Position Opened)')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relationship between Equity Change and Rewards
    ax4 = plt.subplot(gs[1, 1])
    ax4.scatter(df['equity_pct_change'], df['reward'], alpha=0.6, 
               c=df['action_type'].map({0: 'gray', 1: 'green', 2: 'red'}))
    
    # Calculate correlation for title
    equity_reward_corr = df['equity_pct_change'].corr(df['reward'])
    ax4.set_title(f'Reward vs. Equity Change (Correlation: {equity_reward_corr:.4f})')
    ax4.set_xlabel('Equity Change (%)')
    ax4.set_ylabel('Reward')
    ax4.grid(True, alpha=0.3)
    
    # Add trendline
    try:
        from scipy import stats
        # Remove outliers for better trend visualization
        mask = (abs(df['equity_pct_change']) < df['equity_pct_change'].std() * 3)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df.loc[mask, 'equity_pct_change'], 
            df.loc[mask, 'reward']
        )
        x_vals = np.linspace(df['equity_pct_change'].min(), df['equity_pct_change'].max(), 100)
        ax4.plot(x_vals, slope * x_vals + intercept, '--', color='black')
    except:
        pass
    
    # 5. Trade Metrics with Reward Overlay
    ax5 = plt.subplot(gs[2, 0])
    ax5.plot(df['step'], df['win_rate'], label='Win Rate', linewidth=2)
    ax5_2 = ax5.twinx()  # Create second y-axis
    ax5_2.plot(df['step'], df['reward_rolling_mean'], label='Reward (MA)', 
              color='purple', alpha=0.7, linewidth=1.5, linestyle=':')
    ax5.set_title('Trade Performance vs Reward')
    ax5.set_ylabel('Win Rate')
    ax5_2.set_ylabel('Reward MA', color='purple')
    ax5.legend(loc='upper left')
    ax5_2.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Reward Distribution by Action Type
    ax6 = plt.subplot(gs[2, 1])
    
    # Create boxplots for reward by action type
    action_types = sorted(df['action_type'].unique())
    action_labels = ['HOLD', 'BUY', 'SELL']
    reward_by_action = [df[df['action_type'] == act]['reward'] for act in action_types]
    
    ax6.boxplot(reward_by_action, labels=[action_labels[act] for act in action_types])
    ax6.set_title('Reward Distribution by Action Type')
    ax6.set_ylabel('Reward')
    ax6.grid(True, alpha=0.3)
    
    # Add means as text
    for i, act in enumerate(action_types):
        mean_reward = df[df['action_type'] == act]['reward'].mean()
        ax6.text(i+1, mean_reward, f"{mean_reward:.5f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # 7. Actions and Positions
    ax7 = plt.subplot(gs[3, 0])
    # Convert action_type to categorical for better visualization
    action_colors = {0: 'gray', 1: 'green', 2: 'red'}  # HOLD, BUY, SELL
    action_colors_list = [action_colors.get(a, 'gray') for a in df['action_type']]
    
    # Create scatter plot with rewards as color intensity
    reward_normalized = (df['reward'] - df['reward'].min()) / (df['reward'].max() - df['reward'].min() + 1e-10)
    sizes = 30 + 100 * reward_normalized  # Vary point size by reward
    
    ax7.scatter(df['step'], df['position_size'], c=action_colors_list, 
               s=sizes, alpha=0.7)
    ax7.plot(df['step'], df['active_positions']/env.max_positions, 
            label=f'Active Positions (Max: {env.max_positions})', linewidth=2)
    ax7.set_title('Actions and Position Sizes (Point Size = Reward)')
    ax7.set_ylabel('Position Size / Utilization')
    ax7.set_ylim(0, 1.1)
    
    # Create custom legend for action colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='HOLD', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='BUY', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='SELL', markersize=8),
    ]
    ax7.legend(handles=legend_elements + [Line2D([0], [0], color='blue', label='Active Positions')], 
              loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    # 8. Enhanced correlation analysis between key metrics and reward
    ax8 = plt.subplot(gs[3, 1])
    
    # Calculate correlations with reward with more metrics
    metrics_to_check = [
        'equity_change', 'drawdown', 'unrealized_pnl', 'active_positions', 
        'position_size', 'win_rate', 'equity_change_fwd_1', 'equity_change_fwd_5'
    ]
    metrics_labels = [
        'Equity Change', 'Drawdown', 'Unrealized PnL', 'Active Positions',
        'Position Size', 'Win Rate', 'Future Equity +1', 'Future Equity +5'
    ]
    
    # Calculate both Pearson and Spearman correlations
    pearson_correlations = {}
    spearman_correlations = {}
    
    for metric in metrics_to_check:
        pearson_correlations[metric] = df[metric].corr(df['reward'], method='pearson')
        spearman_correlations[metric] = df[metric].corr(df['reward'], method='spearman')
    
    # Prepare bar plot of correlations
    correlation_values = list(pearson_correlations.values())
    
    colors = ['green' if c > 0 else 'red' for c in correlation_values]
    bars = ax8.barh(metrics_labels, correlation_values, color=colors)
    ax8.set_title('Correlation with Reward Signal')
    ax8.set_xlabel('Correlation Coefficient')
    ax8.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax8.grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, (v, s) in enumerate(zip(correlation_values, spearman_correlations.values())):
        ax8.text(v + (0.02 if v >= 0 else -0.12), 
                i, 
                f"P:{v:.3f} S:{s:.3f}", 
                color='black',
                va='center',
                fontweight='bold',
                fontsize=8)
    
    # Set x-axis limits to ensure values are visible
    ax8.set_xlim(min(min(correlation_values) - 0.15, -0.15), 
                max(max(correlation_values) + 0.15, 0.15))
    
    # Final layout adjustments for first figure
    plt.suptitle('Trading Environment Metrics Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig('env_metrics_analysis_1.png', dpi=150, bbox_inches='tight')
    print("Basic visualization saved to 'env_metrics_analysis_1.png'")

    # Create second figure with advanced analyses
    print("Creating advanced visualizations...")
    
    # Create a second figure with advanced reward analysis
    plt.figure(figsize=(20, 22))
    gs2 = GridSpec(4, 2, figure=plt.gcf())
    
    # 1. Reward Heatmap by Action Type and Position Size
    ax1 = plt.subplot(gs2[0, 0])
    
    # Create 2D grid for position sizes and action types
    if has_seaborn:
        # Group data by action type and position size rounded to nearest 0.05
        df['position_size_group'] = (df['position_size'] * 20).round() / 20
        pivot = df.pivot_table(
            values='reward', 
            index='position_size_group', 
            columns='action_type_str', 
            aggfunc='mean'
        ).fillna(0)
        
        # Create heatmap
        sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=True, fmt='.4f', ax=ax1)
        ax1.set_title('Average Reward by Action Type and Position Size')
        ax1.set_ylabel('Position Size')
    else:
        ax1.text(0.5, 0.5, "Seaborn required for this plot", 
                ha='center', va='center', fontsize=14)
    
    # 2. Reward vs. Drawdown Scatter
    ax2 = plt.subplot(gs2[0, 1])
    scatter = ax2.scatter(df['drawdown'], df['reward'], 
                         c=df['action_type'], cmap='viridis', 
                         alpha=0.7)
    ax2.set_title('Reward vs. Drawdown by Action Type')
    ax2.set_xlabel('Drawdown')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    legend1 = ax2.legend(*scatter.legend_elements(),
                        loc="upper right", title="Action Type")
    ax2.add_artist(legend1)
    
    # 3. Time-lagged Correlation Analysis
    ax3 = plt.subplot(gs2[1, 0])
    
    # Calculate time-lagged correlations between reward and future equity changes
    lags = range(1, 11)  # 1 to 10 steps ahead
    correlations = []
    
    for lag in lags:
        future_equity_change = df['equity'].pct_change(periods=lag).shift(-lag).fillna(0)
        correlation = df['reward'].corr(future_equity_change)
        correlations.append(correlation)
    
    ax3.plot(lags, correlations, marker='o', linestyle='-', linewidth=2)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax3.set_title('Correlation Between Current Reward and Future Equity Changes')
    ax3.set_xlabel('Steps Ahead')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, v in enumerate(correlations):
        ax3.text(i + 1, v + (0.02 if v >= 0 else -0.02), 
                f"{v:.3f}", 
                ha='center',
                va='bottom' if v >= 0 else 'top',
                fontsize=9)
    
    # 4. Reward per Trade Outcome Type
    ax4 = plt.subplot(gs2[1, 1])
    
    # Calculate step indices where exits occurred
    tp_steps = []
    sl_steps = []
    time_steps = []
    
    for i in range(1, len(df)):
        if df['tp_exits'][i] > df['tp_exits'][i-1]:
            tp_steps.append(i)
        if df['sl_exits'][i] > df['sl_exits'][i-1]:
            sl_steps.append(i)
        if df['time_exits'][i] > df['time_exits'][i-1]:
            time_steps.append(i)
    
    # Extract rewards for each exit type
    tp_rewards = df.loc[tp_steps, 'reward'] if tp_steps else []
    sl_rewards = df.loc[sl_steps, 'reward'] if sl_steps else []
    time_rewards = df.loc[time_steps, 'reward'] if time_steps else []
    
    # Create boxplot
    box_data = [tp_rewards, sl_rewards, time_rewards]
    box_labels = ['Take Profit', 'Stop Loss', 'Time Exit']
    
    # Filter out empty datasets
    valid_data = [(data, label) for data, label in zip(box_data, box_labels) if len(data) > 0]
    if valid_data:
        valid_box_data, valid_box_labels = zip(*valid_data)
        ax4.boxplot(valid_box_data, labels=valid_box_labels)
        ax4.set_title('Reward Distribution by Exit Type')
        ax4.set_ylabel('Reward')
        
        # Add mean values as text
        for i, data in enumerate(valid_box_data):
            mean_val = np.mean(data)
            ax4.text(i + 1, mean_val, f"{mean_val:.4f}", 
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "Insufficient trade exit data", 
                ha='center', va='center', fontsize=14)
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward Components Analysis (if reward_type is 'combined')
    ax5 = plt.subplot(gs2[2, 0])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Display reward components over time
        steps_display = min(100, len(df))  # Display only the first N steps for clarity
        
        # Get reward component data
        components = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        component_data = [df[comp].iloc[:steps_display].values for comp in components]
        component_labels = [comp.replace('_', ' ').title() for comp in components]
        
        # Plot each component line
        for i, (comp, data) in enumerate(zip(component_labels, component_data)):
            ax5.plot(range(steps_display), data, label=comp, linewidth=1.5)
            
        # Plot total reward line
        ax5.plot(range(steps_display), df['reward'].iloc[:steps_display], 'k-', linewidth=2, label='Total Reward')
        
        ax5.set_title('Reward Components Analysis Over Time')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward Component Value')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Add zero line for reference
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "Combined reward type required\nfor components analysis", 
                ha='center', va='center', fontsize=14)
    
    # 6. Detailed Correlation Heatmap
    ax6 = plt.subplot(gs2[2, 1])
    
    if has_seaborn:
        # Select relevant columns for correlation analysis
        corr_columns = ['reward', 'equity_change', 'drawdown', 'unrealized_pnl', 
                      'active_positions', 'position_size', 'win_rate']
        
        # Create correlation matrix
        corr_matrix = df[corr_columns].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax6)
        ax6.set_title('Correlation Matrix of Trading Metrics')
    else:
        ax6.text(0.5, 0.5, "Seaborn required for this plot", 
                ha='center', va='center', fontsize=14)
    
    # 7. Time-Series Rolling Correlation between Components and Performance
    ax7 = plt.subplot(gs2[3, 0])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Calculate rolling correlations between components and equity change
        window_size = min(30, len(df) // 3)  # Use a reasonable window size
        
        if len(df) > window_size * 2:
            components = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus']
            component_labels = [c.replace('_', ' ').title() for c in components]
              # Calculate rolling correlations with safe handling
            rolling_correlations = {}
            for comp in components:
                # Use pandas built-in rolling correlation but handle NaN values
                rolling_corr = df[comp].rolling(window=window_size).corr(df['equity_change'])
                rolling_correlations[comp] = rolling_corr.fillna(0)  # Replace NaN with zero
            
            # Plot rolling correlations
            for comp, label in zip(components, component_labels):
                ax7.plot(range(len(rolling_correlations[comp])), rolling_correlations[comp], 
                        label=label, linewidth=1.5)
            
            ax7.set_title(f'Rolling Correlation ({window_size}-step) Between Components and Equity Change')
            ax7.set_xlabel('Step')
            ax7.set_ylabel('Correlation Coefficient')
            ax7.legend(loc='best', fontsize=9)
            ax7.grid(True, alpha=0.3)
            
            # Add horizontal line at correlation=0
            ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, "Insufficient data for rolling correlation", 
                    ha='center', va='center', fontsize=14)
    else:
        ax7.text(0.5, 0.5, "Combined reward type required for rolling correlation analysis", 
                ha='center', va='center', fontsize=14)
    
    # 8. Component Impact on Different Time Horizons
    ax8 = plt.subplot(gs2[3, 1])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Calculate correlations between components and future equity changes at different time horizons
        components = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        component_labels = [c.split('_')[0].capitalize() for c in components]  # Shorter labels for plot
        horizons = [1, 3, 5, 10]
        
        # Create matrix to store correlations
        correlation_matrix = np.zeros((len(components), len(horizons)))
          # Calculate correlations using safe method to avoid division by zero warnings
        for i, comp in enumerate(components):
            for j, horizon in enumerate(horizons):
                future_col = f'equity_change_fwd_{horizon}'
                if future_col in df.columns:
                    correlation_matrix[i, j] = safe_corrcoef(df[comp].values, df[future_col].values)
        
        # Create heatmap
        if has_seaborn:
            sns.heatmap(
                correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f', 
                ax=ax8,
                xticklabels=[f't+{h}' for h in horizons],
                yticklabels=component_labels
            )
            ax8.set_title('Component Correlation with Future Equity Changes')
            ax8.set_xlabel('Time Horizon')
            ax8.set_ylabel('Reward Component')
        else:
            # Fallback if no seaborn
            im = ax8.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax8.set_xticks(range(len(horizons)))
            ax8.set_xticklabels([f't+{h}' for h in horizons])
            ax8.set_yticks(range(len(components)))
            ax8.set_yticklabels(component_labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax8)
            
            # Add text annotations
            for i in range(len(components)):
                for j in range(len(horizons)):
                    ax8.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black")
            
            ax8.set_title('Component Correlation with Future Equity Changes')
            ax8.set_xlabel('Time Horizon')
            ax8.set_ylabel('Reward Component')
    else:
        ax8.text(0.5, 0.5, "Combined reward type required for component analysis", 
                ha='center', va='center', fontsize=14)
    
    # Final layout adjustments for second figure
    plt.suptitle('Advanced Reward Engineering Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig('env_metrics_analysis_2.png', dpi=150, bbox_inches='tight')
    print("Advanced visualization saved to 'env_metrics_analysis_2.png'")
    
    # Create third figure specifically for reward component analysis
    print("Creating reward component analysis visualization...")
    
    plt.figure(figsize=(20, 20))
    gs3 = GridSpec(4, 2, figure=plt.gcf())
    
    # 1. Reward Component Contribution Stack Plot
    ax1 = plt.subplot(gs3[0, :])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Limit to a reasonable number of steps for clarity
        steps_display = min(100, len(df))
        step_indices = range(steps_display)
        
        # Extract component data
        base = df['base_reward'].iloc[:steps_display].values
        sharpe = df['sharpe_adjustment'].iloc[:steps_display].values
        drawdown = df['drawdown_penalty'].iloc[:steps_display].values
        win_rate = df['win_rate_bonus'].iloc[:steps_display].values
        consistency = df['consistency_bonus'].iloc[:steps_display].values
        hold_penalty = df['hold_penalty'].iloc[:steps_display].values
        total = df['reward'].iloc[:steps_display].values
        
        # Calculate positive and negative components separately for better visualization
        pos_mask = base > 0
        neg_mask = base < 0
        
        # Create stacked area plot with separated positive and negative contributions
        ax1.fill_between(step_indices, 0, np.where(pos_mask, base, 0), 
                        label='Base Reward (+)', alpha=0.7, color='#1f77b4')
        ax1.fill_between(step_indices, 0, np.where(neg_mask, base, 0), 
                        label='Base Reward (-)', alpha=0.7, color='#1f77b4', hatch='/')
        
        # Stack sharpe on top of base reward
        current_pos = np.where(pos_mask, base, 0)
        current_neg = np.where(neg_mask, base, 0)
        
        # Add sharpe adjustment
        sharpe_pos = np.where(sharpe > 0, sharpe, 0)
        sharpe_neg = np.where(sharpe < 0, sharpe, 0)
        
        ax1.fill_between(step_indices, current_pos, current_pos + sharpe_pos, 
                        label='Sharpe Adjustment (+)', alpha=0.7, color='#ff7f0e')
        ax1.fill_between(step_indices, current_neg, current_neg + sharpe_neg, 
                        label='Sharpe Adjustment (-)', alpha=0.7, color='#ff7f0e', hatch='/')
        
        current_pos += sharpe_pos
        current_neg += sharpe_neg
        
        # Add drawdown penalty (always negative)
        ax1.fill_between(step_indices, current_neg, current_neg + drawdown, 
                        label='Drawdown Penalty', alpha=0.7, color='#2ca02c', hatch='/')
        
        current_neg += drawdown
        
        # Add win rate bonus (can be positive or negative)
        wr_pos = np.where(win_rate > 0, win_rate, 0)
        wr_neg = np.where(win_rate < 0, win_rate, 0)
        
        ax1.fill_between(step_indices, current_pos, current_pos + wr_pos, 
                        label='Win Rate Bonus (+)', alpha=0.7, color='#d62728')
        ax1.fill_between(step_indices, current_neg, current_neg + wr_neg, 
                        label='Win Rate Bonus (-)', alpha=0.7, color='#d62728', hatch='/')
        
        current_pos += wr_pos
        current_neg += wr_neg
        
        # Add consistency bonus (can be positive or negative)
        consist_pos = np.where(consistency > 0, consistency, 0)
        consist_neg = np.where(consistency < 0, consistency, 0)
        
        ax1.fill_between(step_indices, current_pos, current_pos + consist_pos, 
                        label='Consistency Bonus (+)', alpha=0.7, color='#9467bd')
        ax1.fill_between(step_indices, current_neg, current_neg + consist_neg, 
                        label='Consistency Bonus (-)', alpha=0.7, color='#9467bd', hatch='/')
                        
        # Add hold penalty (always negative or zero)
        ax1.fill_between(step_indices, current_neg, current_neg + hold_penalty, 
                        label='Hold Penalty', alpha=0.7, color='#8c564b', hatch='/')
        
        # Plot total reward as a line
        ax1.plot(step_indices, total, 'k-', linewidth=2, label='Total Reward')
        
        # Add horizontal line at y=0
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax1.set_title('Reward Component Contributions Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward Value')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Combined reward type required for components analysis", 
                ha='center', va='center', fontsize=14)
    
    # 2. Component Contribution Pie Chart
    ax2 = plt.subplot(gs3[1, 0])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Calculate absolute contribution of each component
        components = ['Base Reward', 'Sharpe Adjustment', 'Drawdown Penalty', 'Win Rate Bonus', 'Consistency Bonus', 'Hold Penalty']
        component_abs_values = [
            df['base_reward'].abs().sum(),
            df['sharpe_adjustment'].abs().sum(),
            df['drawdown_penalty'].abs().sum(),
            df['win_rate_bonus'].abs().sum(),
            df['consistency_bonus'].abs().sum(),
            df['hold_penalty'].abs().sum()
        ]
        
        # Create pie chart of absolute contributions
        wedges, texts, autotexts = ax2.pie(
            component_abs_values, 
            labels=components, 
            autopct='%1.1f%%', 
            startangle=90,
            explode=[0.05, 0, 0, 0, 0, 0],  # Slightly explode the largest slice
            shadow=True
        )
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            
        ax2.set_title('Absolute Contribution of Reward Components')
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    else:
        ax2.text(0.5, 0.5, "Combined reward type required for components analysis", 
                ha='center', va='center', fontsize=14)
    
    # 3. Component Correlation with Final Reward
    ax3 = plt.subplot(gs3[1, 1])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Calculate correlations between components and total reward
        component_columns = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        component_labels = ['Base Reward', 'Sharpe Adj', 'Drawdown Penalty', 'Win Rate Bonus', 'Consistency Bonus', 'Hold Penalty']
        correlations = []
        for comp in component_columns:
            correlations.append(safe_corrcoef(df[comp].values, df['reward'].values))
        
        # Create bar plot
        colors = ['green' if c > 0 else 'red' for c in correlations]
        bars = ax3.barh(component_labels, correlations, color=colors)
        ax3.set_title('Component Correlation with Total Reward')
        ax3.set_xlabel('Correlation Coefficient')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Add correlation values as text
        for i, v in enumerate(correlations):
            ax3.text(v + (0.02 if v >= 0 else -0.08), 
                    i, 
                    f"{v:.4f}", 
                    color='black',
                    va='center',
                    fontweight='bold')
        
        # Set x-axis limits
        ax3.set_xlim(min(min(correlations) - 0.1, -0.1), 
                    max(max(correlations) + 0.1, 0.1))
    else:
        ax3.text(0.5, 0.5, "Combined reward type required for components analysis", 
                ha='center', va='center', fontsize=14)
    
    # 4. Component Contributions by Action Type
    ax4 = plt.subplot(gs3[2, 0])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Calculate average component values by action type
        action_types = [0, 1, 2]  # HOLD, BUY, SELL
        action_labels = ['HOLD', 'BUY', 'SELL']
        components = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        component_labels = ['Base', 'Sharpe', 'Drawdown', 'Win Rate', 'Consistency', 'Hold Pen']
        
        # Create grouped bar data
        component_by_action = {}
        for comp in components:
            component_by_action[comp] = [df[df['action_type'] == act][comp].mean() for act in action_types]
        
        # Set width and positions
        bar_width = 0.15
        positions = np.arange(len(action_labels))
        
        # Create grouped bar chart
        for i, (comp, values) in enumerate(component_by_action.items()):
            offset = (i - len(components)/2 + 0.5) * bar_width
            ax4.bar(positions + offset, values, bar_width, label=component_labels[i])
        
        # Set chart properties
        ax4.set_title('Average Component Values by Action Type')
        ax4.set_xticks(positions)
        ax4.set_xticklabels(action_labels)
        ax4.set_ylabel('Component Value')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Add table with numerical values below the chart
        cell_text = []
        for act_idx, act in enumerate(action_labels):
            row = [f"{component_by_action[comp][act_idx]:.6f}" for comp in components]
            cell_text.append(row)
            
        the_table = ax4.table(
            cellText=cell_text,
            rowLabels=action_labels,
            colLabels=component_labels,
            loc='bottom',
            bbox=[0, -0.65, 1, 0.3]  # [left, bottom, width, height]
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1, 1.2)
          # Adjust figure to make room for table
        plt.subplots_adjust(bottom=0.35)
    else:
        ax4.text(0.5, 0.5, "Combined reward type required for components analysis", 
                ha='center', va='center', fontsize=14)
    
    # 5. Scatter Plot of Key Components vs Total Reward
    ax5 = plt.subplot(gs3[2, 1])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':        # Find components with highest correlation to reward using safe correlation function
        component_columns = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        correlations = [abs(safe_corrcoef(df[comp].values, df['reward'].values)) for comp in component_columns]
        top_component_idx = np.argmax(correlations)
        top_component = component_columns[top_component_idx]
        
        # Create scatter plot
        scatter = ax5.scatter(df[top_component], df['reward'], 
                             c=df['action_type'], cmap='viridis', 
                             alpha=0.7)
        
        # Add trend line
        try:
            from scipy import stats
            mask = (abs(df[top_component]) < df[top_component].std() * 3)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df.loc[mask, top_component], 
                df.loc[mask, 'reward']
            )
            x_vals = np.linspace(df[top_component].min(), df[top_component].max(), 100)
            ax5.plot(x_vals, slope * x_vals + intercept, '--', color='black')
            
            # Add R² value
            ax5.text(0.05, 0.95, f'R² = {r_value**2:.4f}', 
                    transform=ax5.transAxes, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7))
        except:
            pass
        
        ax5.set_title(f'Top Component vs Total Reward ({top_component.replace("_", " ").title()})')
        ax5.set_xlabel(top_component.replace('_', ' ').title())
        ax5.set_ylabel('Total Reward')
        ax5.grid(True, alpha=0.3)
        
        # Add legend for action types
        legend = ax5.legend(*scatter.legend_elements(),
                          loc="upper right", title="Action Type")
        ax5.add_artist(legend)
    else:
        ax5.text(0.5, 0.5, "Combined reward type required for components analysis", 
                ha='center', va='center', fontsize=14)
    
    # 6. Component Correlation heatmap with performance metrics
    ax6 = plt.subplot(gs3[3, 0])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined' and has_seaborn:
        # Create a correlation heatmap between reward components and performance metrics
        performance_metrics = ['equity_change', 'equity_change_fwd_1', 'equity_change_fwd_5', 'drawdown', 'win_rate']
        component_columns = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        
        # Calculate correlation matrix
        corr_matrix = df[component_columns + performance_metrics].corr()
        
        # Extract just the correlations between components and performance metrics
        corr_subset = corr_matrix.loc[component_columns, performance_metrics]
        
        # Create heatmap
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax6)
        ax6.set_title('Component Correlations with Performance Metrics')
        ax6.set_xlabel('Performance Metrics')
        ax6.set_ylabel('Reward Components')
    else:
        ax6.text(0.5, 0.5, "Combined reward type and seaborn required", 
                ha='center', va='center', fontsize=14)
    
    # 7. Reward Quality Assessment Table
    ax7 = plt.subplot(gs3[3, 1])
    
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        # Create table with reward quality assessments
        ax7.axis('off')  # Turn off axis to show only the table
        
        # Calculate key metrics for reward quality assessment
        equity_reward_corr = df['equity_change'].corr(df['reward'])
        future_equity_corr = df['reward'].corr(df['equity_change_fwd_5'])
        
        # Create reward balance metrics
        action_rewards = {
            'HOLD': df[df['action_type'] == 0]['reward'].mean(),
            'BUY': df[df['action_type'] == 1]['reward'].mean(),
            'SELL': df[df['action_type'] == 2]['reward'].mean()
        }
        
        # Find the predominant component
        component_columns = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        component_avgs = [df[comp].abs().mean() for comp in component_columns]
        predominant_idx = np.argmax(component_avgs)
        predominant_component = component_columns[predominant_idx].replace('_', ' ').title()
        predominant_pct = component_avgs[predominant_idx] / sum(component_avgs) * 100
        
        # Create assessment criteria
        if equity_reward_corr > 0.5:
            equity_alignment = "Strong ✓"
        elif equity_reward_corr > 0.2:
            equity_alignment = "Moderate ⚠"
        else:
            equity_alignment = "Weak ✗"
            
        if future_equity_corr > 0.3:
            predictive_power = "Strong ✓"
        elif future_equity_corr > 0.1:
            predictive_power = "Moderate ⚠"
        else:
            predictive_power = "Weak ✗"
            
        # Check if any action type consistently gets negative rewards
        action_balance = "Good ✓"
        neg_actions = []
        for action, reward in action_rewards.items():
            if reward < -0.0005:  # Threshold for considering an action negatively rewarded
                neg_actions.append(action)
                
        if neg_actions:
            action_balance = f"Poor ✗ ({', '.join(neg_actions)} negative)"
        
        # Check component balance
        component_balance = "Good ✓"
        if predominant_pct > 70:
            component_balance = f"Poor ✗ ({predominant_component} {predominant_pct:.1f}%)"
        elif predominant_pct > 40:
            component_balance = f"Moderate ⚠ ({predominant_component} {predominant_pct:.1f}%)"
        
        # Prepare table data
        assessment_criteria = [
            'Equity-Reward Alignment',
            'Future Predictive Power',
            'Action Reward Balance',
            'Component Balance',
            'Mean Base Reward',
            'Mean Reward Total',
            'Predominant Component'
        ]
        
        assessment_values = [
            f"{equity_alignment} ({equity_reward_corr:.4f})",
            f"{predictive_power} ({future_equity_corr:.4f})",
            action_balance,
            component_balance,
            f"{df['base_reward'].mean():.6f}",
            f"{df['reward'].mean():.6f}",
            f"{predominant_component} ({predominant_pct:.1f}%)"
        ]
          # Create a colorful table - each row must have exactly 1 color entry per cell
        cell_colors = []
        for criterion in assessment_criteria:
            if "Strong" in assessment_values[assessment_criteria.index(criterion)] or "Good" in assessment_values[assessment_criteria.index(criterion)]:
                cell_colors.append(['#d4f7d4'])  # Light green (one entry per row)
            elif "Moderate" in assessment_values[assessment_criteria.index(criterion)]:
                cell_colors.append(['#ffeeba'])  # Light yellow (one entry per row)
            elif "Weak" in assessment_values[assessment_criteria.index(criterion)] or "Poor" in assessment_values[assessment_criteria.index(criterion)]:
                cell_colors.append(['#f7d4d4'])  # Light red (one entry per row)
            else:
                cell_colors.append(['#f0f0f0'])  # Light gray (one entry per row)
        
        # Create table
        table = ax7.table(
            cellText=[[v] for v in assessment_values],
            rowLabels=assessment_criteria,
            colLabels=['Assessment'],
            loc='center',
            cellColours=cell_colors,
            cellLoc='center'
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header
                cell.set_fontsize(11)
                cell.set_text_props(fontweight='bold')
        
        ax7.set_title('Reward Engineering Quality Assessment', fontsize=12)
    else:
        ax7.text(0.5, 0.5, "Combined reward type required for quality assessment", 
                ha='center', va='center', fontsize=14)
    
    # Final layout adjustments for third figure
    plt.suptitle('Reward Component Engineering Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig('reward_component_analysis.png', dpi=150, bbox_inches='tight')
    print("Reward component analysis saved to 'reward_component_analysis.png'")
    
    # Show all figures
    plt.show()
      # Perform additional reward engineering quality analysis
    print("\nReward Engineering Quality Analysis:")
    
    # Check alignment between rewards and equity changes
    equity_reward_correlation = df['equity_change'].corr(df['reward'])
    print(f"1. Reward-Equity Change Correlation: {equity_reward_correlation:.4f}")
    
    if equity_reward_correlation > 0.5:
        print("   ✓ Strong positive correlation: Reward is well aligned with equity growth")
    elif equity_reward_correlation > 0.2:
        print("   ⚠ Moderate correlation: Reward has some alignment with equity growth")
    else:
        print("   ✗ Weak correlation: Reward may not be properly aligned with equity growth")
    
    # Check if rewards predict future performance
    future_correlations = []
    for lag in [1, 5, 10]:
        if f'equity_change_fwd_{lag}' in df.columns:
            future_correlations.append((lag, df['reward'].corr(df[f'equity_change_fwd_{lag}'])))
    
    # Display future correlations if available
    if future_correlations:
        print("\n2. Reward predictive power for future equity changes:")
        for lag, corr in future_correlations:
            print(f"   t+{lag}: {corr:.4f}", end=" ")
            if corr > 0.3:
                print("✓ Good")
            elif corr > 0.1:
                print("⚠ Moderate")
            else:
                print("✗ Weak")
    
    # Check reward distribution by action type
    action_rewards = {
        'HOLD': df[df['action_type'] == 0]['reward'].mean(),
        'BUY': df[df['action_type'] == 1]['reward'].mean(),
        'SELL': df[df['action_type'] == 2]['reward'].mean()
    }
    print("\n3. Average rewards by action type:")
    print(f"   HOLD: {action_rewards['HOLD']:.6f}")
    print(f"   BUY:  {action_rewards['BUY']:.6f}")
    print(f"   SELL: {action_rewards['SELL']:.6f}")
    
    # Analyze balance between action types
    neg_actions = [act for act, val in action_rewards.items() if val < -0.0001]
    if neg_actions:
        print(f"   ⚠ Action imbalance: {', '.join(neg_actions)} have negative mean rewards")
        print("      This may discourage the agent from taking these actions")
    else:
        print("   ✓ Good action balance: No action type has negative mean rewards")
    
    # Check if good trades are rewarded appropriately
    # Find steps where exits occurred (trade closed)
    tp_steps = []
    sl_steps = []
    time_steps = []
    
    for i in range(1, len(df)):
        if df['tp_exits'][i] > df['tp_exits'][i-1]:
            tp_steps.append(i)
        if df['sl_exits'][i] > df['sl_exits'][i-1]:
            sl_steps.append(i)
        if df['time_exits'][i] > df['time_exits'][i-1]:
            time_steps.append(i)
    
    # Analyze exit rewards
    if len(tp_steps) > 0 and len(sl_steps) > 0:
        tp_mean = df.loc[tp_steps, 'reward'].mean()
        sl_mean = df.loc[sl_steps, 'reward'].mean()
        print(f"\n4. Reward ratio (TP/SL): {tp_mean/abs(sl_mean) if sl_mean != 0 else 'N/A':.2f}")
        
        if tp_mean > 0 and sl_mean < 0 and tp_mean > abs(sl_mean):
            print("   ✓ Good structure: TP exits have higher rewards than SL penalties")
        else:
            print("   ⚠ Suboptimal structure: TP/SL reward ratio needs improvement")
    
    # If using combined reward, analyze component contributions
    if hasattr(env, 'reward_calculator') and env.reward_calculator.reward_type == 'combined':
        print("\n5. Reward component analysis:")
        
        # Calculate average contribution per component
        components = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus', 'hold_penalty']
        component_means = {}
        for comp in components:
            component_means[comp] = df[comp].mean()
            
        # Calculate percentage contribution to mean reward
        mean_reward = df['reward'].mean()
        total_component_abs = sum(abs(val) for val in component_means.values())
        
        print("   Component average values and contribution to reward:")
        for comp, val in component_means.items():
            contrib_pct = (abs(val) / total_component_abs * 100) if total_component_abs > 0 else 0
            direction = "+" if val >= 0 else "-"
            print(f"   - {comp.replace('_', ' ').title()}: {val:.6f} {direction} ({contrib_pct:.1f}%)")
        
        # Check component correlations with future performance
        print("\n6. Component correlation with future equity change (t+5):")
        future_col = 'equity_change_fwd_5'
        if future_col in df.columns:
            for comp in components:
                corr = df[comp].corr(df[future_col])
                print(f"   - {comp.replace('_', ' ').title()}: {corr:.4f}", end=" ")
                if abs(corr) > 0.3:
                    print("✓ Strong")
                elif abs(corr) > 0.1:
                    print("⚠ Moderate")
                else:
                    print("✗ Weak")
        
        # Identify reward issues and suggest improvements
        print("\n7. Reward engineering issues and recommendations:")
        
        # Check overall reward level
        if mean_reward < 0:
            print("   ⚠ Issue: Negative mean reward may discourage all actions")
            print("      Suggestion: Add a baseline adjustment to shift rewards upward")
            
        # Check action reward balance
        hold_reward = action_rewards['HOLD']
        trade_reward = (action_rewards['BUY'] + action_rewards['SELL']) / 2
        if hold_reward > trade_reward:
            print("   ⚠ Issue: HOLD actions rewarded more than trading actions")
            print("      Suggestion: Reduce hold penalty or increase trading rewards")
            
        # Check drawdown penalty impact
        if abs(component_means['drawdown_penalty']) > abs(component_means['base_reward']) * 0.5:
            print("   ⚠ Issue: Drawdown penalty has excessive influence on rewards")
            print("      Suggestion: Reduce drawdown penalty scaling factor")
            
        # Check for missing time decay in long-term performance
        if 'corr_reward_future_equity_5' in locals() and abs(equity_reward_correlation) > 1.5 * abs(future_correlations[1][1]):
            print("   ⚠ Issue: Reward strongly favors immediate returns over long-term performance")
            print("      Suggestion: Add time decay for long-term performance evaluation")
            
        # Check sharpe adjustment impact
        if abs(component_means['sharpe_adjustment']) < abs(component_means['base_reward']) * 0.1:
            print("   ⚠ Issue: Sharpe adjustment has minimal impact on rewards")
            print("      Suggestion: Increase sharpe factor scaling in combined reward")

        # Make component-specific recommendations
        print("\n8. Component-specific recommendations:")
        
        # Base reward
        base_scale = 10.0  # Assumed scale factor from code review
        print(f"   - Base Reward: Currently using scale factor of {base_scale}")
        if df['base_reward'].mean() < 0:
            print("     ⚠ Issue: Negative mean base reward")
            print("     Suggestion: Consider adding a baseline offset")
            
        # Drawdown penalty
        print(f"   - Drawdown Penalty: Currently using 2.0 × drawdown × 0.1 scaling")
        if df['drawdown_penalty'].abs().mean() > 0.01:
            print("     ⚠ Issue: High drawdown penalty impact")
            print("     Suggestion: Reduce scaling from 2.0 to 1.0 or less")
            
        # Win rate bonus
        print(f"   - Win Rate Bonus: Currently using (win_rate - 0.5) × 0.1 scaling")
        if df['win_rate_bonus'].abs().mean() < 0.001:
            print("     ⚠ Issue: Win rate bonus has minimal impact")
            print("     Suggestion: Increase scaling factor from 0.1 to 0.2")
            
        # Hold penalty
        print(f"   - Hold Penalty: Currently penalizes after 10 consecutive holds")
        if action_rewards['HOLD'] < 0:
            print("     ⚠ Issue: Negative mean HOLD rewards may cause erratic behavior")
            print("     Suggestion: Start penalty after more steps or reduce magnitude")
            
        # Overall assessment
        equity_growth_pct = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
        print(f"\n9. Overall reward quality assessment:")
        print(f"   - Equity growth: {equity_growth_pct:.2f}%")
        print(f"   - Mean reward: {mean_reward:.6f}")
        print(f"   - Reward-equity correlation: {equity_reward_correlation:.4f}")
        
        if equity_growth_pct > 0 and mean_reward > 0 and equity_reward_correlation > 0.3:
            print("   ✓ GOOD: Reward is generally well aligned with performance")
        elif equity_growth_pct * mean_reward > 0:  # Same sign
            print("   ⚠ MODERATE: Reward has partial alignment with performance")
        else:
            print("   ✗ POOR: Reward is misaligned with performance objectives")
    
    # Show all figures
    plt.show()
    
    # Return the metrics DataFrame for further analysis
    return df

def safe_corrcoef(x, y):
    """
    Calculate correlation coefficient while handling zeros, NaN values, and constant arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Correlation coefficient or 0 if calculation is not possible
    """
    # Remove any NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    # Check if we have enough data points
    if len(x_clean) < 2 or len(y_clean) < 2:
        return 0
    
    # Check for zero standard deviation
    if np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return 0
        
    # Calculate correlation safely
    try:
        return np.corrcoef(x_clean, y_clean)[0, 1]
    except:
        return 0
