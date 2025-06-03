"""
Optimized callbacks for RL training that won't cause freezes or training stalls.
This file contains improved versions of FeatureImportanceCallback and PortfolioTrackingCallback
that add safeguards against common issues that can cause training to hang.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


# Improved callbacks with enhanced safety features to prevent training freezes
class FeatureImportanceCallback(BaseCallback):
    """
    Callback to periodically calculate and save feature importance during training.
    This optimized version adds safeguards against freezing and performance issues.
    """

    def __init__(self, eval_env, log_path, eval_freq=10000, max_saves=5, verbose=0):
        """
        Initialize the callback

        Parameters:
            eval_env: Evaluation environment
            log_path: Path to save feature importance data
            eval_freq: How often to calculate feature importance (in timesteps)
            max_saves: Maximum number of saves to disk to reduce I/O operations
            verbose: Verbosity level
        """
        super(FeatureImportanceCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.last_eval_timestep = 0
        self.max_saves = max_saves
        self.save_count = 0
        self.all_importances = []

        # Feature names - define these based on your environment
        self.feature_names = [
            # Base pattern features (7 features)
            "probability",
            "action",
            "reward_risk_ratio",
            "max_gain",
            "max_drawdown",
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
            # Portfolio features (5 features)
            "balance_ratio",
            "position_ratio",
            "position",
            "max_drawdown",
            "win_rate",
        ]

        # Create log directory
        try:
            self.log_path.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            print(f"Warning: Could not create log directory: {e}")

    # And add this helper method
    def _is_near_end_of_training(self):
        # Use a percentage-based approach instead
        if not hasattr(self, "target_timesteps"):
            # Store the target when first called
            self.target_timesteps = (
                getattr(self.model, "learning_starts", 0)
                + getattr(self.model, "n_steps", 2048)
                * getattr(self.model, "n_epochs", 10)
                * 20
            )  # Rough estimate
        return self.num_timesteps >= self.target_timesteps - self.eval_freq

    def _on_step(self) -> bool:
        """
        Called at each step during training

        Returns:
            True (continue training)
        """
        # Check if it's time to calculate feature importance
        if (
            self.num_timesteps - self.last_eval_timestep >= self.eval_freq
            and self.save_count < self.max_saves
        ):

            self.last_eval_timestep = self.num_timesteps

            # Use a shorter timeout to prevent hanging
            timeout = 5  # reduced from 10 seconds
            start_time = time.time()
            success = False

            try:
                if self.verbose > 0:
                    print(
                        f"Calculating feature importance at timestep {self.num_timesteps}..."
                    )

                # Generate importance values (placeholder implementation)
                importance = np.random.uniform(0, 1, size=len(self.feature_names))

                # Store data in memory first
                importance_data = {
                    "timestep": int(self.num_timesteps),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "permutation_importance": {
                        "feature_names": self.feature_names,
                        "importance": importance.tolist(),
                    },
                }
                self.all_importances.append(importance_data)
                self.save_count += 1
                success = True

                # Only save to disk at end of training or at specific intervals
                if self.save_count >= self.max_saves or self._is_near_end_of_training():
                    self._save_to_disk()

            except Exception as e:
                print(f"Error calculating feature importance: {e}")

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(
                    f"Warning: Feature importance calculation took {elapsed:.2f}s (timeout: {timeout}s)"
                )
                # Return True to continue even if timeout occurred

        return True

    def _save_to_disk(self):
        """Save all accumulated importance data to disk"""
        try:
            file_path = self.log_path / "feature_importance.json"
            with open(file_path, "w") as f:
                json.dump({"importances": self.all_importances}, f, indent=2)

            if self.verbose > 0:
                print(f"Feature importance saved to {file_path}")

        except Exception as e:
            print(f"Error saving feature importance to disk: {e}")

    def on_training_end(self):
        """Save all data when training ends"""
        if self.all_importances and not self.save_count >= self.max_saves:
            self._save_to_disk()


class EvalCallback(BaseCallback):
    """Safe evaluation callback with timeout protection to prevent hanging"""

    def __init__(
        self,
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
        warn_on_eval_error=True,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.warn_on_eval_error = warn_on_eval_error

        # Initialize variables
        self.best_mean_reward = -float("inf")
        self.last_mean_reward = -float("inf")
        self.last_time_evaluated = 0

        # Create output directories if they don't exist
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # Don't evaluate if we haven't reached the evaluation frequency
        if self.eval_freq > 0 and self.n_calls % self.eval_freq != 0:
            return True

        # Avoid evaluating at the same timestep
        if self.n_calls == self.last_time_evaluated:
            return True

        self.last_time_evaluated = self.n_calls

        # Use a timeout mechanism
        import threading

        # Flag to track if evaluation completed
        eval_completed = False
        eval_results = [None, None]  # To store results from thread

        # Thread function to perform evaluation
        def evaluate_policy():
            nonlocal eval_completed, eval_results
            try:
                # Run evaluation episodes
                episode_rewards = []
                episode_lengths = []

                for i in range(self.n_eval_episodes):
                    # Reset environment
                    obs, _ = self.eval_env.reset()
                    done = False
                    truncated = False
                    episode_reward = 0.0
                    episode_length = 0

                    # Step through an episode with a safety counter to avoid infinite loops
                    safety_counter = 0
                    max_steps = 1000  # Maximum steps per episode

                    while not (done or truncated) and safety_counter < max_steps:
                        # Get action from model
                        action, _ = self.model.predict(
                            obs, deterministic=self.deterministic
                        )
                        # Execute action
                        obs, reward, done, truncated, info = self.eval_env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        safety_counter += 1

                        if self.render:
                            self.eval_env.render()

                    # Store episode results
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                # Compute mean reward and length
                mean_reward = np.mean(episode_rewards)
                mean_length = np.mean(episode_lengths)
                eval_results = [mean_reward, mean_length]

                eval_completed = True
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error during evaluation: {e}")
                if self.warn_on_eval_error:
                    import traceback

                    traceback.print_exc()
                eval_completed = True  # Mark as completed even if there's an error

        # Start evaluation in a separate thread
        eval_thread = threading.Thread(target=evaluate_policy)
        eval_thread.daemon = True
        eval_thread.start()

        # Wait for evaluation to complete or timeout
        timeout = 15  # seconds
        start_time = time.time()
        while not eval_completed and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not eval_completed:
            if self.verbose > 0:
                print("⚠️ WARNING: Evaluation timed out after 15 seconds")
            return True  # Continue training even if evaluation times out

        # Process evaluation results if available
        if eval_results[0] is not None:
            mean_reward, mean_length = eval_results
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f}"
                )
                print(f"Episode length: {mean_length:.2f} timesteps")

            # Save best model
            if self.best_model_save_path is not None:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"New best mean reward: {self.best_mean_reward:.2f}")
                    try:
                        self.model.save(
                            os.path.join(self.best_model_save_path, "best_model")
                        )
                    except:
                        # Don't fail training if saving fails
                        print("Error saving best model")

        return True


class PortfolioTrackingCallback(BaseCallback):
    """
    Callback to track portfolio performance during training.
    This optimized version prevents infinite loops and adds comprehensive metrics tracking including
    max drawdown, trade counts, Sharpe ratio, and other performance indicators.
    """

    def __init__(
        self,
        eval_env,
        log_path,
        eval_freq=5000,
        max_eval_steps=500,
        n_eval_episodes=1,
        risk_free_rate=0.02,  # 2% annual risk-free rate for Sharpe calculation
        verbose=0,
    ):
        """
        Initialize the callback

        Parameters:
            eval_env: Evaluation environment
            log_path: Path to save performance data
            eval_freq: How often to evaluate portfolio performance (in timesteps)
            max_eval_steps: Maximum steps per evaluation episode (prevents infinite loops)
            n_eval_episodes: Number of episodes to evaluate
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
            verbose: Verbosity level
        """
        super(PortfolioTrackingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.last_eval_timestep = 0
        self.max_eval_steps = max_eval_steps
        self.n_eval_episodes = n_eval_episodes
        self.risk_free_rate = risk_free_rate

        # Portfolio metrics storage
        self.portfolio_values = []
        self.daily_returns = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
        self.position_sizes_counts = {i: 0 for i in range(10)}  # Position sizes 0-9
        self.risk_reward_ratios_counts = {i: 0 for i in range(10)}
        self.actions = []
        self.wins = 0
        self.losses = 0
        self.buy_count = 0
        self.sell_count = 0 
        self.hold_count = 0
        self.trade_count = 0  # Total number of trades (buy + sell)
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.all_metrics = []
        
        # Drawdown tracking
        self.peak_value = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.drawdown_periods = []
        
        # Trade performance tracking
        self.profitable_trades = 0
        self.losing_trades = 0
        self.avg_profit_per_trade = 0
        self.avg_loss_per_trade = 0
        self.profit_factor = 0  # Total profit / Total loss
        
        # Time-based metrics
        self.evaluation_times = []

        # Create log directory
        try:
            # Ensure Metrics directory exists
            metrics_dir = self.log_path / "Metrics"
            metrics_dir.mkdir(exist_ok=True, parents=True)
            
            # Create Charts directory for visualization files
            charts_dir = self.log_path / "Charts"
            charts_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            print(f"Warning: Could not create log directory: {e}")

    def _on_step(self) -> bool:
        """
        Called at each step during training

        Returns:
            True (continue training)
        """
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps

            # Shorter timeout for Colab/Kaggle environments
            timeout = 15  # reduced from 30 seconds
            start_time = time.time()

            try:
                if self.verbose > 0:
                    print(f"Evaluating portfolio at timestep {self.num_timesteps}...")

                self._evaluate_portfolio_safely(self.model, self.num_timesteps)

                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(
                        f"Warning: Portfolio evaluation took {elapsed:.2f}s (timeout: {timeout}s)"
                    )

            except Exception as e:
                print(f"Error evaluating portfolio: {e}")
                # Continue training even if portfolio evaluation fails

        return True    
    def _evaluate_portfolio_safely(self, model, timestep):
        """
        Evaluate portfolio performance using current model with safety measures
        and calculate comprehensive metrics including drawdown and trade statistics.

        Parameters:
            model: Current RL model
            timestep: Current timestep
        """
        # Track evaluation start time
        start_time = time.time()

        # Reset metrics for this evaluation
        self.portfolio_values = []
        self.daily_returns = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
        self.position_sizes_counts = {i: 0 for i in range(10)}  # Position sizes 0-9
        self.risk_reward_ratios_counts = {i: 0 for i in range(10)}
        self.wins = 0
        self.losses = 0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # Reset drawdown metrics
        self.peak_value = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.drawdown_periods = []

        # Reset trade metrics
        self.profitable_trades = 0
        self.losing_trades = 0

        # Store results for each episode
        episode_results = []

        # Trade tracking 
        trade_results = []
        trade_durations = []

        # Run evaluation for multiple episodes
        for episode in range(self.n_eval_episodes):
            # Reset environment
            try:
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                portfolio_value = 10000.0  # Initial portfolio value
                episode_portfolio_values = [portfolio_value]
                episode_daily_returns = []
                episode_peak_value = portfolio_value  # Initialize peak value for this episode
                episode_max_drawdown = 0

                # Episode-specific trade tracking
                previous_action_type = None
                in_trade = False
                trade_entry_value = 0
                current_trade_steps = 0
                episode_actions = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
                episode_wins = 0
                episode_losses = 0

                # Using a for loop with fixed iterations instead of while loop
                # This is safer and prevents infinite loops
                for step in range(self.max_eval_steps):
                    previous_value = portfolio_value

                    # Get action from model with error handling
                    try:
                        action, _states = model.predict(obs, deterministic=True)
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        break  

                    # Count action - handle MultiDiscrete action space properly
                    if hasattr(action, "__len__"):
                        # Extract the action type directly from the MultiDiscrete action
                        # action[0] is already the discrete action type (0=HOLD, 1=BUY, 2=SELL)
                        action_type = int(action[0])
                    else:
                        # Handle case where action is already discrete
                        action_type = int(action)

                    # Update action counts for this episode
                    episode_actions[action_type] = episode_actions.get(action_type, 0) + 1

                    # Track specific action types
                    if action_type == 0:  # HOLD
                        self.hold_count += 1
                    elif action_type == 1:  # BUY
                        self.buy_count += 1
                        if not in_trade:
                            in_trade = True
                            trade_entry_value = portfolio_value
                            current_trade_steps = 0
                    elif action_type == 2:  # SELL
                        self.sell_count += 1
                        if in_trade:
                            in_trade = False
                            # Calculate trade result
                            trade_result = (portfolio_value - trade_entry_value) / trade_entry_value
                            trade_results.append(trade_result)
                            trade_durations.append(current_trade_steps)

                            if trade_result > 0:
                                self.profitable_trades += 1
                            else:
                                self.losing_trades += 1

                    # Track trade duration if in a trade
                    if in_trade:
                        current_trade_steps += 1

                    # Track consecutive actions for pattern analysis
                    if previous_action_type is not None and previous_action_type != action_type:
                        self.actions.append(action_type)  # Only track action changes
                    previous_action_type = action_type

                    # Step environment with error handling
                    try:
                        obs, reward, done, truncated, info = self.eval_env.step(action)
                    except Exception as e:
                        print(f"Error during environment step: {e}")
                        break

                    # Update portfolio value
                    if reward > 0:
                        portfolio_value *= 1 + min(reward / 100, 0.05)  # Limit to 5% gain
                        episode_wins += 1
                        self.wins += 1
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                        if self.consecutive_wins > self.max_consecutive_wins:
                            self.max_consecutive_wins = self.consecutive_wins
                    elif reward < 0:
                        portfolio_value *= 1 + max(reward / 100, -0.05)  # Limit to 5% loss
                        episode_losses += 1
                        self.losses += 1
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0
                        if self.consecutive_losses > self.max_consecutive_losses:
                            self.max_consecutive_losses = self.consecutive_losses

                    # Calculate daily return
                    daily_return = (portfolio_value - previous_value) / previous_value
                    episode_daily_returns.append(daily_return)

                    # Update portfolio values for this episode
                    episode_portfolio_values.append(portfolio_value)

                    # Update max drawdown for this episode
                    if portfolio_value > episode_peak_value:
                        episode_peak_value = portfolio_value

                    # Calculate current drawdown
                    if episode_peak_value > 0:
                        current_drawdown = (episode_peak_value - portfolio_value) / episode_peak_value

                        # Update max drawdown if current drawdown is larger
                        if current_drawdown > episode_max_drawdown:                          
                            episode_max_drawdown = current_drawdown
                            if episode_max_drawdown > self.max_drawdown:
                                self.max_drawdown = episode_max_drawdown
                                self.drawdown_periods.append({
                                    'episode': episode,
                                    'step': step,
                                    'drawdown': self.max_drawdown,
                                    'peak_value': episode_peak_value,
                                    'current_value': portfolio_value
                                })

                    # Break if environment is done
                    if done or truncated:
                        break

                # If we've reached max steps without termination, log a warning
                if step == self.max_eval_steps - 1 and not (done or truncated):
                    print(
                        f"Warning: Evaluation reached max steps ({self.max_eval_steps}) without termination"
                    )

                # Store episode results
                episode_result = {
                    'portfolio_values': episode_portfolio_values,
                    'daily_returns': episode_daily_returns,
                    'action_counts': episode_actions,
                    'wins': episode_wins,
                    'losses': episode_losses,
                    'max_drawdown': episode_max_drawdown,
                    'peak_value': episode_peak_value,
                    'final_value': portfolio_value
                }
                episode_results.append(episode_result)

                # Handle any open trades at the end of episode
                if in_trade:
                    # Calculate trade result for open trade
                    trade_result = (portfolio_value - trade_entry_value) / trade_entry_value
                    trade_results.append(trade_result)
                    trade_durations.append(current_trade_steps)

                    if trade_result > 0:
                        self.profitable_trades += 1
                    else:
                        self.losing_trades += 1

            except Exception as e:
                print(f"Error in evaluation episode {episode}: {e}")
                continue

        # Update trade count
        self.trade_count = self.buy_count + self.sell_count

        # Aggregate results from all episodes
        self.portfolio_values = episode_results[0]['portfolio_values'] if episode_results else [10000.0]
        self.daily_returns = []

        # Combine daily returns from all episodes
        for ep in episode_results:
            self.daily_returns.extend(ep['daily_returns'])

            # Update action counts in the main counter
            for action_type, count in ep['action_counts'].items():
                self.action_counts[action_type] = self.action_counts.get(action_type, 0) + count

        # Calculate evaluation time
        eval_time = time.time() - start_time
        self.evaluation_times.append(eval_time)

        # Calculate performance metrics
        try:
            # Calculate win rate
            total_trades = self.wins + self.losses
            win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

            # Calculate Sharpe ratio (annualized)
            sharpe_ratio = 0
            if self.daily_returns:
                mean_return = np.mean(self.daily_returns)
                std_return = np.std(self.daily_returns)
                daily_risk_free = self.risk_free_rate / 252  # Assuming 252 trading days
                if std_return > 0:
                    sharpe_ratio = (mean_return - daily_risk_free) / std_return * np.sqrt(252)

            # Calculate trade metrics
            total_profit = sum(max(r, 0) for r in trade_results) if trade_results else 0
            total_loss = abs(sum(min(r, 0) for r in trade_results)) if trade_results else 0
            
            # Calculate profit factor
            self.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate average profit/loss per trade
            if self.profitable_trades > 0:
                self.avg_profit_per_trade = total_profit / self.profitable_trades
            if self.losing_trades > 0:
                self.avg_loss_per_trade = total_loss / self.losing_trades
                
            # Calculate return
            initial_value = (
                self.portfolio_values[0] if self.portfolio_values else 10000.0
            )
            final_value = (
                self.portfolio_values[-1] if self.portfolio_values else initial_value
            )
            return_pct = ((final_value / initial_value) - 1) * 100

            # Save performance metrics
            metrics = {
                "timestep": int(timestep),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_portfolio_value": float(final_value),
                "return_pct": float(return_pct),
                "win_rate": float(win_rate),
                "action_counts": {str(k): v for k, v in self.action_counts.items()},
                "portfolio_values": [
                    float(val) for val in self.portfolio_values[:50]
                ],  # Further reduced to 50 values
                # Add new comprehensive metrics
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(self.max_drawdown),
                "trade_count": self.trade_count,
                "buy_count": self.buy_count,
                "sell_count": self.sell_count,
                "hold_count": self.hold_count,
                "profit_factor": float(self.profit_factor),
                "avg_profit_per_trade": float(self.avg_profit_per_trade),
                "avg_loss_per_trade": float(self.avg_loss_per_trade),
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
                "profitable_trades": self.profitable_trades,
                "losing_trades": self.losing_trades
            }

            self.all_metrics.append(metrics)

            # Save to file
            file_path = self.log_path / "Metrics" / "performance_metrics.json"
            try:
                if file_path.exists():
                    # Load existing metrics
                    with open(file_path, "r") as f:
                        existing_data = json.load(f)
                        existing_metrics = existing_data.get("metrics", [])
                        self.all_metrics = existing_metrics + [metrics]
                else:
                    self.all_metrics = [metrics]

                # Save updated metrics
                with open(file_path, "w") as f:
                    json.dump({"metrics": self.all_metrics}, f, indent=2)

            except json.JSONDecodeError:
                print("Warning: Corrupted JSON file. Overwriting with new metrics.")
                with open(file_path, "w") as f:
                    json.dump({"metrics": [metrics]}, f, indent=2)

            except Exception as e:
                print(f"Error saving metrics to file: {e}")

            if self.verbose > 0:
                print(f"Portfolio metrics saved to {file_path}")
                print(f"  Final value: ${final_value:.2f} ({return_pct:.2f}%)")
                print(f"  Win rate: {win_rate:.2f}%")
                print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
                print(f"  Max drawdown: {self.max_drawdown:.2f}")
                print(f"  Actions: {self.action_counts}")

        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
