"""
Live Training Dashboard for Reinforcement Learning Trading Model

This script creates a real-time monitoring dashboard for the RL trading agent that 
displays key metrics during training and helps visualize the learning process.

Features:
1. Real-time reward and episode length tracking
2. Action distribution visualization
3. Portfolio performance charts
4. Trading activity visualization
5. Feature importance updates
6. Learning progress indicators
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import time
import json
from datetime import datetime
from threading import Thread
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


class TrainingMonitor:
    """Class to monitor and visualize training progress"""
    
    def __init__(self, log_dir=None, refresh_interval=5):
        """
        Initialize the training monitor
        
        Parameters:
            log_dir: Directory containing the training logs
            refresh_interval: How often to refresh data (in seconds)
        """
        # Setup directories
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "Logs"
        self.log_dir = Path(log_dir)
        
        # Setup data containers
        self.rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.portfolio_values = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
        self.win_rates = []
        self.losses = []
        self.refresh_interval = refresh_interval
        
        # Track if monitoring is active
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Initialize UI components
        self.root = None
        self.fig = None
        self.canvas = None
    
    def start_monitoring(self):
        """Start the monitoring process in a separate thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop that updates data at regular intervals"""
        while self.is_monitoring:
            self._update_data()
            time.sleep(self.refresh_interval)
    
    def _update_data(self):
        """Update monitoring data from log files"""
        try:
            # Check for monitor log
            monitor_path = self.log_dir / "monitor.csv"
            if monitor_path.exists():
                # Read monitor data (skip header rows)
                monitor_data = pd.read_csv(monitor_path, skiprows=1)
                
                if not monitor_data.empty:
                    # Extract data
                    self.rewards = monitor_data['r'].tolist()
                    self.episode_lengths = monitor_data['l'].tolist()
                    self.timesteps = monitor_data['t'].tolist()
                    
                    # Calculate rolling averages
                    window_size = min(10, len(self.rewards))
                    if window_size > 0:
                        self.rolling_rewards = pd.Series(self.rewards).rolling(window_size).mean().tolist()
            
            # Check for performance metrics
            metrics_path = self.log_dir / "performance_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                    # Extract relevant metrics
                    if 'win_rate' in metrics:
                        self.win_rates.append(metrics['win_rate'])
                    
                    if 'portfolio_values' in metrics:
                        self.portfolio_values = metrics['portfolio_values']
                    
                    if 'action_counts' in metrics:
                        self.action_counts = metrics['action_counts']
            
        except Exception as e:
            print(f"Error updating monitor data: {e}")
    
    def create_dashboard(self):
        """Create and display the monitoring dashboard"""
        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.title("RL Trading Agent Training Monitor")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        tab_control = ttk.Notebook(main_frame)
        
        # Training progress tab
        train_tab = ttk.Frame(tab_control)
        tab_control.add(train_tab, text="Training Progress")
        self._create_training_plots(train_tab)
        
        # Trading performance tab
        perf_tab = ttk.Frame(tab_control)
        tab_control.add(perf_tab, text="Trading Performance")
        self._create_performance_plots(perf_tab)
        
        # Feature importance tab
        feature_tab = ttk.Frame(tab_control)
        tab_control.add(feature_tab, text="Feature Importance")
        self._create_feature_plots(feature_tab)
        
        tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Add control buttons
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X)
        
        ttk.Button(
            control_frame, 
            text="Refresh Now", 
            command=self._update_dashboard
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Start Auto-Refresh", 
            command=self._start_auto_refresh
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Stop Auto-Refresh", 
            command=self._stop_auto_refresh
        ).pack(side=tk.LEFT, padx=5)
        
        # Setup auto-refresh
        self.auto_refresh = False
        
        # Start the monitoring thread
        self.start_monitoring()
        
        # Start the dashboard
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_dashboard()  # Initial update
        self.root.mainloop()
    
    def _create_training_plots(self, parent):
        """Create plots for training progress tab"""
        # Create figure with subplots
        self.train_fig, self.train_axes = plt.subplots(2, 2, figsize=(10, 8))
        self.train_fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Create canvas for matplotlib figure
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, parent)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_performance_plots(self, parent):
        """Create plots for trading performance tab"""
        # Create figure with subplots
        self.perf_fig, self.perf_axes = plt.subplots(2, 2, figsize=(10, 8))
        self.perf_fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Create canvas for matplotlib figure
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, parent)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_feature_plots(self, parent):
        """Create plots for feature importance tab"""
        # Create figure with subplots
        self.feature_fig, self.feature_axes = plt.subplots(2, 1, figsize=(10, 8))
        self.feature_fig.subplots_adjust(hspace=0.3)
        
        # Create canvas for matplotlib figure
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, parent)
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _update_dashboard(self):
        """Update all plots with current data"""
        # Update data
        self._update_data()
        
        # Update training plots
        self._update_training_plots()
        
        # Update performance plots
        self._update_performance_plots()
        
        # Update feature plots
        self._update_feature_plots()
    
    def _update_training_plots(self):
        """Update plots on training tab"""
        if not hasattr(self, 'train_axes'):
            return
            
        # Clear axes
        for ax in self.train_axes.flat:
            ax.clear()
        
        # Plot reward history
        if self.rewards:
            x_values = list(range(len(self.rewards)))
            self.train_axes[0, 0].plot(x_values, self.rewards, 'b-', alpha=0.5, label='Reward')
            
            # Plot rolling average if available
            if hasattr(self, 'rolling_rewards') and len(self.rolling_rewards) > 0:
                valid_indices = [i for i, v in enumerate(self.rolling_rewards) if not np.isnan(v)]
                if valid_indices:
                    self.train_axes[0, 0].plot(
                        [i for i in valid_indices],
                        [self.rolling_rewards[i] for i in valid_indices],
                        'r-', label='Rolling Avg'
                    )
            
            self.train_axes[0, 0].set_title('Reward History')
            self.train_axes[0, 0].set_xlabel('Episode')
            self.train_axes[0, 0].set_ylabel('Reward')
            self.train_axes[0, 0].legend()
            self.train_axes[0, 0].grid(True)
        
        # Plot episode lengths
        if self.episode_lengths:
            self.train_axes[0, 1].plot(list(range(len(self.episode_lengths))), self.episode_lengths)
            self.train_axes[0, 1].set_title('Episode Lengths')
            self.train_axes[0, 1].set_xlabel('Episode')
            self.train_axes[0, 1].set_ylabel('Steps')
            self.train_axes[0, 1].grid(True)
        
        # Plot learning curve (cumulative reward)
        if self.rewards:
            cumulative_rewards = np.cumsum(self.rewards)
            self.train_axes[1, 0].plot(list(range(len(cumulative_rewards))), cumulative_rewards)
            self.train_axes[1, 0].set_title('Cumulative Reward')
            self.train_axes[1, 0].set_xlabel('Episode')
            self.train_axes[1, 0].set_ylabel('Cumulative Reward')
            self.train_axes[1, 0].grid(True)
        
        # Plot action distribution as pie chart
        if sum(self.action_counts.values()) > 0:
            labels = ['Hold', 'Buy', 'Sell']
            values = [self.action_counts.get(0, 0), self.action_counts.get(1, 0), self.action_counts.get(2, 0)]
            
            self.train_axes[1, 1].pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                colors=['gray', 'green', 'red'],
                startangle=90
            )
            self.train_axes[1, 1].set_title('Action Distribution')
        
        self.train_fig.tight_layout()
        self.train_canvas.draw()
    
    def _update_performance_plots(self):
        """Update plots on performance tab"""
        if not hasattr(self, 'perf_axes'):
            return
            
        # Clear axes
        for ax in self.perf_axes.flat:
            ax.clear()
        
        # Plot portfolio value
        if self.portfolio_values:
            self.perf_axes[0, 0].plot(list(range(len(self.portfolio_values))), self.portfolio_values)
            self.perf_axes[0, 0].set_title('Portfolio Value')
            self.perf_axes[0, 0].set_xlabel('Step')
            self.perf_axes[0, 0].set_ylabel('Value ($)')
            self.perf_axes[0, 0].grid(True)
        
        # Plot win rate history
        if self.win_rates:
            self.perf_axes[0, 1].plot(list(range(len(self.win_rates))), self.win_rates)
            self.perf_axes[0, 1].set_title('Win Rate History')
            self.perf_axes[0, 1].set_xlabel('Update')
            self.perf_axes[0, 1].set_ylabel('Win Rate (%)')
            self.perf_axes[0, 1].set_ylim([0, 100])
            self.perf_axes[0, 1].grid(True)
        
        # Placeholder plots - these would need actual data
        self.perf_axes[1, 0].text(
            0.5, 0.5, 
            "Profit/Loss Distribution\n(No data yet)", 
            ha='center', va='center'
        )
        self.perf_axes[1, 0].set_title('Profit/Loss Distribution')
        
        self.perf_axes[1, 1].text(
            0.5, 0.5, 
            "Trade Duration Analysis\n(No data yet)", 
            ha='center', va='center'
        )
        self.perf_axes[1, 1].set_title('Trade Duration Analysis')
        
        self.perf_fig.tight_layout()
        self.perf_canvas.draw()
    
    def _update_feature_plots(self):
        """Update plots on feature tab"""
        if not hasattr(self, 'feature_axes'):
            return
            
        # Clear axes
        for ax in self.feature_axes:
            ax.clear()
        
        # Check if feature importance data exists
        feature_file = self.log_dir / "feature_importance.json"
        if feature_file.exists():
            try:
                with open(feature_file, 'r') as f:
                    feature_data = json.load(f)
                
                if 'permutation_importance' in feature_data:
                    # Extract feature importance data
                    features = feature_data['permutation_importance']['feature_names']
                    importance = feature_data['permutation_importance']['importance']
                    
                    # Sort by importance
                    sorted_indices = np.argsort(importance)
                    features = [features[i] for i in sorted_indices]
                    importance = [importance[i] for i in sorted_indices]
                    
                    # Plot horizontal bar chart
                    y_pos = np.arange(len(features))
                    self.feature_axes[0].barh(y_pos, importance, align='center')
                    self.feature_axes[0].set_yticks(y_pos)
                    self.feature_axes[0].set_yticklabels(features)
                    self.feature_axes[0].invert_yaxis()  # labels read top-to-bottom
                    self.feature_axes[0].set_title('Feature Importance (Permutation Method)')
                    self.feature_axes[0].set_xlabel('Importance')
                else:
                    self.feature_axes[0].text(
                        0.5, 0.5, 
                        "Feature Importance Analysis\n(No data yet)", 
                        ha='center', va='center'
                    )
            except Exception as e:
                print(f"Error loading feature importance data: {e}")
                self.feature_axes[0].text(
                    0.5, 0.5, 
                    "Error loading feature data", 
                    ha='center', va='center'
                )
        else:
            self.feature_axes[0].text(
                0.5, 0.5, 
                "Feature Importance Analysis\n(No data yet)", 
                ha='center', va='center'
            )
        
        # Placeholder for SHAP plot
        self.feature_axes[1].text(
            0.5, 0.5, 
            "SHAP Analysis\n(No data yet)", 
            ha='center', va='center'
        )
        self.feature_axes[1].set_title('SHAP Analysis')
        
        self.feature_fig.tight_layout()
        self.feature_canvas.draw()
    
    def _start_auto_refresh(self):
        """Start auto-refreshing the dashboard"""
        if not self.auto_refresh:
            self.auto_refresh = True
            self._auto_refresh_loop()
    
    def _stop_auto_refresh(self):
        """Stop auto-refreshing the dashboard"""
        self.auto_refresh = False
    
    def _auto_refresh_loop(self):
        """Refresh dashboard at regular intervals"""
        if self.auto_refresh and self.root:
            self._update_dashboard()
            self.root.after(int(self.refresh_interval * 1000), self._auto_refresh_loop)
    
    def _on_close(self):
        """Handle window close event"""
        self.stop_monitoring()
        if self.root:
            self.root.destroy()


def run_dashboard(log_dir=None, refresh_interval=5):
    """
    Run the training monitoring dashboard
    
    Parameters:
        log_dir: Directory containing training logs
        refresh_interval: How often to refresh the dashboard (in seconds)
    """
    monitor = TrainingMonitor(log_dir, refresh_interval)
    monitor.create_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training monitoring dashboard")
    parser.add_argument('--log-dir', type=str, default=None, help="Directory containing training logs")
    parser.add_argument('--refresh', type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    run_dashboard(args.log_dir, args.refresh)
