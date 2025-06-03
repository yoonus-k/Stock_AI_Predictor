# Monitoring Tools for Reinforcement Learning Trading Agent

This directory contains a comprehensive set of monitoring tools for analyzing and visualizing the performance of the RL trading agent. These tools help track learning progress, assess performance, analyze feature importance, and gain insights into the model's decision-making process.

## Available Tools

### 1. Unified Monitoring Dashboard

The main entry point for all monitoring tools that provides a comprehensive interface with different analysis modules.

```bash
python Scripts/monitoring_dashboard.py --model=path/to/model.zip --log-dir=path/to/logs
```

### 2. TensorBoard Integration

Visualize training metrics, rewards, and other statistics using TensorBoard.

```bash
tensorboard --logdir=RL/Logs/tensorboard
```

Or use the provided script:

```bash
python Scripts/view_tensorboard.py --log-dir=RL/Logs/tensorboard --port=6006
```

### 3. Checkpoint Analysis

Evaluate model checkpoints to identify the best performing model during training.

```bash
python Scripts/analyze_checkpoints.py --checkpoint-dir=RL/Logs/checkpoints --episodes=3
```

### 4. Feature Importance Analysis

Analyze which features have the greatest impact on the model's decisions.

```bash
python Scripts/monitor_feature_importance.py --model=RL/Models/model.zip --output-dir=RL/Analysis --episodes=5
```

### 5. Trading Strategy Analysis

Analyze the trading strategy and portfolio performance of the trained model.

```bash
python Scripts/monitor_trading_strategy.py --model=RL/Models/model.zip --output-dir=RL/Analysis/strategy --episodes=1
```

### 6. Decision Boundary Analysis

Visualize how the model makes decisions based on different features.

```bash
python Scripts/monitor_decision_boundaries.py --model=RL/Models/model.zip --output-dir=RL/Analysis/decision_boundaries
```

### 7. Live Training Dashboard

Monitor training progress in real-time with an interactive dashboard.

```bash
python Scripts/live_training_dashboard.py --log-dir=RL/Logs --refresh=5
```

## Training with Monitoring

To train a model with enhanced monitoring capabilities, use the `train_with_monitoring.py` script:

```bash
python Scripts/train_with_monitoring.py
```

This script includes:
- TensorBoard integration
- Checkpoint saving
- Feature importance tracking during training
- Portfolio performance monitoring
- Metadata logging

## Analysis Outputs

The monitoring tools generate various outputs that are stored in the following directories:

- `RL/Logs/tensorboard` - TensorBoard logs
- `RL/Logs/checkpoints` - Model checkpoints
- `RL/Analysis/feature_importance` - Feature importance analysis
- `RL/Analysis/strategy` - Trading strategy analysis
- `RL/Analysis/decision_boundaries` - Decision boundary visualizations

## Key Features

### Feature Importance Analysis
- Permutation importance calculation
- Feature ablation tests
- SHAP value analysis

### Trading Strategy Analysis
- Portfolio performance tracking
- Action distribution analysis
- Trade visualization
- Performance metrics calculation

### Decision Boundary Analysis
- State-value function visualization
- Pattern recognition analysis
- Feature distribution analysis

### Performance Monitoring
- Reward tracking
- Win rate analysis
- Portfolio value tracking
- Maximum drawdown calculation

## Requirements

The monitoring tools require the following packages:
- matplotlib
- seaborn
- pandas
- numpy
- stable-baselines3
- tensorboard
- shap (for SHAP analysis)
- scikit-learn (for dimensionality reduction)

## Troubleshooting

If you encounter issues with the monitoring tools, try the following:

1. Ensure all required packages are installed
2. Check that log directories exist and have appropriate permissions
3. Verify that the model file path is correct
4. Check for error messages in the terminal output
