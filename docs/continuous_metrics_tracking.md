# Continuous Metrics Tracking Across Training Runs

This document describes how to use the continuous metrics tracking feature to monitor model performance across multiple training runs and enhancements.

## Overview

When enhancing or continuing to train a model, it's important to track metrics continuously to understand how performance evolves across multiple training sessions. This system allows you to:

1. Track timesteps continuously across runs by maintaining an offset
2. Generate plots that show metric progression across the entire enhancement chain
3. Create comprehensive reports comparing base models with their enhanced versions

## How It Works

The continuous metrics tracking system uses several components:

1. **MLflowLoggingCallback**: Modified to accept a `previous_timesteps` parameter that offsets all metrics
2. **MLflowMetricConnector**: A new class that analyzes runs in an enhancement chain and creates continuous visualizations
3. **Enhancement reports**: HTML reports that show before/after comparisons and continuous performance plots

## Using Continuous Metrics Tracking

### In Training Scripts

When enhancing or continuing training of a model, the training script automatically retrieves the previous model's total timesteps and passes them to the callback:

```python
# Get previous timesteps from base model
previous_timesteps = 0  # Default value 
if base_model_run_id:
    # Query MLflow for metrics from previous run
    previous_run = mlflow.tracking.MlflowClient().get_run(base_model_run_id)
    if "total_timesteps" in previous_run.data.metrics:
        previous_timesteps = int(previous_run.data.metrics["total_timesteps"])

# Pass to the callback
unified_callback = MLflowLoggingCallback(
    # other parameters...
    previous_timesteps=previous_timesteps
)
```

### Generating Continuous Plots

The `MLflowMetricConnector` class provides tools to analyze and visualize performance across runs:

```python
from RL.Utils.mlflow_manager import MLflowMetricConnector

# Initialize the connector
connector = MLflowMetricConnector(experiment_name="your_experiment_name")

# Find runs in an enhancement chain
runs = connector.get_run_chain("final_run_id")
run_ids = [run.info.run_id for run in runs]

# Generate a continuous plot for a specific metric
connector.generate_continuous_plot(
    run_ids,
    "evaluation/mean_reward",
    title="Reward Progress Across Training Runs",
    save_path="reward_progress.png"
)

# Create a comprehensive enhancement report
report_data, html_path = connector.generate_enhancement_report(
    "final_run_id",
    output_path="enhancement_report.html"
)
```

## Example Script

An example script showing how to use these features is included at `RL/Examples/continuous_metrics_example.py`. You can run it as follows:

```bash
python -m RL.Examples.continuous_metrics_example \
  --experiment-name "stock_trading_rl_1H" \
  --final-run-id "your_enhanced_model_run_id" \
  --output-path "enhancement_report.html" \
  --metrics "evaluation/mean_reward" "portfolio/total_return" "portfolio/sharpe_ratio"
```

## Best Practices

1. **Always link enhancements**: When enhancing a model, always set the `base_model_run_id` tag to establish the enhancement chain
2. **Log previous timesteps**: Always log the previous timesteps metric to ensure accurate tracking
3. **Use consistent metric names**: Keep metric names consistent across runs for proper tracking
4. **Generate reports after multiple enhancements**: Enhancement reports are most useful after several training iterations

## Limitations

1. The system relies on MLflow run tags to track the enhancement chain, so proper tagging is essential
2. Very large discrepancies in metric values between runs may make visualization difficult
3. The system assumes monotonically increasing timesteps across runs
