# Stock AI Predictor Model Tagging System

## Overview

The Stock AI Predictor uses a comprehensive tagging system in MLflow to track models throughout their lifecycle. This system enables efficient model discovery, enhancement, versioning, and promotion with clear lineage tracing.

## Primary Tags

| Tag Name | Description | Possible Values | Example |
|----------|-------------|-----------------|---------|
| `timeframe` | Time period for the model | "1H", "4H", "D", "30M", "15M", "5M", "1M" | "1H" |
| `model_type` | Type of model | "base", "continued", "curriculum", "adaptive" | "base" |
| `version_type` | Version status of the model | "latest", "old" | "latest" |
| `promotion_stage` | The promotion stage of the model | "development", "beta", "champion", "archived" | "development" |
| `enhancement_type` | Type of enhancement applied | "continued", "replay", "curriculum", "adaptive" | "continued" |
| `base_model_path` | Path to the base model used | ML framework URI path | "runs:/abc123/models/1H/model.zip" |
| `base_model_run_id` | ML framework run ID of the base model | Run ID string | "abc123def456" |

## Model Lifecycle and Tagging

### Scenario 1: New Base Model

When a new model is trained from scratch:

| Tag | Value |
|-----|-------|
| `model_type` | "base" |
| `version_type` | "latest" |
| `timeframe` | e.g., "1H" |
| `promotion_stage` | "development" |

### Scenario 2: Enhanced Model (First Enhancement)

When a base model is enhanced:

| Tag | Value |
|-----|-------|
| `model_type` | Enhancement type (e.g., "continued") |
| `version_type` | "latest" |
| `timeframe` | Same as base model (e.g., "1H") |
| `promotion_stage` | "development" |
| `enhancement_type` | Enhancement method (e.g., "continued") |
| `base_model_path` | Path to base model |
| `base_model_run_id` | Run ID of base model |

*Note: The base model's `version_type` is automatically changed from "latest" to "old"*

### Scenario 3: Continued Enhancement (Enhancing an Enhanced Model)

When an already enhanced model is further enhanced:

| Tag | Value |
|-----|-------|
| `model_type` | Enhancement type (e.g., "continued") |
| `version_type` | "latest" |
| `timeframe` | Same as previous model (e.g., "1H") |
| `promotion_stage` | "development" |
| `enhancement_type` | Enhancement method (e.g., "continued") |
| `base_model_path` | Path to previous enhanced model |
| `base_model_run_id` | Run ID of previous enhanced model |

*Note: The previous enhanced model's `version_type` is automatically changed from "latest" to "old"*

## Model Promotion System

The model promotion system allows moving models through different stages of maturity:

### Promotion Stages

1. **Development** - Initial stage for all new and enhanced models
2. **Beta** - Models that have passed initial validation and are ready for more extensive testing
3. **Champion** - Production-ready models that have passed extensive validation and are the best available
4. **Archived** - Previously used models that have been replaced or are no longer active

### Promotion Workflow

1. Models start in the **Development** stage when they are registered
2. After initial validation, models can be promoted to **Beta**
3. After extensive validation, Beta models can be promoted to **Champion**
4. When a new Champion model is promoted, the previous Champion is automatically demoted to **Archived**

### Valid Stage Transitions

| Current Stage | Valid Target Stages |
|---------------|---------------------|
| development   | beta, archived      |
| beta          | development, champion, archived |
| champion      | beta, archived      |
| archived      | development, beta   |

### Promotion Tags

When a model is promoted, additional tags are added to track its history:

| Tag | Description | Example Value |
|-----|-------------|---------------|
| `promotion_{timestamp}` | Record of promotion | "development -> beta" |
| `last_promoted_at` | Timestamp of last promotion | "20250610_1417" |
| `promotion_reason` | Reason for promotion | "Improved Sharpe ratio by 15%" |
| `demotion_reason` | Reason for demotion (when applicable) | "New champion promoted" |
| `demotion_timestamp` | When demotion occurred (when applicable) | "20250610_1417" |

## Model Finding Methods

### Method 1: Find Latest Model of a Specific Type

```python
base_run_id, model_path = mlflow_manager.find_latest_model(
    model_type="continued",  # Can be any model type: "base", "continued", etc.
    version_type="latest"
)
```

### Method 2: Find Best Model of a Specific Type

```python
base_run_id, model_path = mlflow_manager.find_best_model(
    model_type="continued",  # Can be any model type: "base", "continued", etc.
    metric="evaluation/best_mean_reward"
)
```

### Method 3: Find Models by Promotion Stage

```python
models = mlflow_manager.find_models_by_stage(
    promotion_stage="champion",  # Can be any stage: "development", "beta", "champion", "archived"
    timeframe="1H"  # Optional timeframe filter
)
```

## Auto-Fallback Mechanism

If a model of the specified type (`enhance_model_type`) is not found, the system automatically attempts to find a `"base"` model as a fallback.

## Practical Examples

### Training a New Base Model

```bash
python -m RL.Scripts.Train.train_timeframe_model --timeframe 1H
```

### Enhancing a Base Model

```bash
python -m RL.Scripts.Train.train_timeframe_model --timeframe 1H --enhance best --enhance-type continued
```

### Enhancing a Continued Model (Further Enhancement)

```bash
python -m RL.Scripts.Train.train_timeframe_model --timeframe 1H --enhance best --enhance-model-type continued --enhance-type continued
```

### Promoting a Model to Beta Stage

```bash
python -m RL.Scripts.promote_model --timeframe 1H --model-version 1 --target-stage beta --reason "Passed initial validation"
```

### Promoting a Beta Model to Champion

```bash
python -m RL.Scripts.promote_model --timeframe 1H --model-version 2 --target-stage champion --reason "Best performance in backtesting"
```

### Listing All Champion Models

```bash
python -m RL.Scripts.promote_model --timeframe 1H --list-models --stage champion
```

## Additional Tags for Model Registry

When models are registered in the Model Registry, additional tags are applied:

| Tag | Description | Example Value |
|-----|-------------|---------------|
| `enhancement_timestamp` | When the enhancement was performed | "20250610_1417" |
| `previous_timesteps` | Training steps in the base model | "100000" |
| `total_timesteps` | Total training steps (base + new) | "150000" |

## Improvement Metrics

When a model is enhanced, improvement metrics are automatically calculated and logged with the prefix `improvement_`. For example:

| Metric | Description | Example Value |
|--------|-------------|--------------|
| `improvement_final/mean_reward` | Change in mean reward | "+15.72%" |
| `improvement_portfolio/sharpe_ratio` | Change in Sharpe ratio | "+8.35%" |
| `improvement_portfolio/win_rate` | Change in win rate | "+3.21%" |

These metrics allow for easy tracking of model performance improvements across generations.
