# Model Promotion System

## Overview

The Model Promotion System enables a structured approach to managing model lifecycles in the Stock AI Predictor project. It allows promoting models through different maturity stages from development to production.

## Promotion Stages

Models in the system can be in one of four promotion stages:

1. **Development**: Initial stage for all newly registered models. Models in this stage are experimental and undergoing initial validation.

2. **Beta**: Models that have passed initial validation and are considered good candidates for production. Beta models undergo more extensive validation before being considered for production.

3. **Champion**: The best validated model that is currently in production. Only one model per timeframe can be a champion at any given time.

4. **Archived**: Previously used models (including old champions) that are no longer active but kept for reference.

## Command-Line Interface

The system provides a command-line interface to promote models between stages:

### Promoting a Model to Beta

```bash
python -m RL.Scripts.promote_model --timeframe 1H --model-version 1 --target-stage beta --reason "Passed initial validation"
```

### Promoting a Model to Champion

```bash
python -m RL.Scripts.promote_model --timeframe 1H --model-version 2 --target-stage champion --reason "Best performance in backtesting"
```

### Listing Models by Stage

```bash
# List all champion models for 1H timeframe
python -m RL.Scripts.promote_model --timeframe 1H --list-models --stage champion

# List all beta models for 1H timeframe
python -m RL.Scripts.promote_model --timeframe 1H --list-models --stage beta

# List all development models for 1H timeframe
python -m RL.Scripts.promote_model --timeframe 1H --list-models --stage development

# List all archived models for 1H timeframe
python -m RL.Scripts.promote_model --timeframe 1H --list-models --stage archived
```

## Programmatic Usage

The promotion system can also be used programmatically:

```python
from RL.Utils.mlflow_manager import MLflowManager

# Initialize with specific timeframe
mlflow_manager = MLflowManager(timeframe="1H")

# Promote a model to beta
mlflow_manager.promote_model(
    model_name="1H_trading_model",  # Always format as "{timeframe}_trading_model"
    version="1",                     # The model version to promote
    target_stage="beta",            # Target stage: development, beta, champion, or archived
    reason="Passed initial validation tests"  # Optional reason for promotion
)

# Find all models in beta stage for a timeframe
beta_models = mlflow_manager.find_models_by_stage("beta", timeframe="1H")
```

## Automatic Demotion

When promoting a model to champion stage, any existing champion model for that timeframe is automatically demoted to archived stage.

## Usage Example

See the included example script for a demonstration of the promotion workflow:

```bash
python -m RL.Examples.model_promotion_example
```

This script demonstrates promoting models through the different stages and shows how automatic demotion works when a new champion is promoted.
