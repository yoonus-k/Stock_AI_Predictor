# Stock AI Predictor Hugging Face Deployment

This documentation outlines the deployment architecture for the Stock AI Predictor project on Hugging Face Hub. The deployment consists of multiple components that work together to provide a complete trading prediction system.

## Architecture Overview

![Architecture Diagram](https://mermaid.ink/img/pako:eNqNk0FPwzAMhf-K5QMc2HQYbGzSNE6T4MABiUM20Uy0zZZEapLBNO2_EzdpV6YJiVNiv_f82XE8YGoUYYxPxiJ4MNZAFqQVXroU9Ahi7YtHjkE5URmcEjgjNbIkVDuvHK-RY4ZbZ3N_WyMsTVai2LqSWTfP91pT5RMYpRV3N9N28FQNKV8jqimpwnJwf_EF88UQzsL3zvD7r8CVofxnt-_u7u3pwhw1tallfzvOED3ZXCoejHYLhFlPm1FEX7cjtlubVWKPXEmO-VFaoj9NUw9I2QZrLZ49bRqbrPrngKR-FAVJ3dHJIcCmLAeBxFUWcvmVqJWg0hOaKO2sCEgs0yZn7FSDYrabWb5lAew5mTovrfUt5_orBiTNnLIBCY9x3tQ99SVpDJyWAlIMybm-LqHoS83dZDLxZGVO3ihH6nFbsjmYjRa_vqTo899j7qT3WqzA_1cZe4DRXWvLAuMnjK-wNrk3vThcxzheJ9nvD-DMuSI)

## Components

### 1. Models on Hugging Face Hub

- **Parameter Tester Model**: Optimizes trading parameters for various patterns
  - Repository: `your_username/stock-ai-parameter-tester`
  - Functionality: Finds optimal trading parameters for identified patterns

- **RL Trading Model**: Reinforcement learning model for trading decisions
  - Repository: `your_username/stock-ai-rl-trader`
  - Functionality: Makes trading decisions based on market state

### 2. Dataset on Hugging Face Datasets

- **Stock Market Dataset**: Contains historical price data and derived features
  - Repository: `your_username/stock-market-data`
  - Updates: Weekly through scheduled Kaggle notebook

### 3. API on Hugging Face Spaces

- **Stock AI Predictor API**: Serves predictions and trading signals
  - Space: `your_username/stock-ai-predictor-api`
  - Endpoints:
    - `/api/parameters/optimize`: Optimize parameters for a symbol/timeframe
    - `/api/trading/signal`: Get trading signal for a symbol/timeframe

### 4. Trading Client (Local)

- **Trading Bot**: Connects to the API and executes trades on trading platforms
  - Supports: MT5, Binance, and other platforms
  - Usage: Run locally to execute trades based on API signals

### 5. Automated Workflows (Kaggle)

- **Dataset Updates**: Weekly updates to the dataset
- **Model Retraining**: Periodic retraining of models

## Deployment Steps

1. **Set Up Hugging Face Account and Repositories**
   - Create an account on Hugging Face
   - Create model, dataset, and space repositories

2. **Deploy Initial Models and Dataset**
   - Run `deploy_to_huggingface.py` to upload models and data

3. **Deploy API on Hugging Face Spaces**
   - Configure and deploy the FastAPI application

4. **Set Up Kaggle Automation**
   - Upload and schedule the Kaggle notebook for regular updates

5. **Configure Trading Client**
   - Set up the trading client with your API keys and preferences

## Usage

### Running the Trading Client

```bash
# Configure environment variables
cp Deployment/.env.example .env
# Edit .env with your API keys and preferences

# Run the trading client
python Deployment/trading_client.py --symbol AAPL --timeframe 1D --platform mt5
```

### Updating Models and Dataset Manually

```bash
# Deploy all components
python Deployment/deploy_to_huggingface.py --all

# Deploy specific components
python Deployment/deploy_to_huggingface.py --param-tester
python Deployment/deploy_to_huggingface.py --rl-model
python Deployment/deploy_to_huggingface.py --dataset
python Deployment/deploy_to_huggingface.py --api-space
```

### Scheduling Workflows

The project includes a workflow scheduler that can be run to update datasets and retrain models:

```bash
# Update dataset
python Deployment/workflow_scheduler.py --update-dataset

# Retrain models
python Deployment/workflow_scheduler.py --retrain-models

# Run on Kaggle (for Kaggle automated scripts)
python Deployment/workflow_scheduler.py --kaggle
```

## Security Considerations

1. **API Key**: Protect your API key to prevent unauthorized access
2. **Trading Platform Credentials**: Store securely and never commit to version control
3. **Hugging Face Token**: Secure your Hugging Face token with appropriate permissions

## Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## Troubleshooting

If you encounter any issues with the deployment, check the following:

1. **Environment Variables**: Ensure all required environment variables are set
2. **API Access**: Verify that your API keys and tokens are valid
3. **Logs**: Check the log files for errors (`hf_deployment.log`, `trading_client.log`)
4. **Hugging Face Status**: Check if Hugging Face services are operational
