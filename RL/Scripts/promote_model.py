#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Promotion Script for Stock AI Predictor

This script allows promotion of models between different stages in MLflow:
- development: Initial stage for all models
- beta: Models that have passed initial validation
- champion: Production-ready models
- archived: Previously used models that are no longer active

Usage:
    python -m RL.Scripts.promote_model --timeframe 1H --model-version 1 --target-stage beta
    python -m RL.Scripts.promote_model --timeframe 1H --list-models --stage champion
"""

import argparse
import os
import sys
from tabulate import tabulate
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RL.Mlflow.mlflow_manager import MLflowManager

def parse_args():
    parser = argparse.ArgumentParser(description="Promote models between stages in MLflow")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe of the model (e.g. '1H', '4H', 'D')")
    parser.add_argument("--model-version", type=str, help="Model version to promote")
    parser.add_argument("--target-stage", type=str, choices=["development", "beta", "champion", "archived"], 
                        help="Target stage to promote the model to")
    parser.add_argument("--reason", type=str, help="Reason for promotion")
    parser.add_argument("--list-models", action="store_true", help="List models in the specified stage")
    parser.add_argument("--stage", type=str, choices=["development", "beta", "champion", "archived"],
                       help="Filter models by stage when listing")
    
    return parser.parse_args()

def list_models(mlflow_manager, stage, timeframe):
    """List models by stage"""
    models = mlflow_manager.find_models_by_stage(stage, timeframe)
    
    if not models:
        print(f"No models found in stage '{stage}' for timeframe '{timeframe}'")
        return
    
    # Prepare data for tabulate
    headers = ["Model Name", "Version", "MLflow Stage", "Promotion Stage", "Created", "Last Updated"]
    rows = []
    
    for model in models:
        # Format timestamps for readability
        created = datetime.fromtimestamp(model["creation_timestamp"]/1000).strftime("%Y-%m-%d %H:%M:%S")
        updated = datetime.fromtimestamp(model["last_updated_timestamp"]/1000).strftime("%Y-%m-%d %H:%M:%S")
        
        rows.append([
            model["name"],
            model["version"],
            model["stage"],
            model["promotion_stage"],
            created,
            updated
        ])
    
    # Print table
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def promote_model(mlflow_manager, timeframe, version, target_stage, reason=None):
    """Promote a model to a new stage"""
    model_name = f"{timeframe}_trading_model"
    
    # Check if model exists
    try:
        existing_model = mlflow_manager.mlflow_client.get_model_version(name=model_name, version=version)
        if not existing_model:
            print(f"Model {model_name} version {version} not found")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Promote the model
    result = mlflow_manager.promote_model(
        model_name=model_name,
        version=version,
        target_stage=target_stage,
        reason=reason
    )
    
    if result:
        print(f"✅ Successfully promoted model {model_name} version {version} to {target_stage}")
        
        # If promoting to champion, give extra feedback
        if target_stage == "champion":
            print("\n🏆 NEW CHAMPION MODEL 🏆")
            print(f"Model {model_name} version {version} is now the champion model")
            print("All previous champion models have been archived\n")
        
        return True
    else:
        print(f"❌ Failed to promote model {model_name} version {version}")
        return False

def main():
    args = parse_args()
    
    # Initialize MLflow manager with the specified timeframe
    mlflow_manager = MLflowManager(timeframe=args.timeframe)
    
    # List models if requested
    if args.list_models:
        if args.stage:
            list_models(mlflow_manager, args.stage, args.timeframe)
        else:
            print("Please specify --stage when using --list-models")
        return
    
    # Validate required args for promotion
    if not args.model_version or not args.target_stage:
        print("Error: --model-version and --target-stage are required for promotion")
        return
    
    # Promote the model
    promote_model(mlflow_manager, args.timeframe, args.model_version, args.target_stage, args.reason)

if __name__ == "__main__":
    main()
