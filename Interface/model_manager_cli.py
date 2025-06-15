#!/usr/bin/env python
"""
MLflow Model Manager CLI
-----------------------
A comprehensive tool for managing MLflow models in the Stock AI Predictor project.
This CLI tool allows you to:
1. List experiments and runs with filtering options
2. View detailed information about specific runs
3. Register models from successful runs
4. View and manage registered models
5. Promote/demote models between stages
6. Compare models based on metrics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom utilities
from RL.Mlflow.mlflow_manager import MLflowManager

# Initialize rich console for pretty output
console = Console()

# Constants
DEFAULT_EXPERIMENT_NAME = "stock_trading_rl"
DEFAULT_TIMEFRAME = "1H"
VALID_STAGES = ["development", "beta", "champion", "archived"]
STAGE_COLORS = {
    "development": "blue",
    "beta": "yellow",
    "champion": "green",
    "archived": "red"
}

def setup_mlflow() -> MlflowClient:
    """Set up MLflow client with proper tracking URI"""
    try:
        # Ensure Windows-compatible file URL with triple slash for absolute paths
        mlruns_path = project_root / "mlruns"
        # Debug: Print the project_root and mlruns_path
        console.print(f"[dim]Project root: {project_root}[/dim]")
        console.print(f"[dim]Checking for MLflow data at: {mlruns_path}[/dim]")
        
        # Use file:/// prefix for proper URI format on Windows
        tracking_uri = f"file:///{str(mlruns_path).replace(chr(92), '/')}"
        
        console.print(f"Using MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        return client
    except Exception as e:
        console.print(f"[bold red]Error setting up MLflow client: {e}[/bold red]")
        console.print(Panel(
            f"Try providing the correct tracking URI format:\n"
            f"[green]For Windows:[/green] file:///D:/path/to/mlruns (with three slashes)\n"
            f"[green]For Linux/Mac:[/green] file:///path/to/mlruns", 
            title="MLflow URI Help"
        ))
        sys.exit(1)

def get_experiment_id(client: MlflowClient, experiment_name: str, timeframe: str = None) -> str:
    """Get experiment ID by name, optionally with timeframe"""
    # Form experiment name with timeframe if provided
    if timeframe:
        full_experiment_name = f"{experiment_name}_{timeframe}"
    else:
        full_experiment_name = experiment_name
    
    experiment = client.get_experiment_by_name(full_experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        console.print(f"[bold red]Experiment '{full_experiment_name}' not found.[/bold red]")
        sys.exit(1)

def format_duration(duration_seconds: float) -> str:
    """Format duration in seconds to a human-readable string"""
    if duration_seconds < 60:
        return f"{duration_seconds:.1f}s"
    elif duration_seconds < 3600:
        return f"{duration_seconds/60:.1f}m"
    else:
        return f"{duration_seconds/3600:.1f}h"

def format_datetime(timestamp_ms: float) -> str:
    """Format timestamp to a readable date string"""
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def list_experiments(args: argparse.Namespace) -> None:
    """List all MLflow experiments with their IDs and counts"""
    client = setup_mlflow()
    experiments = client.search_experiments()
    
    # Create rich table
    table = Table(title="Available MLflow Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Run Count", style="magenta")
    table.add_column("Creation Date", style="yellow")
    
    for exp in experiments:
        # Count runs in the experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
        run_count = len(runs)
        
        # Add to table
        table.add_row(
            exp.experiment_id, 
            exp.name, 
            str(run_count),
            format_datetime(exp.creation_time)
        )
    
    console.print(table)

def list_runs(args: argparse.Namespace) -> None:
    """List all runs for a given experiment with optional filters"""
    client = setup_mlflow()
    
    # Get experiment ID
    experiment_id = get_experiment_id(client, args.experiment_name, args.timeframe)
    
    # Build filter string
    filter_parts = []
    if args.model_type:
        filter_parts.append(f"tags.model_type = '{args.model_type}'")
    if args.version_type:
        filter_parts.append(f"tags.version_type = '{args.version_type}'")
    if args.enhancement_type:
        filter_parts.append(f"tags.enhancement_type = '{args.enhancement_type}'")
    
    filter_str = " and ".join(filter_parts) if filter_parts else ""
    
    # Order by creation time (most recent first)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_str,
        order_by=["attributes.start_time DESC"],
        max_results=args.limit
    )
    
    if not runs:
        console.print(f"[bold yellow]No runs found with the specified filters.[/bold yellow]")
        return
    
    # Create rich table
    table = Table(title=f"MLflow Runs for Experiment: {args.experiment_name}" + 
                        (f"_{args.timeframe}" if args.timeframe else ""))
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Start Time", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Model Type", style="blue")
    table.add_column("Version", style="yellow")
    table.add_column("Mean Reward", style="green")
    table.add_column("Sharpe Ratio", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Timesteps", style="blue")
    
    for run in runs:
        run_id = run.info.run_id
        run_info = run.info
        run_data = run.data
        
        # Extract relevant metrics and tags
        metrics = run_data.metrics
        tags = run_data.tags
        
        # Get model type and version
        model_type = tags.get("model_type", "unknown")
        version_type = tags.get("version_type", "unknown")
        
        # Get primary metrics if available
        mean_reward = metrics.get("evaluation/best_mean_reward", metrics.get("final_eval_mean_reward", "N/A"))
        sharpe_ratio = metrics.get("portfolio_sharpe_ratio", "N/A")
        
        # Calculate duration
        duration = format_duration((run_info.end_time or datetime.now().timestamp() * 1000) - run_info.start_time)
        
        # Get timesteps
        timesteps = metrics.get("total_timesteps", metrics.get("current_run_timesteps", "N/A"))
        
        # Color status
        status = run_info.status
        status_style = {
            "FINISHED": "[green]FINISHED[/green]",
            "RUNNING": "[blue]RUNNING[/blue]",
            "FAILED": "[red]FAILED[/red]"
        }.get(status, status)
        
        # Determine version style
        version_style = {
            "latest": f"[green]{version_type}[/green]",
            "old": f"[yellow]{version_type}[/yellow]"
        }.get(version_type, version_type)
        
        # Format metrics
        if isinstance(mean_reward, (int, float)):
            mean_reward = f"{mean_reward:.2f}"
        if isinstance(sharpe_ratio, (int, float)):
            sharpe_ratio = f"{sharpe_ratio:.2f}"
            
        # Add row
        table.add_row(
            run_id[:8] + "...",
            format_datetime(run_info.start_time),
            status_style,
            model_type,
            version_style,
            str(mean_reward),
            str(sharpe_ratio),
            duration,
            str(timesteps)
        )
    
    console.print(table)
    console.print(f"\n[bold]Total runs: {len(runs)}[/bold]")
    console.print("[italic]To view full details of a run, use the 'show-run' command with the Run ID.[/italic]")

def show_run_details(args: argparse.Namespace) -> None:
    """Show detailed information about a specific run"""
    client = setup_mlflow()
    
    try:
        # Get run by ID
        run = client.get_run(args.run_id)
        
        # Print run information in a rich panel
        run_info = run.info
        run_data = run.data
        metrics = run_data.metrics
        tags = run_data.tags
        params = run_data.params
        
        # Create general info panel
        general_info = Table(show_header=False, box=None)
        general_info.add_column(style="cyan")
        general_info.add_column(style="white")
        
        general_info.add_row("Run ID:", run_info.run_id)
        general_info.add_row("Status:", run_info.status)
        general_info.add_row("Start Time:", format_datetime(run_info.start_time))
        if run_info.end_time:
            general_info.add_row("End Time:", format_datetime(run_info.end_time))
            duration = format_duration((run_info.end_time - run_info.start_time) / 1000)
            general_info.add_row("Duration:", duration)
        general_info.add_row("Experiment ID:", run_info.experiment_id)
        general_info.add_row("Artifact URI:", run_info.artifact_uri)
        
        model_name = tags.get("mlflow.runName", "Unknown")
        timeframe = tags.get("timeframe", "Unknown")
        model_type = tags.get("model_type", "Unknown")
        version_type = tags.get("version_type", "Unknown")
        enhancement_type = tags.get("enhancement_type", "None")
        
        general_info.add_row("Model Name:", model_name)
        general_info.add_row("Timeframe:", timeframe)
        general_info.add_row("Model Type:", model_type)
        general_info.add_row("Version:", version_type)
        general_info.add_row("Enhancement:", enhancement_type)
        
        if "base_model_run_id" in tags:
            general_info.add_row("Base Model:", tags["base_model_run_id"])
        
        console.print(Panel(general_info, title=f"Run Details", border_style="blue"))
        
        # Create metrics panel (show most important metrics)
        key_metrics = [
            "evaluation/best_mean_reward", "final_eval_mean_reward",
            "portfolio_sharpe_ratio", "portfolio_win_rate", "portfolio_profit_factor",
            "portfolio_max_drawdown", "total_timesteps", "previous_timesteps",
            "training_duration_minutes"
        ]
        
        metrics_table = Table(title="Key Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        # Add available metrics
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                # Format value if it's a float
                if isinstance(value, float):
                    value = f"{value:.4f}"
                metrics_table.add_row(metric, str(value))
        
        # Add enhancement metrics if available
        enhancement_metrics = {k: v for k, v in metrics.items() if k.startswith("improvement_")}
        if enhancement_metrics:
            metrics_table.add_row("", "")  # Empty row as separator
            metrics_table.add_row("[bold]Enhancement Improvements[/bold]", "")
            
            for metric, value in enhancement_metrics.items():
                formatted_name = metric.replace("improvement_", "")
                # Color improvements green (positive) or red (negative)
                if value > 0:
                    formatted_value = f"[green]+{value:.2f}%[/green]"
                else:
                    formatted_value = f"[red]{value:.2f}%[/red]"
                metrics_table.add_row(formatted_name, formatted_value)
        
        console.print(metrics_table)
        
        # Model parameters
        params_table = Table(title="Model Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="yellow")
        
        # List key parameters first, then others
        key_params = ["timesteps", "learning_rate", "batch_size", "n_steps", 
                     "n_epochs", "ent_coef", "reward_type", "normalize_observations"]
        
        # First add key parameters
        for param in key_params:
            if param in params:
                params_table.add_row(param, params[param])
        
        # Add separator
        params_table.add_row("", "")
        
        # Then add other parameters
        other_params = {k: v for k, v in params.items() if k not in key_params}
        for param, value in other_params.items():
            params_table.add_row(param, value)
        
        console.print(params_table)
        
        # Show artifact paths
        if args.show_artifacts:
            artifacts = client.list_artifacts(run_info.run_id)
            
            artifacts_table = Table(title="Artifacts")
            artifacts_table.add_column("Path", style="cyan")
            artifacts_table.add_column("Type", style="yellow")
            artifacts_table.add_column("Size", style="green")
            
            for artifact in artifacts:
                artifact_type = "Directory" if artifact.is_dir else "File"
                size = f"{artifact.file_size / 1024:.1f} KB" if hasattr(artifact, 'file_size') and artifact.file_size else "N/A"
                artifacts_table.add_row(artifact.path, artifact_type, size)
            
            console.print(artifacts_table)
        
        # Show registration options
        model_path = None
        for artifact in client.list_artifacts(run_info.run_id, "models"):
            if artifact.is_dir:
                timeframe_artifacts = client.list_artifacts(run_info.run_id, artifact.path)
                model_files = [a for a in timeframe_artifacts if a.path.endswith(".zip")]
                if model_files:
                    model_path = f"runs:/{run_info.run_id}/{model_files[0].path}"
                    break
        
        if model_path:
            console.print(Panel(
                f"This run contains a trained model that can be registered.\n"
                f"To register this model, use the command:\n\n"
                f"[bold]python model_manager_cli.py register-model --run-id {run_info.run_id} "
                f"--name {timeframe}_trading_model --stage development[/bold]",
                title="Model Registration", border_style="green"))
    
    except Exception as e:
        console.print(f"[bold red]Error retrieving run details: {e}[/bold red]")
        sys.exit(1)

def list_registered_models(args: argparse.Namespace) -> None:
    """List all registered models in MLflow"""
    client = setup_mlflow()
    
    try:
        # Get all registered models
        models = client.search_registered_models()
        
        if not models:
            console.print("[yellow]No registered models found.[/yellow]")
            return
        
        # Create table
        table = Table(title="Registered Models")
        table.add_column("Name", style="cyan")
        table.add_column("Latest Version", style="green")
        table.add_column("Current Stage", style="yellow")
        table.add_column("Promotion Stage", style="blue")
        table.add_column("Last Updated", style="magenta")
        
        for model in models:
            model_name = model.name
            
            # Get latest version
            latest_versions = client.get_latest_versions(model_name)
            if not latest_versions:
                continue
                
            latest = latest_versions[0]
            
            # Get stage tags
            promotion_stage = latest.tags.get("promotion_stage", "unknown")
            
            # Format stage with color
            stage_color = STAGE_COLORS.get(promotion_stage, "white")
            formatted_promotion_stage = f"[{stage_color}]{promotion_stage}[/{stage_color}]"
            
            table.add_row(
                model_name,
                str(latest.version),
                latest.current_stage,
                formatted_promotion_stage,
                format_datetime(latest.last_updated_timestamp)
            )
        
        console.print(table)
        console.print("[italic]To view details about a specific registered model, use the 'show-registered-model' command.[/italic]")
    
    except Exception as e:
        console.print(f"[bold red]Error listing registered models: {e}[/bold red]")

def show_registered_model(args: argparse.Namespace) -> None:
    """Show details of a specific registered model"""
    client = setup_mlflow()
    
    try:
        # Get model details
        model_versions = client.get_latest_versions(args.model_name)
        
        if not model_versions:
            console.print(f"[bold red]No versions found for model '{args.model_name}'[/bold red]")
            return
        
        # Create table
        table = Table(title=f"Versions of Model: {args.model_name}")
        table.add_column("Version", style="cyan")
        table.add_column("MLflow Stage", style="yellow")
        table.add_column("Promotion Stage", style="blue")
        table.add_column("Created", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Run ID", style="white")
        
        for version in model_versions:
            # Get promotion stage
            promotion_stage = version.tags.get("promotion_stage", "unknown")
            
            # Format stage with color
            stage_color = STAGE_COLORS.get(promotion_stage, "white")
            formatted_promotion_stage = f"[{stage_color}]{promotion_stage}[/{stage_color}]"
            
            # Format creation date
            creation_date = format_datetime(version.creation_timestamp)
            
            table.add_row(
                str(version.version),
                version.current_stage,
                formatted_promotion_stage,
                creation_date,
                version.status,
                version.run_id[:8] + "..." if version.run_id else "N/A"
            )
        
        console.print(table)
        
        # Show promotion options
        console.print(Panel(
            "To promote a model version to a new stage, use the command:\n"
            f"[bold]python model_manager_cli.py promote-model --name {args.model_name} "
            "--version <version> --stage <target_stage>[/bold]\n\n"
            "Valid stages: development, beta, champion, archived",
            title="Model Promotion Options", border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error showing registered model: {e}[/bold red]")

def register_model(args: argparse.Namespace) -> None:
    """Register a model from a specific run"""
    client = setup_mlflow()
    
    try:
        # Get run details
        run = client.get_run(args.run_id)
        run_info = run.info
        tags = run.data.tags
        
        # Find model artifact path
        model_path = None
        timeframe = tags.get("timeframe", "unknown")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Looking for model artifacts...", total=None)
            
            for artifact in client.list_artifacts(args.run_id, "models"):
                if artifact.is_dir:
                    if timeframe.lower() in artifact.path.lower():
                        timeframe_artifacts = client.list_artifacts(args.run_id, artifact.path)
                        model_files = [a for a in timeframe_artifacts if a.path.endswith(".zip")]
                        if model_files:
                            model_path = f"runs:/{args.run_id}/{model_files[0].path}"
                            break
            
            progress.update(task, completed=1)
        
        if not model_path:
            console.print(f"[bold red]No model artifacts found for run {args.run_id}[/bold red]")
            return
        
        # Create a MLflowManager instance for handling registration
        mlflow_manager = MLflowManager(
            experiment_name=tags.get("experiment", "stock_trading_rl"),
            timeframe=timeframe,
            model_type=tags.get("model_type", "base")
        )
        
        # Prepare version tags
        version_tags = {
            "timeframe": timeframe,
            "model_type": tags.get("model_type", "base"),
            "registration_timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        }
        
        # Add enhancement-specific tags if available
        if "base_model_run_id" in tags:
            version_tags["base_model_run_id"] = tags["base_model_run_id"]
            version_tags["enhancement_type"] = tags.get("enhancement_type", "enhanced")
        
        # Register the model
        version = mlflow_manager.register_model_version(
            model_path=model_path,
            model_name=args.name,
            base_run_id=tags.get("base_model_run_id"),
            stage=args.stage,
            promotion_stage=args.promotion_stage,
            tags=version_tags
        )
        
        console.print(f"[bold green]Successfully registered model '{args.name}' version {version}![/bold green]")
        console.print(f"[green]Model set to MLflow stage '{args.stage}' and promotion stage '{args.promotion_stage}'[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error registering model: {e}[/bold red]")
        sys.exit(1)

def promote_model(args: argparse.Namespace) -> None:
    """Promote a registered model to a new stage"""
    client = setup_mlflow()
    
    try:
        # Check if model exists
        model = client.get_registered_model(args.name)
        if not model:
            console.print(f"[bold red]Model '{args.name}' not found.[/bold red]")
            return
        
        # Get the specific version
        try:
            version = client.get_model_version(args.name, args.version)
        except Exception:
            console.print(f"[bold red]Version {args.version} of model '{args.name}' not found.[/bold red]")
            return
        
        # Create MLflowManager for promotion
        timeframe = version.tags.get("timeframe", "unknown")
        mlflow_manager = MLflowManager(
            experiment_name="stock_trading_rl",
            timeframe=timeframe,
            model_type=version.tags.get("model_type", "base")
        )
        
        # Validate target stage
        if args.stage not in VALID_STAGES:
            valid_stages_str = ", ".join(VALID_STAGES)
            console.print(f"[bold red]Invalid stage '{args.stage}'. Valid stages are: {valid_stages_str}[/bold red]")
            return
        
        # Get current stage
        current_stage = version.tags.get("promotion_stage", "unknown")
        
        # Ask for confirmation
        console.print(f"[bold]Promoting model '{args.name}' version {args.version}[/bold]")
        console.print(f"From stage: [blue]{current_stage}[/blue]")
        console.print(f"To stage: [{STAGE_COLORS[args.stage]}]{args.stage}[/{STAGE_COLORS[args.stage]}]")
        
        if args.stage == "champion":
            console.print("[yellow]Warning: Promoting to 'champion' will automatically demote the current champion to 'archived'.[/yellow]")
        
        if not args.yes:
            confirm = input("Proceed with promotion? [y/N]: ")
            if confirm.lower() not in ["y", "yes"]:
                console.print("[yellow]Promotion canceled.[/yellow]")
                return
        
        # Execute promotion
        success = mlflow_manager.promote_model(
            model_name=args.name,
            version=args.version,
            target_stage=args.stage,
            reason=args.reason
        )
        
        if success:
            console.print(f"[bold green]Successfully promoted model '{args.name}' version {args.version} to {args.stage}![/bold green]")
            
            # Check if we promoted to champion and a previous champion was demoted
            if args.stage == "champion":
                # Get all versions to find any that were demoted
                all_versions = client.get_latest_versions(args.name)
                demoted_versions = [v for v in all_versions 
                                  if v.version != args.version and
                                     v.tags.get("demotion_timestamp") is not None]
                
                if demoted_versions:
                    console.print("[yellow]The following versions were demoted to 'archived':[/yellow]")
                    for v in demoted_versions:
                        if "demotion_timestamp" in v.tags:
                            console.print(f"  - Version {v.version} (demoted at {v.tags['demotion_timestamp']})")
        else:
            console.print(f"[bold red]Failed to promote model.[/bold red]")
        
    except Exception as e:
        console.print(f"[bold red]Error promoting model: {e}[/bold red]")
        sys.exit(1)

def compare_models(args: argparse.Namespace) -> None:
    """Compare metrics between two model runs"""
    client = setup_mlflow()
    
    try:
        # Get run details for both runs
        run1 = client.get_run(args.run_id_1)
        run2 = client.get_run(args.run_id_2)
        
        run1_metrics = run1.data.metrics
        run2_metrics = run2.data.metrics
        
        run1_tags = run1.data.tags
        run2_tags = run2.data.tags
        
        # Prepare model info for display
        model1_info = {
            "run_id": args.run_id_1[:8] + "...",
            "model_type": run1_tags.get("model_type", "unknown"),
            "version": run1_tags.get("version_type", "unknown"),
            "timeframe": run1_tags.get("timeframe", "unknown")
        }
        
        model2_info = {
            "run_id": args.run_id_2[:8] + "...",
            "model_type": run2_tags.get("model_type", "unknown"),
            "version": run2_tags.get("version_type", "unknown"),
            "timeframe": run2_tags.get("timeframe", "unknown")
        }
        
        # Display model information
        console.print(Panel(
            f"[bold]Model 1:[/bold] {model1_info['model_type']} (version: {model1_info['version']}, timeframe: {model1_info['timeframe']})\n"
            f"[bold]Model 2:[/bold] {model2_info['model_type']} (version: {model2_info['version']}, timeframe: {model2_info['timeframe']})",
            title="Model Comparison", border_style="blue"
        ))
        
        # Get common metrics
        common_metrics = [m for m in run1_metrics if m in run2_metrics]
        
        # Filter metrics if specified
        if args.metrics:
            metric_filter = args.metrics.split(',')
            common_metrics = [m for m in common_metrics if any(filter_item in m for filter_item in metric_filter)]
        
        # Calculate improvements
        improvements = {}
        for metric in common_metrics:
            value1 = run1_metrics[metric]
            value2 = run2_metrics[metric]
            
            if abs(value1) > 0:
                improvement_pct = ((value2 - value1) / abs(value1)) * 100
            else:
                improvement_pct = float('inf') if value2 > 0 else float('-inf') if value2 < 0 else 0.0
            
            improvements[metric] = improvement_pct
        
        # Create comparison table
        table = Table(title="Metric Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column(f"Model 1 ({model1_info['model_type']})", style="blue")
        table.add_column(f"Model 2 ({model2_info['model_type']})", style="green")
        table.add_column("Difference", style="yellow")
        table.add_column("% Change", style="magenta")
        
        # Sort metrics by absolute percent change
        sorted_metrics = sorted(common_metrics, key=lambda x: abs(improvements.get(x, 0)), reverse=True)
        
        for metric in sorted_metrics:
            value1 = run1_metrics[metric]
            value2 = run2_metrics[metric]
            diff = value2 - value1
            pct_change = improvements[metric]
            
            # Format values
            if isinstance(value1, float):
                value1_str = f"{value1:.4f}"
            else:
                value1_str = str(value1)
                
            if isinstance(value2, float):
                value2_str = f"{value2:.4f}"
            else:
                value2_str = str(value2)
                
            if isinstance(diff, float):
                diff_str = f"{diff:.4f}"
            else:
                diff_str = str(diff)
            
            # Format percent change with color
            if pct_change > 0:
                pct_str = f"[green]+{pct_change:.2f}%[/green]"
            elif pct_change < 0:
                pct_str = f"[red]{pct_change:.2f}%[/red]"
            else:
                pct_str = "0.00%"
            
            table.add_row(metric, value1_str, value2_str, diff_str, pct_str)
        
        console.print(table)
        
        # Create visual comparison charts if requested
        if args.visualize:
            # Filter to most important metrics for chart
            key_metrics = ["evaluation/best_mean_reward", "final_eval_mean_reward",
                          "portfolio_sharpe_ratio", "portfolio_win_rate", 
                          "portfolio_profit_factor", "portfolio_max_drawdown"]
            
            chart_metrics = [m for m in sorted_metrics if m in key_metrics or "improvement_" in m][:10]
            
            if chart_metrics:
                plt.figure(figsize=(12, 6))
                
                # Prepare data for chart
                metric_names = [m.split('/')[-1] for m in chart_metrics]
                values1 = [run1_metrics[m] for m in chart_metrics]
                values2 = [run2_metrics[m] for m in chart_metrics]
                
                x = range(len(chart_metrics))
                width = 0.35
                
                plt.bar([i - width/2 for i in x], values1, width, label=f"Model 1 ({model1_info['model_type']})")
                plt.bar([i + width/2 for i in x], values2, width, label=f"Model 2 ({model2_info['model_type']})")
                
                plt.xlabel('Metrics')
                plt.ylabel('Value')
                plt.title('Model Comparison')
                plt.xticks(x, metric_names, rotation=45)
                plt.legend()
                plt.tight_layout()
                
                # Save to temporary file
                temp_chart_path = os.path.join(os.getcwd(), "model_comparison_chart.png")
                plt.savefig(temp_chart_path)
                
                console.print(f"[green]Chart saved to: {temp_chart_path}[/green]")
    
    except Exception as e:
        console.print(f"[bold red]Error comparing models: {e}[/bold red]")
        sys.exit(1)

def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser"""
    parser = argparse.ArgumentParser(description="MLflow Model Manager for Stock AI Predictor")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List experiments command
    list_exp_parser = subparsers.add_parser("list-experiments", help="List all MLflow experiments")
    
    # List runs command
    list_runs_parser = subparsers.add_parser("list-runs", help="List MLflow runs")
    list_runs_parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME,
                                 help="Name of the MLflow experiment")
    list_runs_parser.add_argument("--timeframe", type=str, default=None,
                                 help="Timeframe filter (e.g., '1H', 'D')")
    list_runs_parser.add_argument("--model-type", type=str, default=None,
                                 help="Filter by model type (e.g., 'base', 'continued')")
    list_runs_parser.add_argument("--version-type", type=str, default=None,
                                 help="Filter by version type (e.g., 'latest', 'old')")
    list_runs_parser.add_argument("--enhancement-type", type=str, default=None,
                                 help="Filter by enhancement type (e.g., 'continued', 'curriculum')")
    list_runs_parser.add_argument("--limit", type=int, default=25,
                                 help="Maximum number of runs to display")
    
    # Show run details command
    show_run_parser = subparsers.add_parser("show-run", help="Show details for a specific run")
    show_run_parser.add_argument("run_id", type=str,
                                help="ID of the run to display")
    show_run_parser.add_argument("--show-artifacts", action="store_true",
                                help="Show list of artifacts")
    
    # List registered models command
    list_models_parser = subparsers.add_parser("list-models", help="List registered models")
    
    # Show registered model details
    show_model_parser = subparsers.add_parser("show-model", help="Show details for a registered model")
    show_model_parser.add_argument("model_name", type=str,
                                  help="Name of the registered model")
    
    # Register model command
    register_parser = subparsers.add_parser("register-model", help="Register a model from a run")
    register_parser.add_argument("--run-id", type=str, required=True,
                               help="ID of the run containing the model to register")
    register_parser.add_argument("--name", type=str, required=True,
                               help="Name to register the model under")
    register_parser.add_argument("--stage", type=str, default="Staging",
                               choices=["None", "Staging", "Production", "Archived"],
                               help="MLflow stage for the model")
    register_parser.add_argument("--promotion-stage", type=str, default="development",
                               choices=["development", "beta", "champion", "archived"],
                               help="Custom promotion stage for the model")
    
    # Promote model command
    promote_parser = subparsers.add_parser("promote-model", help="Promote a registered model to a new stage")
    promote_parser.add_argument("--name", type=str, required=True,
                              help="Name of the registered model")
    promote_parser.add_argument("--version", type=str, required=True,
                              help="Version of the model to promote")
    promote_parser.add_argument("--stage", type=str, required=True,
                              choices=["development", "beta", "champion", "archived"],
                              help="Target stage for promotion")
    promote_parser.add_argument("--reason", type=str, default=None,
                              help="Reason for the promotion")
    promote_parser.add_argument("--yes", "-y", action="store_true",
                              help="Skip confirmation prompt")
    
    # Compare models command
    compare_parser = subparsers.add_parser("compare-models", help="Compare metrics between two runs")
    compare_parser.add_argument("--run-id-1", type=str, required=True,
                              help="ID of the first run to compare")
    compare_parser.add_argument("--run-id-2", type=str, required=True,
                              help="ID of the second run to compare")
    compare_parser.add_argument("--metrics", type=str, default=None,
                              help="Comma-separated list of metrics to compare (substring match)")
    compare_parser.add_argument("--visualize", action="store_true",
                              help="Generate visual comparison charts")
    
    return parser

def main():
    """Main entry point for the CLI"""
    parser = build_parser()
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Execute the appropriate command
    commands = {
        "list-experiments": list_experiments,
        "list-runs": list_runs,
        "show-run": show_run_details,
        "list-models": list_registered_models,
        "show-model": show_registered_model,
        "register-model": register_model,
        "promote-model": promote_model,
        "compare-models": compare_models
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    # Add a nice banner
    print("""
╔════════════════════════════════════════════════════╗
║                                                    ║
║  MLflow Model Manager for Stock AI Predictor       ║
║                                                    ║
╚════════════════════════════════════════════════════╝
""")
    
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(0)
