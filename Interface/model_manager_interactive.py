#!/usr/bin/env python
"""
Interactive MLflow Model Manager CLI
-----------------------------------
A menu-driven interface for managing MLflow models in the Stock AI Predictor project.

This tool allows you to:
1. Browse and search experiments and runs
2. View detailed information about specific runs
3. Register models from successful runs
4. View and manage registered models
5. Promote/demote models between stages
6. Compare models based on metrics
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import from the existing model_manager_cli module
from Interface.model_manager_cli import (
    setup_mlflow, list_experiments, list_runs, show_run_details,
    list_registered_models, show_registered_model, register_model,
    promote_model, compare_models, DEFAULT_EXPERIMENT_NAME, DEFAULT_TIMEFRAME,
    VALID_STAGES, STAGE_COLORS
)

# Rich for terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.text import Text
from rich.style import Style

# Initialize console
console = Console()

# Mock args namespace for compatibility with existing functions
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """Print a header for menus"""
    clear_screen()
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(60)}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def print_menu(title: str, options: List[Tuple[str, str]]):
    """
    Print a menu with options
    
    Args:
        title: The title of the menu
        options: List of tuples (option_key, option_description)
    """
    print_header(title)
    
    for i, (key, description) in enumerate(options):
        console.print(f"  [bold cyan]{key}[/bold cyan]: {description}")
    
    console.print("\n")


def wait_for_key():
    """Wait for user to press a key"""
    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()


def get_run_id_input():
    """Get run ID input from user"""
    run_id = Prompt.ask("[bold cyan]Enter Run ID[/bold cyan]")
    return run_id


def get_model_details_input():
    """Get model name and version input from user"""
    model_name = Prompt.ask("[bold cyan]Enter Model Name[/bold cyan]")
    version = Prompt.ask("[bold cyan]Enter Version[/bold cyan]")
    return model_name, version


# Main menu functions
def main_menu():
    """Show main menu and handle user selection"""
    while True:
        options = [
            ("1", "Experiments & Runs Management"),
            ("2", "Model Registration"),
            ("3", "Model Promotion"),
            ("4", "Model Comparison"),
            ("q", "Exit")
        ]
        
        print_menu("MLflow Model Manager", options)
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "2", "3", "4", "q"], show_choices=False)
        
        if choice == "1":
            experiments_menu()
        elif choice == "2":
            registration_menu()
        elif choice == "3":
            promotion_menu()
        elif choice == "4":
            comparison_menu()
        elif choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)


def experiments_menu():
    """Experiments and runs menu"""
    while True:
        options = [
            ("1", "List Experiments"),
            ("2", "List Runs"),
            ("3", "Show Run Details"),
            ("b", "Back to Main Menu"),
            ("q", "Exit")
        ]
        
        print_menu("Experiments & Runs Management", options)
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "2", "3", "b", "q"], show_choices=False)
        
        if choice == "1":
            list_experiments_interactive()
        elif choice == "2":
            list_runs_interactive()
        elif choice == "3":
            show_run_interactive()
        elif choice == "b":
            return
        elif choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)


def registration_menu():
    """Model registration menu"""
    while True:
        options = [
            ("1", "List Registered Models"),
            ("2", "Show Registered Model Details"),
            ("3", "Register Model from Run"),
            ("b", "Back to Main Menu"),
            ("q", "Exit")
        ]
        
        print_menu("Model Registration", options)
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "2", "3", "b", "q"], show_choices=False)
        
        if choice == "1":
            list_models_interactive()
        elif choice == "2":
            show_model_interactive()
        elif choice == "3":
            register_model_interactive()
        elif choice == "b":
            return
        elif choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)


def promotion_menu():
    """Model promotion menu"""
    while True:
        options = [
            ("1", "Promote Model"),
            ("b", "Back to Main Menu"),
            ("q", "Exit")
        ]
        
        print_menu("Model Promotion", options)
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "b", "q"], show_choices=False)
        
        if choice == "1":
            promote_model_interactive()
        elif choice == "b":
            return
        elif choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)


def comparison_menu():
    """Model comparison menu"""
    while True:
        options = [
            ("1", "Compare Model Metrics"),
            ("b", "Back to Main Menu"),
            ("q", "Exit")
        ]
        
        print_menu("Model Comparison", options)
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "b", "q"], show_choices=False)
        
        if choice == "1":
            compare_models_interactive()
        elif choice == "b":
            return
        elif choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)


# Interactive implementations of main functions
def list_experiments_interactive():
    """Interactive version of list experiments"""
    print_header("List MLflow Experiments")
    
    try:
        args = Args()
        list_experiments(args)
        wait_for_key()
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        wait_for_key()


def list_runs_interactive():
    """Interactive version of list runs"""
    print_header("List MLflow Runs")
    
    experiment_name = Prompt.ask("[bold cyan]Enter experiment name[/bold cyan]", default=DEFAULT_EXPERIMENT_NAME)
    timeframe = Prompt.ask("[bold cyan]Enter timeframe filter (leave empty for all)[/bold cyan]", default="")
    model_type = Prompt.ask("[bold cyan]Enter model type filter (leave empty for all)[/bold cyan]", default="")
    version_type = Prompt.ask("[bold cyan]Enter version type filter (leave empty for all)[/bold cyan]", default="")
    enhancement_type = Prompt.ask("[bold cyan]Enter enhancement type filter (leave empty for all)[/bold cyan]", default="")
    limit = IntPrompt.ask("[bold cyan]Maximum number of runs to display[/bold cyan]", default=25)
    
    try:
        args = Args(
            experiment_name=experiment_name,
            timeframe=timeframe if timeframe else None,
            model_type=model_type if model_type else None,
            version_type=version_type if version_type else None,
            enhancement_type=enhancement_type if enhancement_type else None,
            limit=limit
        )
        
        list_runs(args)
        wait_for_key()
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        wait_for_key()


def show_run_interactive():
    """Interactive version of show run details"""
    print_header("Show Run Details")
    
    run_id = get_run_id_input()
    show_artifacts = Confirm.ask("[bold cyan]Show artifacts?[/bold cyan]", default=False)
    
    try:
        args = Args(
            run_id=run_id,
            show_artifacts=show_artifacts
        )
        
        show_run_details(args)
        
        # Additional options after viewing run
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("  [bold cyan]1[/bold cyan]: Register this model")
        console.print("  [bold cyan]2[/bold cyan]: Back")
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "2"], default="2")
        
        if choice == "1":
            register_model_from_run(run_id)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    
    wait_for_key()


def list_models_interactive():
    """Interactive version of list registered models"""
    print_header("List Registered Models")
    
    try:
        args = Args()
        list_registered_models(args)
        wait_for_key()
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        wait_for_key()


def show_model_interactive():
    """Interactive version of show registered model details"""
    print_header("Show Registered Model Details")
    
    model_name = Prompt.ask("[bold cyan]Enter model name[/bold cyan]")
    
    try:
        args = Args(model_name=model_name)
        show_registered_model(args)
        
        # Additional options after viewing model
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("  [bold cyan]1[/bold cyan]: Promote a version of this model")
        console.print("  [bold cyan]2[/bold cyan]: Back")
        
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "2"], default="2")
        
        if choice == "1":
            version = Prompt.ask("[bold cyan]Enter version to promote[/bold cyan]")
            promote_model_with_details(model_name, version)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    
    wait_for_key()


def register_model_from_run(run_id: str = None):
    """Register a model from a specific run"""
    if run_id is None:
        run_id = get_run_id_input()
    
    model_name = Prompt.ask("[bold cyan]Enter model name[/bold cyan]")
    
    # MLflow stages
    mlflow_stages = ["None", "Staging", "Production", "Archived"]
    console.print("[bold cyan]MLflow stages:[/bold cyan]")
    for i, stage in enumerate(mlflow_stages):
        console.print(f"  {i+1}. {stage}")
    
    stage_idx = IntPrompt.ask("[bold cyan]Select MLflow stage[/bold cyan]", default=2, choices=[str(i+1) for i in range(len(mlflow_stages))])
    stage = mlflow_stages[stage_idx-1]
    
    # Promotion stages
    console.print("[bold cyan]Promotion stages:[/bold cyan]")
    for i, stage_name in enumerate(VALID_STAGES):
        color = STAGE_COLORS.get(stage_name, "white")
        console.print(f"  {i+1}. [{color}]{stage_name}[/{color}]")
    
    promotion_idx = IntPrompt.ask("[bold cyan]Select promotion stage[/bold cyan]", default=1, choices=[str(i+1) for i in range(len(VALID_STAGES))])
    promotion_stage = VALID_STAGES[promotion_idx-1]
    
    try:
        args = Args(
            run_id=run_id,
            name=model_name,
            stage=stage,
            promotion_stage=promotion_stage
        )
        
        register_model(args)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    
    wait_for_key()


def register_model_interactive():
    """Interactive version of register model"""
    print_header("Register Model from Run")
    register_model_from_run()


def promote_model_with_details(model_name: str = None, version: str = None):
    """Promote a model with specified details"""
    if model_name is None or version is None:
        model_name, version = get_model_details_input()
    
    # Promotion stages
    console.print("[bold cyan]Target promotion stages:[/bold cyan]")
    for i, stage_name in enumerate(VALID_STAGES):
        color = STAGE_COLORS.get(stage_name, "white")
        console.print(f"  {i+1}. [{color}]{stage_name}[/{color}]")
    
    stage_idx = IntPrompt.ask("[bold cyan]Select target stage[/bold cyan]", default=2, choices=[str(i+1) for i in range(len(VALID_STAGES))])
    stage = VALID_STAGES[stage_idx-1]
    
    reason = Prompt.ask("[bold cyan]Enter reason for promotion (optional)[/bold cyan]", default="")
    
    try:
        args = Args(
            name=model_name,
            version=version,
            stage=stage,
            reason=reason if reason else None,
            yes=True  # Skip confirmation as we're in interactive mode
        )
        
        promote_model(args)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


def promote_model_interactive():
    """Interactive version of promote model"""
    print_header("Promote Model")
    promote_model_with_details()
    wait_for_key()


def compare_models_interactive():
    """Interactive version of compare models"""
    print_header("Compare Models")
    
    run_id_1 = Prompt.ask("[bold cyan]Enter first run ID[/bold cyan]")
    run_id_2 = Prompt.ask("[bold cyan]Enter second run ID[/bold cyan]")
    
    metrics_filter = Prompt.ask("[bold cyan]Enter metrics filter (comma separated, leave empty for all)[/bold cyan]", default="")
    visualize = Confirm.ask("[bold cyan]Generate visual comparison?[/bold cyan]", default=True)
    
    try:
        args = Args(
            run_id_1=run_id_1,
            run_id_2=run_id_2,
            metrics=metrics_filter,
            visualize=visualize
        )
        
        compare_models(args)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    
    wait_for_key()


def welcome_screen():
    """Display welcome screen with animation"""
    clear_screen()
    
    welcome_text = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                [bold green]MLflow Model Manager[/bold green]                        ║
║                                                              ║
║         [cyan]Interactive Interface for Stock AI Predictor[/cyan]        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

            Loading system components...
"""
    
    console.print(welcome_text)
    
    # Simulate loading
    with console.status("[bold green]Initializing MLflow connection...", spinner="dots"):
        time.sleep(1)
    
    clear_screen()


if __name__ == "__main__":
    try:
        welcome_screen()
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(0)
