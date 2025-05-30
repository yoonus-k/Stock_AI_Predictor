"""
CLI tool to run all monitoring tools for the RL trading agent

This script provides a command-line interface for running all monitoring tools
from a single entry point. It allows users to quickly access and run various
analysis and visualization tools.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


def print_header(text):
    """Print a styled header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)


def run_tool(script_name, args):
    """Run a monitoring tool with the given arguments"""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False
    
    try:
        cmd = [sys.executable, str(script_path)] + args
        print(f"Running: {' '.join(cmd)}")
        
        subprocess.run(
            cmd,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Monitor and analyze RL trading agent performance"
    )
    
    # Main arguments
    parser.add_argument(
        "--model", 
        type=str, 
        help="Path to the trained model file (.zip)",
        default=None
    )
    
    parser.add_argument(
        "--log-dir", 
        type=str, 
        help="Path to the log directory",
        default=None
    )
    
    # Tool selection
    parser.add_argument(
        "--tool", 
        type=str,
        choices=[
            "dashboard", 
            "tensorboard", 
            "checkpoints", 
            "features", 
            "strategy", 
            "decision", 
            "live",
            "all"
        ],
        default="dashboard",
        help="Specific tool to run (default: dashboard)"
    )
    
    # Common parameters
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006,
        help="Port for TensorBoard"
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=3,
        help="Number of episodes for evaluation"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save analysis results"
    )
    
    return parser.parse_args()


def find_model():
    """Find the most recent model file"""
    models_dir = Path(__file__).parent.parent / "Models"
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return None
    
    model_files = list(models_dir.glob("*.zip"))
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return None
    
    # Get the most recent model
    latest_model = sorted(model_files, key=os.path.getmtime)[-1]
    print(f"Found model: {latest_model}")
    
    return latest_model


def find_log_dir():
    """Find the logs directory"""
    log_dir = Path(__file__).parent.parent / "Logs"
    
    if not log_dir.exists():
        print(f"Logs directory not found: {log_dir}")
        return None
    
    return log_dir


def find_checkpoint_dir():
    """Find the checkpoint directory"""
    checkpoint_dir = Path(__file__).parent.parent / "Logs" / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    return checkpoint_dir


def setup_output_dir():
    """Setup output directory for analysis results"""
    analysis_dir = Path(__file__).parent.parent / "Analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    return analysis_dir


def main():
    """Main function"""
    args = parse_args()
    
    # Find model if not specified
    if not args.model:
        args.model = find_model()
    
    # Find log directory if not specified
    if not args.log_dir:
        args.log_dir = find_log_dir()
    
    # Setup output directory if not specified
    if not args.output_dir:
        args.output_dir = setup_output_dir()
    
    # Check tool selection and run appropriate tool
    if args.tool == "dashboard" or args.tool == "all":
        print_header("Running Monitoring Dashboard")
        run_tool("monitoring_dashboard.py", [
            "--model", str(args.model) if args.model else "",
            "--log-dir", str(args.log_dir) if args.log_dir else ""
        ])
    
    if args.tool == "tensorboard" or args.tool == "all":
        print_header("Running TensorBoard")
        run_tool("view_tensorboard.py", [
            "--log-dir", str(args.log_dir / "tensorboard") if args.log_dir else "",
            "--port", str(args.port)
        ])
    
    if args.tool == "checkpoints" or args.tool == "all":
        print_header("Running Checkpoint Analysis")
        checkpoint_dir = find_checkpoint_dir()
        if checkpoint_dir:
            run_tool("analyze_checkpoints.py", [
                "--checkpoint-dir", str(checkpoint_dir),
                "--episodes", str(args.episodes)
            ])
    
    if args.tool == "features" or args.tool == "all":
        print_header("Running Feature Importance Analysis")
        if args.model:
            feature_dir = Path(args.output_dir) / "feature_importance" if args.output_dir else None
            run_tool("monitor_feature_importance.py", [
                "--model", str(args.model),
                "--output-dir", str(feature_dir) if feature_dir else "",
                "--episodes", str(args.episodes)
            ])
    
    if args.tool == "strategy" or args.tool == "all":
        print_header("Running Trading Strategy Analysis")
        if args.model:
            strategy_dir = Path(args.output_dir) / "strategy" if args.output_dir else None
            run_tool("monitor_trading_strategy.py", [
                "--model", str(args.model),
                "--output-dir", str(strategy_dir) if strategy_dir else "",
                "--episodes", str(args.episodes)
            ])
    
    if args.tool == "decision" or args.tool == "all":
        print_header("Running Decision Boundary Analysis")
        if args.model:
            decision_dir = Path(args.output_dir) / "decision_boundaries" if args.output_dir else None
            run_tool("monitor_decision_boundaries.py", [
                "--model", str(args.model),
                "--output-dir", str(decision_dir) if decision_dir else ""
            ])
    
    if args.tool == "live" or args.tool == "all":
        print_header("Running Live Training Dashboard")
        run_tool("live_training_dashboard.py", [
            "--log-dir", str(args.log_dir) if args.log_dir else "",
            "--refresh", "5"
        ])


if __name__ == "__main__":
    main()
