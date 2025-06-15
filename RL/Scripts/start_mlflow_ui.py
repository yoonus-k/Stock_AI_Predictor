#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Start the MLflow UI to view experiments and registered models

This script launches the MLflow UI server to view experiment tracking data and
the model registry.

Usage:
    python -m RL.Scripts.start_mlflow_ui [--port PORT]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    parser = argparse.ArgumentParser(description="Start the MLflow UI")
    parser.add_argument("--port", type=int, default=5000, help="Port for the MLflow UI (default: 5000)")
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Change to the project root directory
    os.chdir(project_root)
    
    print(f"Starting MLflow UI on port {args.port}")
    print("Press Ctrl+C to stop the server")
    print("\nAccess the UI at:")
    print(f"http://localhost:{args.port}")
    print("\nTo view registered models, click on 'Models' in the top navigation bar.")
    
    # Start MLflow UI with the specified port
    try:
        subprocess.run(["mlflow", "ui", "--port", str(args.port)])
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped")

if __name__ == "__main__":
    main()
