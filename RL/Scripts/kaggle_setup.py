"""
Kaggle Environment Setup Script

This script prepares the Kaggle environment by:
1. Installing required dependencies
2. Downloading the project code from GitHub
3. Setting up data access
4. Configuring the environment for training
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def run_command(cmd, verbose=True):
    """Run a shell command and print output"""
    if verbose:
        print(f"Running: {cmd}")
    
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    stdout, stderr = process.communicate()
    
    if verbose:
        print(stdout)
        if stderr:
            print(f"Error: {stderr}")
            
    return process.returncode == 0

def install_dependencies():
    """Install required packages"""
    print("\n=== Installing Dependencies ===\n")
    
    # List of required packages
    requirements = [
        "stable-baselines3==2.0.0",
        "gymnasium==0.28.1",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "numpy==1.25.2",
        "tqdm==4.66.1",
        "scikit-learn==1.3.0", 
        "shap==0.42.1"
    ]
    
    # Check which packages need to be installed
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    to_install = []
    
    for req in requirements:
        pkg_name = req.split('==')[0]
        if pkg_name not in installed:
            to_install.append(req)
            print(f"Need to install: {req}")
        else:
            print(f"Already installed: {pkg_name} (version {installed[pkg_name]})")
    
    # Install missing packages
    if to_install:
        print(f"Installing {len(to_install)} packages...")
        install_cmd = f"pip install {' '.join(to_install)}"
        success = run_command(install_cmd)
        if not success:
            print("Failed to install packages!")
            return False
        print("Packages installed successfully!")
    else:
        print("All packages already installed.")
    
    return True

def clone_repository():
    """Clone the GitHub repository"""
    print("\n=== Cloning Repository ===\n")
    
    repo_url = "https://github.com/yoonus-k/Stock_AI_Predictor.git"
    repo_dir = Path("/kaggle/working/Stock_AI_Predictor")
    
    if repo_dir.exists():
        print(f"Repository directory {repo_dir} already exists!")
        # Pull latest changes
        cmd = f"cd {repo_dir} && git pull"
        run_command(cmd)
        return True
    
    # Clone the repository
    cmd = f"git clone {repo_url} {repo_dir}"
    success = run_command(cmd)
    
    if not success:
        print("Failed to clone repository!")
        return False
    
    print(f"Repository cloned to {repo_dir}")
    return True

def setup_data_paths():
    """Configure data paths for Kaggle"""
    print("\n=== Setting Up Data Paths ===\n")
    
    # We need to know where the data is stored
    # Check if input data exists under /kaggle/input
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        print(f"Kaggle input directory {input_dir} not found!")
        print("This script should be run in a Kaggle notebook environment.")
        return False
    
    # List available datasets
    print("Available datasets in Kaggle input directory:")
    datasets = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset.name}")
    
    # Create symbolic links if needed
    # Example: link_cmd = f"ln -s /kaggle/input/trading-data/samples.db /kaggle/working/samples.db"
    # run_command(link_cmd)
    
    return True

def configure_environment():
    """Set up environment variables and paths"""
    print("\n=== Configuring Environment ===\n")
    
    # Add project directories to Python path
    project_dir = Path("/kaggle/working/Stock_AI_Predictor")
    if not project_dir.exists():
        print(f"Project directory {project_dir} not found!")
        return False
    
    sys.path.append(str(project_dir))
    print(f"Added {project_dir} to Python path")
    
    # Create output directory
    output_dir = Path("/kaggle/working/output")
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    return True

def main():
    """Main setup function"""
    print("\n=== Kaggle Environment Setup for RL Trading Model Training ===\n")
    
    # Check if running on Kaggle
    if not os.path.exists("/kaggle/input"):
        print("This script is designed to be run in a Kaggle notebook environment.")
        print("You appear to be running it in a different environment.")
        return
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Cloning repository", clone_repository),
        ("Setting up data paths", setup_data_paths),
        ("Configuring environment", configure_environment)
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"Failed at step: {step_name}")
            success = False
            break
        print(f"✅ {step_name} completed successfully!")
    
    if success:
        print("\n=== Setup Complete! ===")
        print("\nTo train the model, run:")
        print("```python")
        print("from RL.Scripts.kaggle_training import train_model")
        print("train_model(")
        print("    db_path='/kaggle/input/YOUR-DATASET-PATH/samples.db',")
        print("    output_dir='/kaggle/working/output',")
        print("    timesteps=500000")
        print(")")
        print("```")
    else:
        print("\n❌ Setup failed! Please check the errors above.")

if __name__ == "__main__":
    main()
