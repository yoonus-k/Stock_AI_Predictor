# Helper functions for Google Colab environment
import os
import sys
import importlib
import shutil
from typing import Optional, Dict, Tuple, List

def setup_colab_environment():
    """
    Set up the Colab environment for Stock AI Predictor parameter testing
    Returns the correct database path and ensures necessary directories exist
    """
    # Base paths
    project_root = '/content/Stock_AI_Predictor'
    drive_root = '/content/drive/MyDrive/Stock_AI_Predictor'
    
    # Check if we're using Drive or local storage
    using_drive = os.path.exists('/content/drive/MyDrive')
    
    if using_drive:
        print("Using Google Drive for storage")
        # Make sure Drive directories exist
        os.makedirs(f"{drive_root}/Data/Storage", exist_ok=True)
        os.makedirs(f"{drive_root}/Images/ParamTesting", exist_ok=True)
        os.makedirs(f"{drive_root}/docs", exist_ok=True)
        
        # Create the local directories too
        os.makedirs(f"{project_root}/Data/Storage", exist_ok=True)
        
        # Check if DB exists in Drive
        drive_db_path = f"{drive_root}/Data/Storage/data.db"
        local_db_path = f"{project_root}/Data/Storage/data.db"
        
        if os.path.exists(drive_db_path):
            print(f"Found database in Drive: {drive_db_path}")
            # Copy to local if needed
            if not os.path.exists(local_db_path):
                print(f"Copying database to local storage...")
                shutil.copy(drive_db_path, local_db_path)
            return local_db_path
        else:
            print(f"No database found in Drive. Will use local path: {local_db_path}")
            return local_db_path
    else:
        print("Google Drive not mounted. Using local storage only.")
        # Ensure local directory exists
        os.makedirs(f"{project_root}/Data/Storage", exist_ok=True)
        return f"{project_root}/Data/Storage/data.db"

def check_critical_imports():
    """
    Check all critical imports needed for parameter testing
    Returns True if all imports succeed, False otherwise
    """
    critical_modules = [
        "mplfinance", 
        "pyclustering", 
        "matplotlib", 
        "numpy", 
        "pandas", 
        "sklearn",
        "tqdm",
        "sqlitecloud"
    ]
    
    missing_modules = []
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✓ {module:15} available")
        except ImportError:
            print(f"✗ {module:15} missing")
            missing_modules.append(module)
    
    # Install missing modules
    for module in missing_modules:
        print(f"Installing {module}...")
        os.system(f"pip install {module} -q")
    
    return len(missing_modules) == 0

def check_pattern_miner_imports():
    """
    Check imports specifically for Pattern_Miner
    """
    pip_pattern_file = '/content/Stock_AI_Predictor/Pattern/pip_pattern_miner.py'
    
    if not os.path.exists(pip_pattern_file):
        print("Could not find pip_pattern_miner.py at expected location!")
        os.system("find /content/Stock_AI_Predictor -name 'pip_pattern_miner.py'")
        return False
        
    print("Checking pip_pattern_miner.py imports...")
    os.system(f"head -10 {pip_pattern_file}")
    
    # Try importing modules used by pip_pattern_miner
    modules_to_check = [
        "mplfinance", 
        "pyclustering.cluster.silhouette", 
        "pyclustering.cluster.kmeans",
        "pyclustering.cluster.center_initializer",
        "sklearn.preprocessing"
    ]
    
    success = True
    for module in modules_to_check:
        try:
            if "." in module:
                parent = module.split(".")[0]
                importlib.import_module(parent)
            else:
                importlib.import_module(module)
            print(f"✓ Successfully imported {module}")
        except ImportError as e:
            print(f"✗ Failed to import {module}: {e}")
            # Try to install the module
            simple_name = module.split(".")[0] 
            print(f"  Installing {simple_name}...")
            os.system(f"pip install {simple_name} -q")
            success = False
            
    return success

def setup_python_path():
    """Ensure the project root is in the Python path"""
    project_root = '/content/Stock_AI_Predictor'
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    
    # Print the Python path to confirm
    print("Python path includes:")
    for p in sys.path:
        print(f"  {p}")
    
    return project_root in sys.path
