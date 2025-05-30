#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hugging Face Deployment Script

This script automates the deployment of models and datasets to Hugging Face Hub.
It handles:
1. Parameter tester model upload
2. RL model upload
3. Dataset upload
4. Spaces API deployment
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom modules
from Deployment.parameter_tester_model import ParameterTesterModel
from Deployment.trading_rl_model import TradingRLModel
from Deployment.dataset_uploader import prepare_dataset, upload_dataset

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hf_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hf_deployment")

# Hugging Face configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
PARAM_TESTER_REPO_ID = os.environ.get("PARAM_TESTER_REPO_ID")
RL_MODEL_REPO_ID = os.environ.get("RL_MODEL_REPO_ID")
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID")
API_SPACE_ID = os.environ.get("API_SPACE_ID")

class HuggingFaceDeployer:
    """Class for deploying models and datasets to Hugging Face Hub"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the deployer
        
        Args:
            token: Hugging Face token
        """
        self.token = token or HF_TOKEN
        if not self.token:
            raise ValueError("Hugging Face token is required")
        
        self.api = HfApi(token=self.token)
        
    def deploy_parameter_tester(self, repo_id: str = None, model_path: Optional[str] = None) -> None:
        """
        Deploy parameter tester model to Hugging Face Hub
        
        Args:
            repo_id: Repository ID
            model_path: Path to model directory
        """
        repo_id = repo_id or PARAM_TESTER_REPO_ID
        if not repo_id:
            raise ValueError("Parameter tester repository ID is required")
            
        logger.info(f"Deploying parameter tester model to {repo_id}")
        
        # Create repository if it doesn't exist
        try:
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise
            
        # Prepare model for upload
        if not model_path:
            # Create temp directory for model files
            import tempfile
            temp_dir = tempfile.mkdtemp()
            model_path = temp_dir
            
            # Create a model instance and save it
            model = ParameterTesterModel()
            # You would load your actual model data here
            model.save_pretrained(model_path)
        
        # Upload model files
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update parameter tester model",
            ignore_patterns=["*.log", "*.pyc", "__pycache__/*"]
        )
        
        logger.info(f"Parameter tester model deployed to {repo_id}")
        
    def deploy_rl_model(self, repo_id: str = None, model_path: Optional[str] = None) -> None:
        """
        Deploy RL trading model to Hugging Face Hub
        
        Args:
            repo_id: Repository ID
            model_path: Path to model directory
        """
        repo_id = repo_id or RL_MODEL_REPO_ID
        if not repo_id:
            raise ValueError("RL model repository ID is required")
            
        logger.info(f"Deploying RL trading model to {repo_id}")
        
        # Create repository if it doesn't exist
        try:
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise
            
        # Prepare model for upload
        if not model_path:
            # Create temp directory for model files
            import tempfile
            temp_dir = tempfile.mkdtemp()
            model_path = temp_dir
            
            # Create a model instance and save it
            model = TradingRLModel()
            # You would load your actual model data here
            model.save_pretrained(model_path)
        
        # Upload model files
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update RL trading model",
            ignore_patterns=["*.log", "*.pyc", "__pycache__/*"]
        )
        
        logger.info(f"RL trading model deployed to {repo_id}")
        
    def deploy_dataset(self, repo_id: str = None) -> None:
        """
        Deploy dataset to Hugging Face Hub
        
        Args:
            repo_id: Repository ID
        """
        repo_id = repo_id or DATASET_REPO_ID
        if not repo_id:
            raise ValueError("Dataset repository ID is required")
            
        logger.info(f"Deploying dataset to {repo_id}")
        
        # Prepare and upload dataset
        try:
            # This function should be implemented in dataset_uploader.py
            prepare_dataset()
            upload_dataset(repo_id=repo_id, token=self.token)
        except Exception as e:
            logger.error(f"Error deploying dataset: {e}")
            raise
        
        logger.info(f"Dataset deployed to {repo_id}")
        
    def deploy_api_space(self, space_id: str = None) -> None:
        """
        Deploy API Space to Hugging Face Hub
        
        Args:
            space_id: Space ID
        """
        space_id = space_id or API_SPACE_ID
        if not space_id:
            raise ValueError("API Space ID is required")
            
        logger.info(f"Deploying API Space to {space_id}")
        
        # Create Space if it doesn't exist
        try:
            self.api.create_repo(
                space_id, 
                repo_type="space",
                space_sdk="docker",
                exist_ok=True
            )
        except Exception as e:
            logger.error(f"Error creating Space: {e}")
            raise
            
        # Upload Space files
        deployment_dir = Path(__file__).resolve().parent
        files_to_upload = [
            deployment_dir / "app.py",
            deployment_dir / "requirements.txt",
            deployment_dir / ".env.example",
            deployment_dir / "model_card.md",
            deployment_dir / "Dockerfile",
        ]
        
        for file_path in files_to_upload:
            if file_path.exists():
                try:
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=space_id,
                        repo_type="space",
                        commit_message=f"Update {file_path.name}"
                    )
                except Exception as e:
                    logger.error(f"Error uploading {file_path.name}: {e}")
                    
        logger.info(f"API Space deployed to {space_id}")

def main():
    """Main function for deploying to Hugging Face Hub"""
    
    parser = argparse.ArgumentParser(description="Deploy models and datasets to Hugging Face Hub")
    parser.add_argument("--all", action="store_true", help="Deploy all components")
    parser.add_argument("--param-tester", action="store_true", help="Deploy parameter tester model")
    parser.add_argument("--rl-model", action="store_true", help="Deploy RL trading model")
    parser.add_argument("--dataset", action="store_true", help="Deploy dataset")
    parser.add_argument("--api-space", action="store_true", help="Deploy API Space")
    
    args = parser.parse_args()
    
    # If no arguments provided, default to deploying everything
    if not any(vars(args).values()):
        args.all = True
        
    try:
        deployer = HuggingFaceDeployer()
        
        if args.all or args.param_tester:
            deployer.deploy_parameter_tester()
            
        if args.all or args.rl_model:
            deployer.deploy_rl_model()
            
        if args.all or args.dataset:
            deployer.deploy_dataset()
            
        if args.all or args.api_space:
            deployer.deploy_api_space()
            
        logger.info("Deployment completed successfully")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
