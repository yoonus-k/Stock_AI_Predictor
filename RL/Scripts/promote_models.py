"""
Model selection and promotion utility for the RL trading pipeline
Automates the process of identifying best models and promoting them to production
"""

import os
import sys
import mlflow
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Custom imports
from RL.Utils.mlflow_manager import MLflowManager


class ModelSelector:
    """
    Utility for selecting the best models from MLflow experiments
    and promoting them to production status
    """
    
    def __init__(
        self,
        experiment_name: str = "stock_trading_rl",
        tracking_uri: Optional[str] = None,
        production_dir: Optional[str] = None
    ):
        """
        Initialize the model selector
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
            production_dir: Directory for production models
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri is None:
            # Use local MLflow tracking in the project directory
            tracking_uri = f"file://{project_root}/mlruns"
        
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")
        
        # Set production directory
        if production_dir is None:
            self.production_dir = project_root / "RL" / "Models" / "Production"
        else:
            self.production_dir = Path(production_dir)
        
        os.makedirs(self.production_dir, exist_ok=True)
        
        # Get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            
            self.experiment_id = self.experiment.experiment_id
            print(f"Found experiment: {experiment_name} (ID: {self.experiment_id})")
        except Exception as e:
            print(f"Error getting experiment: {e}")
            self.experiment_id = None
    
    def find_best_model_run(
        self,
        timeframe: str,
        metric: str = "final_eval_mean_reward",
        min_timesteps: int = 50000,
        status: str = "FINISHED"
    ) -> Optional[mlflow.entities.Run]:
        """
        Find the best model run for a given timeframe
        
        Args:
            timeframe: Model timeframe (daily, weekly, monthly, meta)
            metric: Metric to optimize
            min_timesteps: Minimum training timesteps
            status: Run status to filter by
            
        Returns:
            Best run or None if not found
        """
        if self.experiment_id is None:
            return None
        
        # Build filter string
        filter_str = f"""
            experiment_id = '{self.experiment_id}' 
            and status = '{status}'
            and tags.timeframe = '{timeframe}'
            and tags.model_type != 'baseline'
        """
        
        # Query runs
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_str
        )
        
        if runs.empty:
            print(f"No runs found for timeframe: {timeframe}")
            return None
        
        # Filter runs with the specified metric and minimum timesteps
        metric_col = f"metrics.{metric}"
        timesteps_col = "metrics.timesteps"
        
        valid_runs = runs[runs[metric_col].notnull()]
        
        # Apply timesteps filter if column exists
        if timesteps_col in valid_runs.columns:
            valid_runs = valid_runs[valid_runs[timesteps_col] >= min_timesteps]
        
        if valid_runs.empty:
            print(f"No valid runs found for timeframe: {timeframe}")
            return None
        
        # Find best run
        best_run_idx = valid_runs[metric_col].idxmax()
        best_run_id = valid_runs.loc[best_run_idx, "run_id"]
        
        # Get full run info
        best_run = mlflow.get_run(best_run_id)
        
        print(f"Best {timeframe} model:")
        print(f"  - Run ID: {best_run.info.run_id}")
        print(f"  - {metric}: {best_run.data.metrics.get(metric, 'N/A')}")
        
        return best_run
    
    def promote_model_to_production(
        self,
        timeframe: str,
        run_id: Optional[str] = None,
        metric: str = "final_eval_mean_reward",
        min_timesteps: int = 50000,
        copy_artifacts: bool = True
    ) -> Optional[str]:
        """
        Promote a model to production status
        
        Args:
            timeframe: Model timeframe
            run_id: Specific run ID (if None, will find best run)
            metric: Metric to optimize when finding best run
            min_timesteps: Minimum training timesteps
            copy_artifacts: Whether to copy model artifacts to production dir
            
        Returns:
            Path to production model or None if failed
        """
        # Get run to promote
        if run_id is not None:
            run = mlflow.get_run(run_id)
        else:
            run = self.find_best_model_run(
                timeframe=timeframe,
                metric=metric,
                min_timesteps=min_timesteps
            )
        
        if run is None:
            print(f"No suitable model found for {timeframe} timeframe")
            return None
        
        # Create production model directory
        timeframe_dir = self.production_dir / timeframe
        os.makedirs(timeframe_dir, exist_ok=True)
        
        # Prepare model details
        model_details = {
            "run_id": run.info.run_id,
            "timeframe": timeframe,
            "promoted_at": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "metrics": {}
        }
        
        # Add metrics to details
        for key, value in run.data.metrics.items():
            model_details["metrics"][key] = float(value)
        
        # Add selected parameters and tags
        model_details["parameters"] = {}
        for key, value in run.data.params.items():
            model_details["parameters"][key] = value
        
        model_details["tags"] = {}
        for key, value in run.data.tags.items():
            if key.startswith("mlflow.") or key == "git":
                continue
            model_details["tags"][key] = value
        
        # Save details
        details_path = timeframe_dir / "model_details.json"
        with open(details_path, "w") as f:
            json.dump(model_details, f, indent=2)
        
        # Copy model artifacts if requested
        if copy_artifacts:
            model_path = self._copy_model_artifacts(run, timeframe_dir)
            if model_path:
                print(f"✅ Model promoted to production: {model_path}")
                return model_path
        
        # Register model in MLflow
        try:
            # Get model URI
            artifacts = [f for f in mlflow.artifacts.list_artifacts(run.info.run_id)]
            model_artifacts = [a for a in artifacts if a.endswith(".zip") or "meta_model_final" in a]
            
            if model_artifacts:
                # Use the first model artifact found
                model_name = f"stock-predictor-{timeframe}"
                model_uri = f"runs:/{run.info.run_id}/{model_artifacts[0]}"
                
                # Register model
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
                
                # Set production stage
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model.version,
                    stage="Production"
                )
                
                print(f"✅ Model registered in MLflow as {model_name} (v{registered_model.version}) in 'Production' stage")
            else:
                print(f"⚠️ No model artifacts found in run {run.info.run_id}")
        except Exception as e:
            print(f"⚠️ Error registering model in MLflow: {e}")
        
        return str(timeframe_dir)
    
    def _copy_model_artifacts(self, run: mlflow.entities.Run, dest_dir: Path) -> Optional[str]:
        """
        Copy model artifacts to production directory
        
        Args:
            run: MLflow run
            dest_dir: Destination directory
            
        Returns:
            Path to copied model or None if failed
        """
        try:
            # Get artifacts
            artifacts = mlflow.artifacts.list_artifacts(run.info.run_id)
            
            # Find model artifacts
            model_artifacts = [
                a for a in artifacts 
                if a.endswith(".zip") or "meta_model_final" in a
            ]
            
            if not model_artifacts:
                print(f"No model artifacts found in run {run.info.run_id}")
                return None
            
            # Download and copy artifacts
            for artifact_path in model_artifacts:
                # Target path
                artifact_name = os.path.basename(artifact_path)
                target_path = dest_dir / artifact_name
                
                # Download artifact
                mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id,
                    artifact_path=artifact_path,
                    dst_path=str(dest_dir)
                )
                
                if os.path.exists(target_path):
                    print(f"Copied artifact: {artifact_path} -> {target_path}")
                    return str(target_path)
            
            return None
        
        except Exception as e:
            print(f"Error copying model artifacts: {e}")
            return None
    
    def promote_all_timeframe_models(
        self,
        timeframes: List[str] = None,
        metric: str = "final_eval_mean_reward",
        min_timesteps: int = 50000
    ) -> Dict[str, str]:
        """
        Promote best models for all specified timeframes to production
        
        Args:
            timeframes: List of timeframes to promote
            metric: Metric to optimize
            min_timesteps: Minimum training timesteps
        
        Returns:
            Dictionary of production model paths by timeframe
        """
        if timeframes is None:
            timeframes = ["daily", "weekly", "monthly", "meta"]
        
        production_models = {}
        
        for timeframe in timeframes:
            print(f"\n===== PROMOTING {timeframe.upper()} MODEL =====")
            model_path = self.promote_model_to_production(
                timeframe=timeframe,
                metric=metric,
                min_timesteps=min_timesteps
            )
            
            if model_path:
                production_models[timeframe] = model_path
        
        print("\n===== MODEL PROMOTION SUMMARY =====")
        for timeframe, path in production_models.items():
            print(f"✅ {timeframe}: {path}")
        
        # Create symlinks to latest models for easier access
        self._create_latest_symlinks(production_models)
        
        return production_models
    
    def _create_latest_symlinks(self, model_paths: Dict[str, str]):
        """Create symlinks to the latest models"""
        latest_dir = self.production_dir / "latest"
        os.makedirs(latest_dir, exist_ok=True)
        
        # Create a symlink for each timeframe
        for timeframe, path in model_paths.items():
            if os.path.exists(path):
                # Find model file (zip)
                model_files = list(Path(path).glob("*.zip"))
                if model_files:
                    model_file = model_files[0]
                    link_path = latest_dir / f"{timeframe}_model.zip"
                    
                    # Remove existing symlink if it exists
                    if os.path.exists(link_path) or os.path.islink(link_path):
                        os.remove(link_path)
                    
                    # Create relative symlink
                    try:
                        os.symlink(
                            os.path.relpath(model_file, latest_dir),
                            link_path
                        )
                        print(f"Created symlink: {link_path} -> {model_file}")
                    except Exception as e:
                        # Fall back to file copy if symlink fails (e.g. on Windows)
                        print(f"Symlink failed, copying file instead: {e}")
                        shutil.copy2(model_file, link_path)
                        print(f"Copied file: {model_file} -> {link_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote RL models to production")
    
    parser.add_argument("--experiment", type=str, default="stock_trading_rl",
                      help="MLflow experiment name")
    parser.add_argument("--timeframes", type=str, default="daily,weekly,monthly,meta",
                      help="Comma-separated list of timeframes to promote")
    parser.add_argument("--metric", type=str, default="final_eval_mean_reward",
                      help="Metric to optimize when selecting models")
    parser.add_argument("--production-dir", type=str, default=None,
                      help="Production models directory")
    
    args = parser.parse_args()
    
    # Parse timeframes
    timeframes = args.timeframes.split(",")
    
    # Create model selector
    selector = ModelSelector(
        experiment_name=args.experiment,
        production_dir=args.production_dir
    )
    
    # Promote models
    selector.promote_all_timeframe_models(
        timeframes=timeframes,
        metric=args.metric
    )
