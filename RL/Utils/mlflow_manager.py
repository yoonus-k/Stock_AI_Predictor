"""
MLflow experiment tracking manager for Stock AI Predictor
Manages experiment tracking, model versioning, and artifact storage
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

class MLflowManager:
    """
    Comprehensive MLflow experiment tracking for RL Stock Trading models
    """
    
    def __init__(
        self, 
        experiment_name: str = "stock_trading_rl", 
        tracking_uri: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_type: str = "base"
    ):
        """
        Initialize MLflow tracking
        
        Args:
            experiment_name: Base name for the MLflow experiment
            tracking_uri: URI for MLflow tracking server (None for local)
            timeframe: Model timeframe (daily, weekly, monthly, meta)
            model_type: Model type (base, enhanced, production, etc.)
        """
        self.base_experiment_name = experiment_name
        self.timeframe = timeframe
        self.model_type = model_type
        
        # Create a specific experiment name if timeframe is provided
        if timeframe:
            self.experiment_name = f"{experiment_name}_{timeframe}"
        else:
            self.experiment_name = experiment_name
        
        # Set tracking URI - use local directory if not specified
        if tracking_uri is None:
            # Use local MLflow tracking in the project directory
            project_root = Path(__file__).parent.parent.parent
            tracking_uri = f"file://{project_root}/mlruns"
        
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            print(f"Created new experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
        self.current_run = None
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = False):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run (generated if None)
            tags: Dictionary of tags to apply to the run
            nested: Whether this is a nested run
            
        Returns:
            MLflow run object
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.timeframe:
                run_name = f"{self.timeframe}_{self.model_type}_{timestamp}"
            else:
                run_name = f"{self.model_type}_{timestamp}"
        
        # Default tags
        default_tags = {
            "model_type": self.model_type,
            "timeframe": self.timeframe or "unspecified",
            "framework": "stable_baselines3",
            "algorithm": "PPO"
        }
        
        # Merge with provided tags
        all_tags = {**default_tags, **(tags or {})}
        
        # End current run if exists
        if self.current_run is not None:
            self.end_run()
        
        # Start the run
        self.current_run = mlflow.start_run(
            run_name=run_name, 
            tags=all_tags,
            nested=nested
        )
        
        print(f"Started MLflow run: {run_name} (ID: {self.current_run.info.run_id})")
        return self.current_run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters and configuration
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        for key, value in params.items():
            # Convert numpy/complex types to Python primitives
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Warning: Could not log parameter {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        for key, value in metrics.items():
            # Ensure value is numeric
            if isinstance(value, (int, float, np.integer, np.floating)):
                mlflow.log_metric(key, float(value), step=step)
    
    def log_model(self, model, model_name: str = "trading_model", signature=None):
        """
        Log model as an MLflow artifact
        
        Args:
            model: Model object to log
            model_name: Name for the model
            signature: Optional model signature
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            # Use PyTorch flavor for SB3 models (they use PyTorch internally)
            mlflow.pytorch.log_model(model, model_name, signature=signature)
            print(f"Model logged as artifact: {model_name}")
        except Exception as e:
            print(f"Error logging model with PyTorch flavor: {e}")
            try:
                # Fallback to pyfunc flavor
                mlflow.pyfunc.log_model(model, model_name)
                print(f"Model logged as artifact using pyfunc flavor: {model_name}")
            except Exception as e2:
                print(f"Error logging model with pyfunc flavor: {e2}")
    
    def log_sb3_model(self, model, model_name: str = "trading_model"):
        """
        Log a Stable-Baselines3 model as an artifact (with custom saving logic)
        
        Args:
            model: SB3 model object
            model_name: Name for the model
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            # Create temporary directory for model
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, f"{model_name}.zip")
                model.save(model_path)
                mlflow.log_artifact(model_path, f"models/{self.timeframe}")
                print(f"SB3 model logged as artifact: {model_name}")
                
        except Exception as e:
            print(f"Error logging SB3 model: {e}")
    
    def log_file_artifact(self, file_path: str, artifact_path: str = None):
        """
        Log a file as an artifact
        
        Args:
            file_path: Path to the file to log
            artifact_path: Path within artifact directory
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            mlflow.log_artifact(file_path, artifact_path)
            print(f"File logged as artifact: {file_path}")
        except Exception as e:
            print(f"Error logging file {file_path}: {e}")
    
    def log_directory_artifacts(self, directory_path: str, artifact_path: str = None):
        """
        Log all files in a directory as artifacts
        
        Args:
            directory_path: Path to the directory
            artifact_path: Path within artifact directory
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            mlflow.log_artifacts(directory_path, artifact_path)
            print(f"Directory logged as artifacts: {directory_path}")
        except Exception as e:
            print(f"Error logging directory {directory_path}: {e}")
    
    def log_dataset_info(self, dataset: pd.DataFrame, name: str):
        """
        Log dataset information and statistics
        
        Args:
            dataset: Pandas DataFrame
            name: Name for the dataset 
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        # Calculate dataset statistics
        dataset_info = {
            f"{name}_size": len(dataset),
            f"{name}_columns": len(dataset.columns),
            f"{name}_memory_mb": dataset.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        # Log basic stats as metrics
        self.log_metrics(dataset_info)
        
        # Create detailed dataset report
        report = {
            "name": name,
            "shape": list(dataset.shape),
            "columns": list(dataset.columns),
            "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
            "memory_usage_mb": float(dataset.memory_usage(deep=True).sum() / 1024 / 1024),
            "missing_values": dataset.isnull().sum().to_dict(),
            "sample_data": dataset.head(3).to_dict('records')
        }
        
        # Save report as JSON artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report, f, indent=2, default=str)
            temp_path = f.name
        
        try:
            self.log_file_artifact(temp_path, f"datasets/{name}_info.json")
        finally:
            os.unlink(temp_path)
    
    def log_figure(self, figure, figure_name: str):
        """
        Log a matplotlib or plotly figure as an artifact
        
        Args:
            figure: Matplotlib or Plotly figure
            figure_name: Name for the figure
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            # Create temporary file for figure
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Check if it's a matplotlib figure
            if hasattr(figure, 'savefig'):
                figure.savefig(temp_path, dpi=300, bbox_inches='tight')
            else:
                # Assume it's a plotly figure
                figure.write_image(temp_path)
            
            # Log the figure file as an artifact
            self.log_file_artifact(temp_path, f"figures/{figure_name}")
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error logging figure: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        if self.current_run is not None:
            mlflow.end_run()
            print(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
    
    def get_best_run(self, metric: str = "final_return", ascending: bool = False):
        """
        Get the best run based on a metric
        
        Args:
            metric: Metric to use for comparison
            ascending: Whether a lower value is better
            
        Returns:
            Dictionary with best run information
        """
        runs_df = self.compare_runs(metric=metric, ascending=ascending)
        if runs_df.empty:
            return None
            
        best_run = runs_df.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "run_name": best_run.get("tags.mlflow.runName", "unknown"),
            "metric_value": best_run.get(f"metrics.{metric}", None),
            "start_time": best_run["start_time"],
            "model_type": best_run.get("tags.model_type", "unknown")
        }
    
    def compare_runs(self, metric: str = "final_return", ascending: bool = False):
        """
        Compare runs based on a specific metric
        
        Args:
            metric: Metric to use for comparison
            ascending: Whether a lower value is better
            
        Returns:
            DataFrame with run comparisons
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {order}"]
        )
        return runs
    
    # New methods for model enhancement and versioning
    
    def fetch_model_metrics(self, run_id: str) -> Dict[str, float]:
        """
        Fetch metrics from a specific run
        
        Args:
            run_id: ID of the run to fetch metrics from
            
        Returns:
            Dictionary of metrics
        """
        try:
            run = self.mlflow_client.get_run(run_id)
            metrics = {k: v for k, v in run.data.metrics.items()}
            print(f"Successfully fetched {len(metrics)} metrics from run {run_id}")
            return metrics
        except Exception as e:
            print(f"Error fetching metrics from run {run_id}: {e}")
            return {}
    
    def calculate_improvement(self, 
                              base_metrics: Dict[str, float], 
                              enhanced_metrics: Dict[str, float]
                             ) -> Dict[str, float]:
        """
        Calculate improvement percentages between base and enhanced metrics
        
        Args:
            base_metrics: Metrics from base model
            enhanced_metrics: Metrics from enhanced model
            
        Returns:
            Dictionary of improvement metrics
        """
        improvements = {}
        
        # Core metrics to compare
        key_metrics = [
            "final/mean_reward",
            "final/mean_return", 
            "portfolio/sharpe_ratio",
            "portfolio/win_rate", 
            "portfolio/profit_factor",
            "evaluation/mean_reward"
        ]
        
        for key in key_metrics:
            if key in base_metrics and key in enhanced_metrics:
                base_val = base_metrics[key]
                enhanced_val = enhanced_metrics[key]
                
                # Avoid division by zero
                if base_val != 0:
                    improvement_pct = ((enhanced_val - base_val) / abs(base_val)) * 100
                else:
                    improvement_pct = float('inf') if enhanced_val > 0 else float('-inf') if enhanced_val < 0 else 0.0
                
                improvements[f"improvement_{key}"] = improvement_pct
        
        return improvements
    
    def generate_comparison_plots(self, 
                                 base_run_id: str, 
                                 enhanced_run_id: str, 
                                 metrics_to_compare: List[str] = None):
        """
        Generate comparison visualizations between base and enhanced models
        
        Args:
            base_run_id: ID of the base model run
            enhanced_run_id: ID of the enhanced model run
            metrics_to_compare: List of metrics to compare
            
        Returns:
            List of saved plot paths
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                "final/mean_reward",
                "final/mean_return", 
                "portfolio/sharpe_ratio",
                "portfolio/win_rate", 
                "portfolio/profit_factor"
            ]
        
        try:
            # Fetch metrics
            base_metrics = self.fetch_model_metrics(base_run_id)
            enhanced_metrics = self.fetch_model_metrics(enhanced_run_id)
            
            if not base_metrics or not enhanced_metrics:
                print("Error: Could not fetch metrics for comparison")
                return []
            
            # Calculate improvements
            improvements = self.calculate_improvement(base_metrics, enhanced_metrics)
            
            # Create comparison visualizations
            plot_paths = []
            
            # 1. Bar chart comparison of key metrics
            common_metrics = [m for m in metrics_to_compare if m in base_metrics and m in enhanced_metrics]
            
            if common_metrics:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x = np.arange(len(common_metrics))
                width = 0.35
                
                base_values = [base_metrics[m] for m in common_metrics]
                enhanced_values = [enhanced_metrics[m] for m in common_metrics]
                
                rects1 = ax.bar(x - width/2, base_values, width, label='Base Model')
                rects2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced Model')
                
                ax.set_ylabel('Value')
                ax.set_title('Model Comparison: Base vs Enhanced')
                ax.set_xticks(x)
                ax.set_xticklabels([m.split('/')[-1] for m in common_metrics], rotation=45, ha='right')
                ax.legend()
                
                # Add improvement percentages above bars
                for i, metric in enumerate(common_metrics):
                    improvement_key = f"improvement_{metric}"
                    if improvement_key in improvements:
                        plt.annotate(f"{improvements[improvement_key]:.1f}%", 
                                   xy=(x[i] + width/2, enhanced_values[i]),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom',
                                   color='green' if improvements[improvement_key] > 0 else 'red')
                
                plt.tight_layout()
                
                # Save comparison plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    comparison_path = temp_file.name
                    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                    plot_paths.append(comparison_path)
                
                # Log the plot
                self.log_file_artifact(comparison_path, f"comparison/metrics_comparison.png")
                plt.close(fig)
            
            # 2. Create enhancement summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            improvement_metrics = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
            metric_names = [m[0].replace('improvement_', '').split('/')[-1] for m in improvement_metrics]
            improvement_values = [m[1] for m in improvement_metrics]
            
            colors = ['green' if v > 0 else 'red' for v in improvement_values]
            
            ax.barh(metric_names, improvement_values, color=colors)
            ax.set_xlabel('Improvement (%)')
            ax.set_title('Model Enhancement: Improvement Percentages')
            
            # Add values on bars
            for i, v in enumerate(improvement_values):
                ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
            
            plt.tight_layout()
            
            # Save improvement plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                improvement_path = temp_file.name
                plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
                plot_paths.append(improvement_path)
                  # Log the plot
            self.log_file_artifact(improvement_path, f"comparison/improvement_summary.png")
            plt.close(fig)
            
            # 3. Create comprehensive metrics table for easy comparison
            fig, ax = plt.subplots(figsize=(12, len(common_metrics) * 0.5 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare table data
            table_data = []
            for metric in common_metrics:
                base_val = base_metrics[metric]
                enhanced_val = enhanced_metrics[metric]
                improvement_key = f"improvement_{metric}"
                improv = improvements.get(improvement_key, 0)
                
                metric_name = metric.split('/')[-1]
                table_data.append([
                    metric_name, 
                    f"{base_val:.4f}", 
                    f"{enhanced_val:.4f}",
                    f"{improv:+.2f}%"
                ])
            
            # Create the table
            table = ax.table(
                cellText=table_data,
                colLabels=['Metric', 'Base Model', 'Enhanced Model', 'Improvement'],
                loc='center',
                cellLoc='center'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color the improvement cells
            for i, row in enumerate(table_data):
                cell = table[i+1, 3]
                improv_val = float(row[3].replace('%', '').replace('+', ''))
                if improv_val > 0:
                    cell.set_facecolor('#d8f3dc')  # light green
                elif improv_val < 0:
                    cell.set_facecolor('#ffccd5')  # light red
            
            plt.title('Model Enhancement Metrics Comparison')
            plt.tight_layout()
            
            # Save metrics table
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                table_path = temp_file.name
                plt.savefig(table_path, dpi=300, bbox_inches='tight')
                plot_paths.append(table_path)
            
            # Log the plot
            self.log_file_artifact(table_path, f"comparison/metrics_table.png")
            plt.close(fig)
            
            return plot_paths
            
        except Exception as e:
            print(f"Error generating comparison plots: {e}")
            return []
        
    def register_model_version(self, 
                              model_path: str, 
                              model_name: str = None,
                              base_run_id: str = None,
                              stage: str = "Development",
                              tags: Dict[str, str] = None) -> str:
        """
        Register a model in the MLflow Model Registry with versioning info
        
        Args:
            model_path: Path to the model file or run URI
            model_name: Name to register the model under
            base_run_id: ID of the base model run (for enhancement tracking)
            stage: Stage to register the model in (Development, Staging, Production)
            tags: Tags to apply to the registered model version
            
        Returns:
            Model version
        """
        if model_name is None:
            if self.timeframe:
                model_name = f"{self.timeframe}_trading_model"
            else:
                model_name = "trading_model"
        
        try:
            # Prepare run URI if needed
            if not model_path.startswith("runs:/"):
                if self.current_run is not None:
                    model_uri = f"runs:/{self.current_run.info.run_id}/models/{self.timeframe}/{os.path.basename(model_path)}"
                else:
                    model_uri = model_path
            else:
                model_uri = model_path
            
            # Register the model
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            version = registered_model.version
            print(f"Registered model '{model_name}' version {version}")
            
            # Set tags if provided
            if tags:
                for key, value in tags.items():
                    self.mlflow_client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=value
                    )
            
            # Track lineage for enhanced models
            if base_run_id:
                self.mlflow_client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="base_model_run_id",
                    value=base_run_id
                )
            
            # Set stage
            if stage:
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
                print(f"Transitioned model to {stage}")
            
            return version
            
        except Exception as e:
            print(f"Error registering model version: {e}")
            return None    
    def find_latest_model(self, model_type: str = "base", version_type: str = "latest") -> Tuple[str, str]:
        """
        Find the latest model of specified type using standardized tagging system
        
        Args:
            model_type: Type of model to find (e.g. 'base', 'continued', 'curriculum')
            version_type: Version type to find ('latest' or 'old')
            
        Returns:
            Tuple of (run_id, model_path) or (None, None) if not found
        """
        try:
            # Build filter string with multiple criteria based on standardized tags
            filter_str = f"tags.model_type = '{model_type}'"
            
            # Add version_type filter if specified (using the latest/old system)
            if version_type:
                filter_str += f" and tags.version_type = '{version_type}'"
                
            # Add timeframe filter if specified
            if self.timeframe:
                filter_str += f" and tags.timeframe = '{self.timeframe}'"
                
            # Search for runs with the specified filters
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_str,
                order_by=["start_time DESC"]  # Get the most recent
            )
            
            if runs.empty:
                print(f"No {model_type} model with version_type={version_type} found for timeframe {self.timeframe}")
                return None, None
            
            latest_run = runs.iloc[0]
            run_id = latest_run["run_id"]
            
            # Get the artifacts to find the model file
            artifacts = self.mlflow_client.list_artifacts(run_id, "models")
            if not artifacts:
                print(f"No model artifacts found in run {run_id}")
                return run_id, None
            
            # Get the model path - assuming model is saved as a .zip file
            timeframe_path = f"models/{self.timeframe}" if self.timeframe else "models"
            timeframe_artifacts = self.mlflow_client.list_artifacts(run_id, timeframe_path)
            
            model_artifacts = [a for a in timeframe_artifacts if a.path.endswith(".zip")]
            if not model_artifacts:
                print(f"No model file found in run {run_id}")
                return run_id, None
                
            model_path = f"runs:/{run_id}/{model_artifacts[0].path}"
            print(f"Found {model_type} model with version_type={version_type} (run_id: {run_id})")
            return run_id, model_path
            
        except Exception as e:
            print(f"Error finding latest model: {e}")
            return None, None
            
    def update_model_tags(self, run_id: str, new_tags: Dict[str, str]):
        """
        Update tags for a specific run
        
        Args:
            run_id: ID of the run to update
            new_tags: Dictionary of tags to add or update
        """
        try:
            for key, value in new_tags.items():
                self.mlflow_client.set_tag(run_id, key, value)
            print(f"Updated tags for run {run_id}")
            return True
        except Exception as e:
            print(f"Error updating tags for run {run_id}: {e}")
            return False
            
    def log_enhancement_metrics(self, base_run_id: str):
        """
        Log enhancement metrics comparing to a base model run
        
        Args:
            base_run_id: ID of the base model run
            
        Returns:
            Dictionary of improvement metrics
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
            
        try:
            # Fetch metrics from base model
            base_metrics = self.fetch_model_metrics(base_run_id)
            
            # Fetch metrics from current run
            current_metrics = {k: v for k, v in mlflow.get_run(self.current_run.info.run_id).data.metrics.items()}
            
            # Calculate improvements
            improvements = self.calculate_improvement(base_metrics, current_metrics)
            
            # Log improvement metrics
            for key, value in improvements.items():
                mlflow.log_metric(key, value)
                
            # Log the base model run ID as a tag
            mlflow.set_tag("base_model_run_id", base_run_id)
            
            # Generate comparison visualizations
            self.generate_comparison_plots(base_run_id, self.current_run.info.run_id)
            
            return improvements
            
        except Exception as e:
            print(f"Error logging enhancement metrics: {e}")
            return {}
    
    def find_models(self, 
                   model_type: Optional[str] = None,
                   version_type: Optional[str] = None,
                   timeframe: Optional[str] = None,
                   enhancement_type: Optional[str] = None,
                   max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find models matching the specified criteria with flexible filtering
        
        Args:
            model_type: Type of model to find ('base' or 'enhanced') or None for any
            version_type: Version type to find ('latest' or 'old') or None for any
            timeframe: Specific timeframe to filter by or None for any/current
            enhancement_type: Type of enhancement to filter by or None for any
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with model information
        """
        try:
            # Build dynamic filter string based on provided parameters
            filter_parts = []
            
            # Use the class timeframe if not explicitly provided
            tf = timeframe or self.timeframe
            if tf:
                filter_parts.append(f"tags.timeframe = '{tf}'")
                
            if model_type:
                filter_parts.append(f"tags.model_type = '{model_type}'")
                
            if version_type:
                filter_parts.append(f"tags.version_type = '{version_type}'")
                
            if enhancement_type:
                filter_parts.append(f"tags.enhancement_type = '{enhancement_type}'")
                
            # Combine all filters with AND
            filter_str = " and ".join(filter_parts) if filter_parts else ""
            
            # Search for runs with the specified filters
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_str,
                order_by=["start_time DESC"],
                max_results=max_results
            )
            
            if runs.empty:
                return []
                
            # Extract relevant information from the runs
            results = []
            for _, run in runs.iterrows():
                model_info = {
                    "run_id": run["run_id"],
                    "run_name": run.get("tags.mlflow.runName", "unknown"),
                    "start_time": run["start_time"],
                    "model_type": run.get("tags.model_type", "unknown"),
                    "version_type": run.get("tags.version_type", "unknown"),
                    "timeframe": run.get("tags.timeframe", "unknown"),
                }
                
                # Add enhancement-specific information if available
                if "tags.enhancement_type" in run:
                    model_info["enhancement_type"] = run["tags.enhancement_type"]
                    
                if "tags.base_model_run_id" in run:
                    model_info["base_model_run_id"] = run["tags.base_model_run_id"]
                    
                # Add key metrics if available
                for metric in ["final_eval_mean_reward", "portfolio_sharpe_ratio", "portfolio_win_rate"]:
                    metric_key = f"metrics.{metric}"
                    if metric_key in run:
                        model_info[metric] = run[metric_key]
                        
                results.append(model_info)
                
            return results
            
        except Exception as e:
            print(f"Error finding models: {e}")
            return []
          # The migrate_model_tags method has been removed as it is no longer needed
    # The system now natively uses "latest"/"old" versioning
