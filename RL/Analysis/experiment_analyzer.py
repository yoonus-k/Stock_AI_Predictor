"""
Experiment comparison and visualization utility for RL trading models
Generates visualizations of model performance across timeframes
"""

import os
import sys
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Custom imports
from RL.Utils.mlflow_manager import MLflowManager


class ExperimentAnalyzer:
    """
    Analyzer for RL trading experiments
    Compares models across different timeframes and configurations
    """
    
    def __init__(
        self,
        experiment_name: str = "stock_trading_rl",
        tracking_uri: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the experiment analyzer
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
            output_dir: Directory to save outputs
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri is None:
            # Use local MLflow tracking in the project directory
            project_root = Path(__file__).parent.parent.parent
            tracking_uri = f"file://{project_root}/mlruns"
        
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")
        
        # Set output directory
        if output_dir is None:
            self.output_dir = project_root / "RL" / "Analysis" / "Reports"
        else:
            self.output_dir = Path(output_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
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
    
    def get_runs(
        self,
        timeframes: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        status: str = "FINISHED",
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Get runs from the experiment
        
        Args:
            timeframes: List of timeframes to filter by
            model_types: List of model types to filter by
            status: Run status to filter by
            max_results: Maximum number of runs to return
            
        Returns:
            DataFrame with run information
        """
        if self.experiment_id is None:
            return pd.DataFrame()
        
        # Build filter string
        filter_str = f"experiment_id = '{self.experiment_id}'"
        
        if status:
            filter_str += f" and status = '{status}'"
        
        if timeframes:
            timeframe_filters = [f"tags.timeframe = '{tf}'" for tf in timeframes]
            filter_str += f" and ({' or '.join(timeframe_filters)})"
        
        if model_types:
            model_type_filters = [f"tags.model_type = '{mt}'" for mt in model_types]
            filter_str += f" and ({' or '.join(model_type_filters)})"
        
        # Query runs
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_str,
            max_results=max_results
        )
        
        return runs
    
    def compare_model_performance(
        self,
        metric_name: str = "final_eval_mean_reward",
        timeframes: Optional[List[str]] = None,
        save_plot: bool = True
    ):
        """
        Compare model performance across timeframes
        
        Args:
            metric_name: Name of metric to compare
            timeframes: List of timeframes to include
            save_plot: Whether to save the plot to file
        """
        if timeframes is None:
            timeframes = ["daily", "weekly", "monthly", "meta"]
        
        # Get runs with the specified metric
        all_runs = self.get_runs(timeframes=timeframes)
        
        if all_runs.empty:
            print("No runs found with the specified criteria")
            return
        
        # Filter runs with the metric
        runs_with_metric = all_runs[all_runs[f"metrics.{metric_name}"].notnull()]
        
        if runs_with_metric.empty:
            print(f"No runs found with metric: {metric_name}")
            return
        
        # Plot setup
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        # Create plot
        ax = sns.boxplot(
            x="tags.timeframe",
            y=f"metrics.{metric_name}",
            data=runs_with_metric
        )
        
        # Add individual points
        sns.stripplot(
            x="tags.timeframe",
            y=f"metrics.{metric_name}",
            data=runs_with_metric,
            size=4,
            color=".3",
            linewidth=0
        )
        
        # Labels and title
        metric_label = metric_name.replace("_", " ").title()
        plt.title(f"Comparison of {metric_label} Across Timeframes", fontsize=16)
        plt.xlabel("Timeframe", fontsize=14)
        plt.ylabel(metric_label, fontsize=14)
        
        # Save or show plot
        if save_plot:
            filename = f"performance_comparison_{metric_name}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def generate_performance_report(self, save_csv: bool = True):
        """
        Generate comprehensive performance report across all models
        
        Args:
            save_csv: Whether to save results as CSV
        """
        # Get all finished runs
        all_runs = self.get_runs(status="FINISHED")
        
        if all_runs.empty:
            print("No finished runs found")
            return
        
        # Extract relevant metrics
        metrics_to_extract = [
            "final_eval_mean_reward",
            "training_reward",
            "training_loss",
            "training_explained_variance"
        ]
        
        # Group by timeframe and model_type
        grouped = all_runs.groupby(["tags.timeframe", "tags.model_type"])
        
        # Prepare report data
        report_data = []
        
        for (timeframe, model_type), group in grouped:
            # Skip if empty
            if group.empty:
                continue
            
            row = {
                "Timeframe": timeframe,
                "Model Type": model_type,
                "Run Count": len(group),
                "Best Run ID": ""
            }
            
            # Find best run based on final evaluation reward
            if "metrics.final_eval_mean_reward" in group.columns:
                valid_runs = group[group["metrics.final_eval_mean_reward"].notnull()]
                if not valid_runs.empty:
                    best_run = valid_runs.loc[valid_runs["metrics.final_eval_mean_reward"].idxmax()]
                    row["Best Run ID"] = best_run["run_id"]
                    
                    # Add metrics from best run
                    for metric in metrics_to_extract:
                        col_name = f"metrics.{metric}"
                        if col_name in best_run and not pd.isna(best_run[col_name]):
                            metric_display = metric.replace("_", " ").title()
                            row[metric_display] = best_run[col_name]
            
            report_data.append(row)
        
        # Create report DataFrame
        report_df = pd.DataFrame(report_data)
        
        if save_csv:
            csv_path = self.output_dir / "model_performance_report.csv"
            report_df.to_csv(csv_path, index=False)
            print(f"Report saved to: {csv_path}")
        
        return report_df
    
    def plot_learning_curves(
        self,
        run_ids: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        metric_name: str = "training_reward",
        save_plot: bool = True
    ):
        """
        Plot learning curves for selected runs
        
        Args:
            run_ids: List of specific run IDs to include
            timeframes: List of timeframes to include (used if run_ids is None)
            metric_name: Metric to plot over time
            save_plot: Whether to save the plot
        """
        if run_ids is None:
            # Get runs by timeframe
            runs = self.get_runs(timeframes=timeframes)
            if runs.empty:
                print("No runs found with the specified criteria")
                return
            
            # Select top run from each timeframe
            best_runs = []
            for tf in timeframes:
                tf_runs = runs[runs["tags.timeframe"] == tf]
                if not tf_runs.empty and "metrics.final_eval_mean_reward" in tf_runs.columns:
                    valid_runs = tf_runs[tf_runs["metrics.final_eval_mean_reward"].notnull()]
                    if not valid_runs.empty:
                        best_run_id = valid_runs.loc[valid_runs["metrics.final_eval_mean_reward"].idxmax(), "run_id"]
                        best_runs.append(best_run_id)
            
            run_ids = best_runs
        
        if not run_ids:
            print("No valid runs to plot")
            return
        
        # Setup plot
        plt.figure(figsize=(14, 7))
        sns.set_theme(style="whitegrid")
        
        # Get data and plot for each run
        for run_id in run_ids:
            try:
                # Get run
                run = mlflow.get_run(run_id)
                
                # Get run metrics history
                client = mlflow.tracking.MlflowClient()
                metrics_history = client.get_metric_history(run_id, metric_name)
                
                if not metrics_history:
                    print(f"No {metric_name} metrics found for run {run_id}")
                    continue
                
                # Extract data
                steps = [m.step for m in metrics_history]
                values = [m.value for m in metrics_history]
                
                # Plot
                timeframe = run.data.tags.get("timeframe", "unknown")
                model_type = run.data.tags.get("model_type", "unknown")
                label = f"{timeframe} ({model_type})"
                
                plt.plot(steps, values, label=label, linewidth=2, alpha=0.8)
                
            except Exception as e:
                print(f"Error plotting run {run_id}: {e}")
        
        # Labels and title
        metric_label = metric_name.replace("_", " ").title()
        plt.title(f"Learning Curves: {metric_label}", fontsize=16)
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel(metric_label, fontsize=14)
        plt.legend(title="Model", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if save_plot:
            filename = f"learning_curves_{metric_name}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def generate_experiment_dashboard(self):
        """Generate comprehensive experiment dashboard with multiple visualizations"""
        # Create directory for dashboard
        dashboard_dir = self.output_dir / "dashboard"
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # List of timeframes to analyze
        timeframes = ["daily", "weekly", "monthly", "meta"]
        
        # 1. Generate performance report
        report_df = self.generate_performance_report(save_csv=True)
        
        # 2. Compare model performance across timeframes for different metrics
        metrics_to_compare = [
            "final_eval_mean_reward",
            "training_reward",
            "training_explained_variance"
        ]
        
        for metric in metrics_to_compare:
            self.compare_model_performance(metric_name=metric, timeframes=timeframes, save_plot=True)
        
        # 3. Plot learning curves for best run of each timeframe
        self.plot_learning_curves(timeframes=timeframes, metric_name="training_reward", save_plot=True)
        
        # 4. Generate HTML dashboard
        self._generate_html_dashboard(dashboard_dir, report_df)
        
        print(f"Experiment dashboard generated at: {dashboard_dir}")
        return dashboard_dir
    
    def _generate_html_dashboard(self, dashboard_dir, report_df):
        """Generate HTML dashboard with all visualizations"""
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL Stock Trading Experiment Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px 40px;
                    line-height: 1.6;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .plot-container {{
                    margin: 30px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                }}
                .footer {{
                    margin-top: 50px;
                    border-top: 1px solid #ddd;
                    padding-top: 20px;
                    font-size: 0.9em;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <h1>RL Stock Trading Experiment Dashboard</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} for experiment "{self.experiment_name}"</p>
            
            <h2>Model Performance Summary</h2>
        """
        
        # Add performance table
        if report_df is not None and not report_df.empty:
            html_content += report_df.to_html(index=False, border=0, classes='dataframe')
        else:
            html_content += "<p>No performance data available.</p>"
        
        # Add plots
        html_content += """
            <h2>Performance Comparisons</h2>
        """
        
        # Find and add all generated plots
        plot_files = list(self.output_dir.glob("*.png"))
        for plot_file in plot_files:
            if plot_file.is_file():
                plot_name = plot_file.stem.replace("_", " ").title()
                html_content += f"""
                <div class="plot-container">
                    <h3>{plot_name}</h3>
                    <img src="../{plot_file.name}" alt="{plot_name}">
                </div>
                """
        
        # Close HTML
        html_content += """
            <div class="footer">
                <p>Stock AI Predictor - RL Trading Pipeline</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        html_path = dashboard_dir / "dashboard.html"
        with open(html_path, 'w') as f:
            f.write(html_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RL trading experiments")
    
    parser.add_argument("--experiment", type=str, default="stock_trading_rl",
                      help="MLflow experiment name")
    parser.add_argument("--output", type=str, default=None,
                      help="Output directory for reports")
    parser.add_argument("--dashboard", action="store_true",
                      help="Generate comprehensive dashboard")
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(
        experiment_name=args.experiment,
        output_dir=args.output
    )
    
    if args.dashboard:
        analyzer.generate_experiment_dashboard()
    else:
        # Generate basic report
        analyzer.generate_performance_report()
        analyzer.compare_model_performance()
        analyzer.plot_learning_curves(timeframes=["daily", "weekly", "monthly", "meta"])
