"""
Comprehensive Monitoring Dashboard for Reinforcement Learning Trading Model

This script provides a unified interface to all monitoring tools for the RL trading model:
1. Training progress visualization (TensorBoard)
2. Feature importance analysis
3. Trading strategy analysis
4. Decision boundary visualization
5. Model checkpoint analysis

It serves as a central hub for model monitoring and evaluation.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


class MonitoringDashboard:
    """Main monitoring dashboard class"""
    
    def __init__(self, model_path=None, log_path=None):
        """
        Initialize the monitoring dashboard
        
        Parameters:
            model_path: Path to the trained model
            log_path: Path to the logs directory
        """
        # Setup directories
        self.model_path = model_path
        self.log_path = log_path or Path(__file__).parent.parent / "Logs"
        self.log_path = Path(self.log_path)
        
        self.tensorboard_path = self.log_path / "tensorboard"
        self.checkpoint_path = self.log_path / "checkpoints"
        self.analysis_path = self.log_path.parent / "Analysis"
        self.analysis_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize UI
        self.root = None
        self.tb_process = None
        
        # Create UI
        self.create_dashboard()
    
    def create_dashboard(self):
        """Create the monitoring dashboard UI"""
        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.title("RL Trading Model Monitoring Dashboard")
        self.root.geometry("800x600")
        self.root.configure(bg='#f5f5f5')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame, 
            text="RL Trading Model Monitoring Dashboard",
            font=("Arial", 16, "bold")
        ).pack(side=tk.LEFT)
        
        # Model selection frame
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.model_path_var = tk.StringVar(value=str(self.model_path) if self.model_path else "")
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        ttk.Button(model_frame, text="Browse...", command=self._browse_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Create tabs for different monitoring tools
        tab_control = ttk.Notebook(main_frame)
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Training progress tab
        train_tab = ttk.Frame(tab_control)
        tab_control.add(train_tab, text="Training Progress")
        self._create_training_tab(train_tab)
        
        # Feature importance tab
        feature_tab = ttk.Frame(tab_control)
        tab_control.add(feature_tab, text="Feature Importance")
        self._create_feature_tab(feature_tab)
        
        # Trading strategy tab
        strategy_tab = ttk.Frame(tab_control)
        tab_control.add(strategy_tab, text="Trading Strategy")
        self._create_strategy_tab(strategy_tab)
        
        # Decision boundaries tab
        decision_tab = ttk.Frame(tab_control)
        tab_control.add(decision_tab, text="Decision Analysis")
        self._create_decision_tab(decision_tab)
        
        # Checkpoint analysis tab
        checkpoint_tab = ttk.Frame(tab_control)
        tab_control.add(checkpoint_tab, text="Checkpoint Analysis")
        self._create_checkpoint_tab(checkpoint_tab)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set up close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start the UI
        self.root.mainloop()
    
    def _create_training_tab(self, parent):
        """Create the training progress tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # TensorBoard section
        tb_frame = ttk.LabelFrame(frame, text="TensorBoard", padding="10")
        tb_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.tb_port_var = tk.IntVar(value=6006)
        
        ttk.Label(tb_frame, text="TensorBoard Port:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(tb_frame, textvariable=self.tb_port_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.tb_status_var = tk.StringVar(value="Not running")
        status_label = ttk.Label(tb_frame, textvariable=self.tb_status_var)
        status_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        tb_button_frame = ttk.Frame(tb_frame)
        tb_button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(
            tb_button_frame, 
            text="Launch TensorBoard",
            command=self._launch_tensorboard
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            tb_button_frame, 
            text="Open in Browser",
            command=self._open_tensorboard_browser
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            tb_button_frame, 
            text="Stop TensorBoard",
            command=self._stop_tensorboard
        ).pack(side=tk.LEFT, padx=5)
        
        # Live monitoring section
        live_frame = ttk.LabelFrame(frame, text="Live Training Dashboard", padding="10")
        live_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            live_frame, 
            text="Launch Live Dashboard",
            command=self._launch_live_dashboard
        ).pack(pady=10)
        
        # Training metadata section
        metadata_frame = ttk.LabelFrame(frame, text="Training Metadata", padding="10")
        metadata_frame.pack(fill=tk.BOTH, expand=True)
        
        self.metadata_text = tk.Text(metadata_frame, height=10, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(
            metadata_frame, 
            text="Load Metadata",
            command=self._load_metadata
        ).pack(pady=5)
    
    def _create_feature_tab(self, parent):
        """Create the feature importance tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Feature importance analysis section
        fi_frame = ttk.LabelFrame(frame, text="Feature Importance Analysis", padding="10")
        fi_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(fi_frame, text="Number of evaluation episodes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.fi_episodes_var = tk.IntVar(value=5)
        ttk.Entry(fi_frame, textvariable=self.fi_episodes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(
            fi_frame, 
            text="Run Feature Importance Analysis",
            command=self._run_feature_importance
        ).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(frame, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the figure
        self.fi_fig_frame = ttk.Frame(results_frame)
        self.fi_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(
            results_frame, 
            text="Load Latest Results",
            command=self._load_feature_importance
        ).pack(pady=10)
    
    def _create_strategy_tab(self, parent):
        """Create the trading strategy tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Strategy analysis section
        strategy_frame = ttk.LabelFrame(frame, text="Trading Strategy Analysis", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(strategy_frame, text="Number of evaluation episodes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.strategy_episodes_var = tk.IntVar(value=1)
        ttk.Entry(strategy_frame, textvariable=self.strategy_episodes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(
            strategy_frame, 
            text="Run Trading Strategy Analysis",
            command=self._run_trading_strategy
        ).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(frame, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the figure
        self.strategy_fig_frame = ttk.Frame(results_frame)
        self.strategy_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Load Portfolio Performance",
            command=lambda: self._load_strategy_result("portfolio_performance.png")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Load Action Distribution",
            command=lambda: self._load_strategy_result("action_distribution.png")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Load Trade Analysis",
            command=lambda: self._load_strategy_result("trade_analysis.png")
        ).pack(side=tk.LEFT, padx=5)
    
    def _create_decision_tab(self, parent):
        """Create the decision analysis tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Decision boundary analysis section
        decision_frame = ttk.LabelFrame(frame, text="Decision Boundary Analysis", padding="10")
        decision_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            decision_frame, 
            text="Run Decision Boundary Analysis",
            command=self._run_decision_analysis
        ).pack(pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(frame, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the figure
        self.decision_fig_frame = ttk.Frame(results_frame)
        self.decision_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dropdown for selecting different visualizations
        self.decision_var = tk.StringVar()
        decision_dropdown = ttk.Combobox(results_frame, textvariable=self.decision_var)
        decision_dropdown['values'] = (
            "Decision Boundary: RSI vs MACD",
            "Decision Boundary: Price vs Position",
            "State Space (PCA)",
            "State Space (t-SNE)",
            "Feature Distribution: RSI",
            "Feature Distribution: MACD",
            "Price Patterns"
        )
        decision_dropdown.current(0)
        decision_dropdown.pack(pady=5)
        
        ttk.Button(
            results_frame, 
            text="Load Selected Visualization",
            command=self._load_decision_visualization
        ).pack(pady=5)
    
    def _create_checkpoint_tab(self, parent):
        """Create the checkpoint analysis tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Checkpoint analysis section
        checkpoint_frame = ttk.LabelFrame(frame, text="Checkpoint Analysis", padding="10")
        checkpoint_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(checkpoint_frame, text="Number of evaluation episodes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.checkpoint_episodes_var = tk.IntVar(value=3)
        ttk.Entry(checkpoint_frame, textvariable=self.checkpoint_episodes_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(
            checkpoint_frame, 
            text="Analyze Checkpoints",
            command=self._run_checkpoint_analysis
        ).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(frame, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the figure
        self.checkpoint_fig_frame = ttk.Frame(results_frame)
        self.checkpoint_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(
            results_frame, 
            text="Load Latest Results",
            command=self._load_checkpoint_analysis
        ).pack(pady=10)
    
    def _browse_model(self):
        """Browse for model file"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("ZIP Files", "*.zip"), ("All Files", "*.*")]
        )
        
        if model_path:
            self.model_path = model_path
            self.model_path_var.set(model_path)
            
            # Try to update log path based on model path
            model_dir = Path(model_path).parent
            potential_log_dir = model_dir.parent / "Logs"
            
            if potential_log_dir.exists():
                self.log_path = potential_log_dir
                self.tensorboard_path = self.log_path / "tensorboard"
                self.checkpoint_path = self.log_path / "checkpoints"
                self.status_var.set(f"Model loaded. Log path updated to {self.log_path}")
            else:
                self.status_var.set(f"Model loaded. Using default log path {self.log_path}")
    
    def _launch_tensorboard(self):
        """Launch TensorBoard server"""
        if self.tb_process is not None and self.tb_process.poll() is None:
            messagebox.showinfo("TensorBoard", "TensorBoard is already running")
            return
        
        # Check if TensorBoard directory exists
        if not self.tensorboard_path.exists():
            messagebox.showerror("Error", f"TensorBoard directory not found: {self.tensorboard_path}")
            return
        
        # Get port
        port = self.tb_port_var.get()
        
        # Launch TensorBoard in a separate process
        try:
            self.tb_process = subprocess.Popen(
                ["tensorboard", "--logdir", str(self.tensorboard_path), "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Update status
            self.tb_status_var.set(f"Running on port {port}")
            self.status_var.set(f"TensorBoard launched on port {port}")
            
            # Open in browser after short delay
            self.root.after(2000, lambda: webbrowser.open(f"http://localhost:{port}"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch TensorBoard: {e}")
            self.status_var.set("TensorBoard launch failed")
    
    def _stop_tensorboard(self):
        """Stop TensorBoard server"""
        if self.tb_process is not None:
            self.tb_process.terminate()
            self.tb_process = None
            self.tb_status_var.set("Not running")
            self.status_var.set("TensorBoard stopped")
        else:
            messagebox.showinfo("TensorBoard", "TensorBoard is not running")
    
    def _open_tensorboard_browser(self):
        """Open TensorBoard in browser"""
        port = self.tb_port_var.get()
        webbrowser.open(f"http://localhost:{port}")
        self.status_var.set(f"Opening TensorBoard in browser: http://localhost:{port}")
    
    def _launch_live_dashboard(self):
        """Launch live training dashboard"""
        # Check if log directory exists
        if not self.log_path.exists():
            messagebox.showerror("Error", f"Log directory not found: {self.log_path}")
            return
        
        # Launch dashboard script
        try:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).parent / "live_training_dashboard.py"), 
                 "--log-dir", str(self.log_path)]
            )
            self.status_var.set("Live training dashboard launched")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch live dashboard: {e}")
            self.status_var.set("Live dashboard launch failed")
    
    def _load_metadata(self):
        """Load training metadata"""
        if not self.model_path:
            messagebox.showerror("Error", "No model selected")
            return
        
        # Try to find metadata file
        model_path = Path(self.model_path)
        metadata_path = Path(f"{model_path.stem}_metadata.json")
        
        if not metadata_path.exists():
            # Try alternative location
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            
            if not metadata_path.exists():
                messagebox.showerror("Error", f"Metadata file not found: {metadata_path}")
                return
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Display metadata
            self.metadata_text.delete(1.0, tk.END)
            
            for key, value in metadata.items():
                self.metadata_text.insert(tk.END, f"{key}: {value}\n")
            
            self.status_var.set(f"Metadata loaded from {metadata_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metadata: {e}")
            self.status_var.set("Metadata loading failed")
    
    def _run_feature_importance(self):
        """Run feature importance analysis"""
        if not self.model_path:
            messagebox.showerror("Error", "No model selected")
            return
        
        # Get parameters
        episodes = self.fi_episodes_var.get()
        output_dir = self.analysis_path / "feature_importance"
        
        # Run analysis in a separate thread
        self.status_var.set("Running feature importance analysis...")
        
        def run_analysis():
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "monitor_feature_importance.py"), 
                     "--model", str(self.model_path),
                     "--output-dir", str(output_dir),
                     "--episodes", str(episodes)],
                    check=True
                )
                # Update UI in main thread
                self.root.after(0, lambda: self.status_var.set("Feature importance analysis complete"))
                self.root.after(0, self._load_feature_importance)
            except Exception as e:
                # Update UI in main thread
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Feature importance analysis failed"))
        
        Thread(target=run_analysis, daemon=True).start()
    
    def _load_feature_importance(self):
        """Load feature importance results"""
        output_dir = self.analysis_path / "feature_importance"
        
        if not output_dir.exists():
            messagebox.showerror("Error", f"Analysis directory not found: {output_dir}")
            return
        
        # Look for permutation importance plot
        plot_path = output_dir / "permutation_importance.png"
        
        if not plot_path.exists():
            messagebox.showerror("Error", f"Analysis results not found: {plot_path}")
            return
        
        # Clear previous plots
        for widget in self.fi_fig_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Display the image
        img = plt.imread(plot_path)
        plt.imshow(img)
        plt.axis('off')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, self.fi_fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        self.status_var.set(f"Feature importance results loaded from {plot_path}")
    
    def _run_trading_strategy(self):
        """Run trading strategy analysis"""
        if not self.model_path:
            messagebox.showerror("Error", "No model selected")
            return
        
        # Get parameters
        episodes = self.strategy_episodes_var.get()
        output_dir = self.analysis_path / "strategy"
        
        # Run analysis in a separate thread
        self.status_var.set("Running trading strategy analysis...")
        
        def run_analysis():
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "monitor_trading_strategy.py"), 
                     "--model", str(self.model_path),
                     "--output-dir", str(output_dir),
                     "--episodes", str(episodes)],
                    check=True
                )
                # Update UI in main thread
                self.root.after(0, lambda: self.status_var.set("Trading strategy analysis complete"))
                self.root.after(0, lambda: self._load_strategy_result("portfolio_performance.png"))
            except Exception as e:
                # Update UI in main thread
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Trading strategy analysis failed"))
        
        Thread(target=run_analysis, daemon=True).start()
    
    def _load_strategy_result(self, filename):
        """Load trading strategy result"""
        output_dir = self.analysis_path / "strategy"
        
        if not output_dir.exists():
            messagebox.showerror("Error", f"Analysis directory not found: {output_dir}")
            return
        
        # Look for specified plot
        plot_path = output_dir / filename
        
        if not plot_path.exists():
            messagebox.showerror("Error", f"Analysis result not found: {plot_path}")
            return
        
        # Clear previous plots
        for widget in self.strategy_fig_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Display the image
        img = plt.imread(plot_path)
        plt.imshow(img)
        plt.axis('off')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, self.strategy_fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        self.status_var.set(f"Trading strategy result loaded: {filename}")
    
    def _run_decision_analysis(self):
        """Run decision boundary analysis"""
        if not self.model_path:
            messagebox.showerror("Error", "No model selected")
            return
        
        output_dir = self.analysis_path / "decision_boundaries"
        
        # Run analysis in a separate thread
        self.status_var.set("Running decision boundary analysis...")
        
        def run_analysis():
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "monitor_decision_boundaries.py"), 
                     "--model", str(self.model_path),
                     "--output-dir", str(output_dir)],
                    check=True
                )
                # Update UI in main thread
                self.root.after(0, lambda: self.status_var.set("Decision boundary analysis complete"))
                self.root.after(0, self._load_decision_visualization)
            except Exception as e:
                # Update UI in main thread
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Decision boundary analysis failed"))
        
        Thread(target=run_analysis, daemon=True).start()
    
    def _load_decision_visualization(self):
        """Load decision boundary visualization"""
        output_dir = self.analysis_path / "decision_boundaries"
        
        if not output_dir.exists():
            messagebox.showerror("Error", f"Analysis directory not found: {output_dir}")
            return
        
        # Map dropdown selection to filenames
        selection = self.decision_var.get()
        filename_map = {
            "Decision Boundary: RSI vs MACD": "decision_boundary_13_14.png",
            "Decision Boundary: Price vs Position": "decision_boundary_0_26.png",
            "State Space (PCA)": "state_space_pca.png",
            "State Space (t-SNE)": "state_space_tsne.png",
            "Feature Distribution: RSI": "feature_distribution_rsi.png",
            "Feature Distribution: MACD": "feature_distribution_macd.png",
            "Price Patterns": "price_patterns.png"
        }
        
        if selection not in filename_map:
            messagebox.showerror("Error", f"Unknown selection: {selection}")
            return
        
        filename = filename_map[selection]
        plot_path = output_dir / filename
        
        if not plot_path.exists():
            messagebox.showerror("Error", f"Visualization not found: {plot_path}")
            return
        
        # Clear previous plots
        for widget in self.decision_fig_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Display the image
        img = plt.imread(plot_path)
        plt.imshow(img)
        plt.axis('off')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, self.decision_fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        self.status_var.set(f"Decision visualization loaded: {selection}")
    
    def _run_checkpoint_analysis(self):
        """Run checkpoint analysis"""
        if not self.checkpoint_path.exists():
            messagebox.showerror("Error", f"Checkpoint directory not found: {self.checkpoint_path}")
            return
        
        # Get parameters
        episodes = self.checkpoint_episodes_var.get()
        
        # Run analysis in a separate thread
        self.status_var.set("Running checkpoint analysis...")
        
        def run_analysis():
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "analyze_checkpoints.py"), 
                     "--checkpoint-dir", str(self.checkpoint_path),
                     "--episodes", str(episodes)],
                    check=True
                )
                # Update UI in main thread
                self.root.after(0, lambda: self.status_var.set("Checkpoint analysis complete"))
                self.root.after(0, self._load_checkpoint_analysis)
            except Exception as e:
                # Update UI in main thread
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Checkpoint analysis failed"))
        
        Thread(target=run_analysis, daemon=True).start()
    
    def _load_checkpoint_analysis(self):
        """Load checkpoint analysis results"""
        # Look for checkpoint analysis plots in the log directory
        plot_files = list(self.log_path.glob("checkpoint_analysis_*.png"))
        
        if not plot_files:
            messagebox.showerror("Error", "Checkpoint analysis results not found")
            return
        
        # Get the most recent plot
        plot_path = sorted(plot_files)[-1]
        
        # Clear previous plots
        for widget in self.checkpoint_fig_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Display the image
        img = plt.imread(plot_path)
        plt.imshow(img)
        plt.axis('off')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, self.checkpoint_fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        self.status_var.set(f"Checkpoint analysis results loaded from {plot_path}")
    
    def _on_close(self):
        """Handle window close event"""
        # Stop TensorBoard if running
        if self.tb_process is not None:
            self.tb_process.terminate()
        
        # Close window
        self.root.destroy()


def run_monitoring_dashboard(model_path=None, log_path=None):
    """
    Run the monitoring dashboard
    
    Parameters:
        model_path: Path to the trained model
        log_path: Path to the logs directory
    """
    dashboard = MonitoringDashboard(model_path, log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL trading model monitoring dashboard")
    parser.add_argument('--model', type=str, default=None, help="Path to the trained model")
    parser.add_argument('--log-dir', type=str, default=None, help="Directory containing logs")
    
    args = parser.parse_args()
    
    run_monitoring_dashboard(args.model, args.log_dir)
