"""
Pattern Visualization Dashboard

This script provides a GUI interface for visualizing and analyzing price patterns
and clusters from the database.
"""

import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import json

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent  # Navigate up to project root
sys.path.append(str(project_root))

from Pattern.Utils.pattern_visualizer import PatternVisualizer

class PatternDashboard:
    """
    A Tkinter-based dashboard for pattern visualization and analysis.
    """
    
    def __init__(self, root):
        """
        Initialize the dashboard.
        
        Parameters:
        -----------
        root : tk.Tk
            Root Tkinter window
        """
        self.root = root
        self.root.title("Pattern Visualization Dashboard")
        self.root.geometry("1200x800")
        
        # Create the visualizer
        self.visualizer = PatternVisualizer()
        
        # Initialize variables
        self.selected_stock_id = tk.IntVar()
        self.selected_timeframe_id = tk.IntVar()
        self.selected_config_id = tk.IntVar()
        self.selected_cluster_id = tk.IntVar()
        
        # Initialize data containers
        self.stocks = []
        self.timeframes = []
        self.configs = []
        self.clusters = []
        
        # Create the GUI
        self.create_gui()
        
        # Load initial data
        self.load_stocks()
    
    def create_gui(self):
        """Create the GUI components."""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Create selection frames for stock, timeframe, config, and cluster
        stock_frame = ttk.LabelFrame(control_frame, text="Stock", padding=5)
        stock_frame.pack(fill=tk.X, pady=(0, 5))
        
        timeframe_frame = ttk.LabelFrame(control_frame, text="Timeframe", padding=5)
        timeframe_frame.pack(fill=tk.X, pady=(0, 5))
        
        config_frame = ttk.LabelFrame(control_frame, text="Configuration", padding=5)
        config_frame.pack(fill=tk.X, pady=(0, 5))
        
        cluster_frame = ttk.LabelFrame(control_frame, text="Cluster", padding=5)
        cluster_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Create comboboxes for selection
        self.stock_combo = ttk.Combobox(stock_frame, width=30)
        self.stock_combo.pack(fill=tk.X, pady=5)
        self.stock_combo.bind("<<ComboboxSelected>>", self.on_stock_selected)
        
        self.timeframe_combo = ttk.Combobox(timeframe_frame, width=30)
        self.timeframe_combo.pack(fill=tk.X, pady=5)
        self.timeframe_combo.bind("<<ComboboxSelected>>", self.on_timeframe_selected)
        
        self.config_combo = ttk.Combobox(config_frame, width=30)
        self.config_combo.pack(fill=tk.X, pady=5)
        self.config_combo.bind("<<ComboboxSelected>>", self.on_config_selected)
        
        self.cluster_combo = ttk.Combobox(cluster_frame, width=30)
        self.cluster_combo.pack(fill=tk.X, pady=5)
        self.cluster_combo.bind("<<ComboboxSelected>>", self.on_cluster_selected)
        
        # Create buttons frame
        buttons_frame = ttk.Frame(control_frame, padding=5)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        # Create action buttons
        ttk.Button(buttons_frame, text="View Cluster", command=self.on_view_cluster).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="View Patterns", command=self.on_view_patterns).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="View Histogram", command=self.on_view_histogram).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Compare Clusters", command=self.on_compare_clusters).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Performance Comparison", command=self.on_performance_comparison).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="MFE/MAE Analysis", command=self.on_mfe_mae_analysis).pack(fill=tk.X, pady=2)
        
        # Create export buttons
        export_frame = ttk.LabelFrame(control_frame, text="Export", padding=5)
        export_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Button(export_frame, text="Generate Report", command=self.on_generate_report).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export to CSV", command=self.on_export_to_csv).pack(fill=tk.X, pady=2)
        
        # Create right panel for visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a figure for plotting
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=5)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Create status message
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(fill=tk.X)
    
    def load_stocks(self):
        """Load stocks from the database into the combobox."""
        try:
            self.stocks = self.visualizer.get_all_stocks()
            stock_items = [f"{s['stock_id']}: {s['symbol']} - {s['name']}" for s in self.stocks]
            self.stock_combo['values'] = stock_items
            
            if stock_items:
                self.stock_combo.current(0)
                self.on_stock_selected(None)
        except Exception as e:
            self.status_var.set(f"Error loading stocks: {e}")
            messagebox.showerror("Error", f"Error loading stocks: {e}")
    
    def on_stock_selected(self, event):
        """Handle stock selection."""
        try:
            selected_index = self.stock_combo.current()
            if selected_index >= 0:
                self.selected_stock_id.set(self.stocks[selected_index]['stock_id'])
                self.load_timeframes()
        except Exception as e:
            self.status_var.set(f"Error selecting stock: {e}")
            messagebox.showerror("Error", f"Error selecting stock: {e}")
    
    def load_timeframes(self):
        """Load timeframes from the database into the combobox."""
        try:
            self.timeframes = self.visualizer.get_all_timeframes()
            timeframe_items = [f"{tf['timeframe_id']}: {tf['name']} ({tf['minutes']} min)" for tf in self.timeframes]
            self.timeframe_combo['values'] = timeframe_items
            
            if timeframe_items:
                self.timeframe_combo.current(0)
                self.on_timeframe_selected(None)
        except Exception as e:
            self.status_var.set(f"Error loading timeframes: {e}")
            messagebox.showerror("Error", f"Error loading timeframes: {e}")
    
    def on_timeframe_selected(self, event):
        """Handle timeframe selection."""
        try:
            selected_index = self.timeframe_combo.current()
            if selected_index >= 0:
                self.selected_timeframe_id.set(self.timeframes[selected_index]['timeframe_id'])
                self.load_configs()
        except Exception as e:
            self.status_var.set(f"Error selecting timeframe: {e}")
            messagebox.showerror("Error", f"Error selecting timeframe: {e}")
    
    def load_configs(self):
        """Load configurations from the database into the combobox."""
        try:
            self.configs = self.visualizer.get_configs_for_stock(self.selected_stock_id.get())
            config_items = [f"{c['config_id']}: {c['name']} (PIPs: {c['n_pips']}, LB: {c['lookback']})" for c in self.configs]
            self.config_combo['values'] = config_items
            
            if config_items:
                self.config_combo.current(0)
                self.on_config_selected(None)
            else:
                self.config_combo.set("")
                self.clusters = []
                self.cluster_combo['values'] = []
                self.cluster_combo.set("")
        except Exception as e:
            self.status_var.set(f"Error loading configurations: {e}")
            messagebox.showerror("Error", f"Error loading configurations: {e}")
    
    def on_config_selected(self, event):
        """Handle configuration selection."""
        try:
            selected_index = self.config_combo.current()
            if selected_index >= 0:
                self.selected_config_id.set(self.configs[selected_index]['config_id'])
                self.load_clusters()
        except Exception as e:
            self.status_var.set(f"Error selecting configuration: {e}")
            messagebox.showerror("Error", f"Error selecting configuration: {e}")
    
    def load_clusters(self):
        """Load clusters from the database into the combobox."""
        try:
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            
            self.clusters = self.visualizer.get_all_clusters(stock_id, timeframe_id, config_id)
            
            cluster_items = []
            for c in self.clusters:
                outcome = f"{c['outcome']:.4f}" if c['outcome'] is not None else "N/A"
                label = c['label'] if c['label'] else ""
                cluster_items.append(f"{c['cluster_id']}: {label} (Outcome: {outcome}, Patterns: {c['pattern_count']})")
            
            self.cluster_combo['values'] = cluster_items
            
            if cluster_items:
                self.cluster_combo.current(0)
                self.on_cluster_selected(None)
            else:
                self.cluster_combo.set("")
        except Exception as e:
            self.status_var.set(f"Error loading clusters: {e}")
            messagebox.showerror("Error", f"Error loading clusters: {e}")
    
    def on_cluster_selected(self, event):
        """Handle cluster selection."""
        try:
            selected_index = self.cluster_combo.current()
            if selected_index >= 0:
                self.selected_cluster_id.set(self.clusters[selected_index]['cluster_id'])
        except Exception as e:
            self.status_var.set(f"Error selecting cluster: {e}")
            messagebox.showerror("Error", f"Error selecting cluster: {e}")
    
    def on_view_cluster(self):
        """Display the selected cluster."""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            cluster_id = self.selected_cluster_id.get()
            
            # Get stock and timeframe info
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            
            # Get cluster info
            cluster_info = self.visualizer.get_cluster_info(cluster_id, stock_id, timeframe_id, config_id)
            
            # Create a subplot
            ax = self.fig.add_subplot(111)
            
            # Plot the pattern
            if cluster_info['avg_price_points']:
                x = np.arange(len(cluster_info['avg_price_points']))
                ax.plot(x, cluster_info['avg_price_points'], marker='o', linewidth=2,
                      color='blue', label=f"Cluster {cluster_id}")
                
                # Add labels and title
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Normalized Price')
                ax.set_title(f"Cluster {cluster_id} Pattern for {stock_info['symbol']} ({timeframe_info['name']})")
                ax.grid(True)
                
                # Highlight expected outcome direction
                outcome = cluster_info['outcome']
                if outcome > 0:
                    ax.axhspan(cluster_info['avg_price_points'][-1], 
                            cluster_info['avg_price_points'][-1] + 0.2, 
                            alpha=0.2, color='green', label='Expected Bullish Outcome')
                elif outcome < 0:
                    ax.axhspan(cluster_info['avg_price_points'][-1] - 0.2, 
                            cluster_info['avg_price_points'][-1], 
                            alpha=0.2, color='red', label='Expected Bearish Outcome')
                
                ax.legend()
                
                # Add text annotations for statistics
                stats_text = f"Pattern Count: {cluster_info['pattern_count']}\n"
                stats_text += f"Outcome: {cluster_info['outcome']:.4f}\n"
                stats_text += f"Max Gain: {cluster_info['max_gain']:.4f}\n"
                stats_text += f"Max Drawdown: {cluster_info['max_drawdown']:.4f}\n"
                stats_text += f"RR Ratio: {cluster_info['reward_risk_ratio']:.2f}"
                
                ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.status_var.set(f"Displaying cluster {cluster_id} for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error viewing cluster: {e}")
            messagebox.showerror("Error", f"Error viewing cluster: {e}")
    
    def on_view_patterns(self):
        """Display patterns in the selected cluster."""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            cluster_id = self.selected_cluster_id.get()
            
            # Plot the patterns
            self.visualizer.plot_cluster_patterns(cluster_id, stock_id, timeframe_id, config_id)
            
            # Update status
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            self.status_var.set(f"Displaying patterns for cluster {cluster_id} for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error viewing patterns: {e}")
            messagebox.showerror("Error", f"Error viewing patterns: {e}")
    
    def on_view_histogram(self):
        """Display histogram of outcomes for the selected cluster."""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            cluster_id = self.selected_cluster_id.get()
            
            # Plot the histogram
            self.visualizer.plot_cluster_histogram(cluster_id, stock_id, timeframe_id, config_id)
            
            # Update status
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            self.status_var.set(f"Displaying histogram for cluster {cluster_id} for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error viewing histogram: {e}")
            messagebox.showerror("Error", f"Error viewing histogram: {e}")
    
    def on_compare_clusters(self):
        """Compare all clusters for the selected configuration."""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            
            # Plot all clusters
            self.visualizer.plot_all_clusters(stock_id, timeframe_id, config_id)
            
            # Update status
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            self.status_var.set(f"Comparing all clusters for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error comparing clusters: {e}")
            messagebox.showerror("Error", f"Error comparing clusters: {e}")
    
    def on_performance_comparison(self):
        """Display performance comparison of all clusters."""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            
            # Plot performance comparison
            self.visualizer.plot_cluster_performance_comparison(stock_id, timeframe_id, config_id)
            
            # Update status
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            self.status_var.set(f"Displaying performance comparison for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error comparing performance: {e}")
            messagebox.showerror("Error", f"Error comparing performance: {e}")
    
    def on_mfe_mae_analysis(self):
        """Display MFE/MAE analysis for the selected cluster."""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            cluster_id = self.selected_cluster_id.get()
            
            # Plot MFE/MAE analysis
            self.visualizer.plot_mfe_mae_analysis(stock_id, timeframe_id, config_id, cluster_id)
            
            # Update status
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            self.status_var.set(f"Displaying MFE/MAE analysis for cluster {cluster_id} for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error analyzing MFE/MAE: {e}")
            messagebox.showerror("Error", f"Error analyzing MFE/MAE: {e}")
    
    def on_generate_report(self):
        """Generate a report for the selected configuration."""
        try:
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            
            # Generate report
            report = self.visualizer.generate_pattern_report(stock_id, timeframe_id, config_id)
            
            # Show the report in a dialog
            report_dialog = tk.Toplevel(self.root)
            report_dialog.title("Pattern Analysis Report")
            report_dialog.geometry("800x600")
            
            # Create a text widget to display the report
            report_text = tk.Text(report_dialog, wrap=tk.WORD, padx=10, pady=10)
            report_text.pack(fill=tk.BOTH, expand=True)
            
            # Add a scrollbar
            scrollbar = ttk.Scrollbar(report_text, command=report_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            report_text.config(yscrollcommand=scrollbar.set)
            
            # Insert the report text
            report_text.insert(tk.END, report)
            report_text.config(state=tk.DISABLED)  # Make it read-only
            
            # Add a save button
            def save_report():
                filename = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
                )
                if filename:
                    self.visualizer.save_report_to_file(report, filename)
                    messagebox.showinfo("Success", f"Report saved to {filename}")
            
            save_button = ttk.Button(report_dialog, text="Save Report", command=save_report)
            save_button.pack(pady=10)
            
            # Update status
            stock_info = self.visualizer.get_stock_info(stock_id)
            timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
            self.status_var.set(f"Generated report for {stock_info['symbol']} ({timeframe_info['name']})")
        except Exception as e:
            self.status_var.set(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Error generating report: {e}")
    
    def on_export_to_csv(self):
        """Export cluster data to CSV."""
        try:
            # Get IDs
            stock_id = self.selected_stock_id.get()
            timeframe_id = self.selected_timeframe_id.get()
            config_id = self.selected_config_id.get()
            
            # Ask for filename
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if filename:
                # Export data
                self.visualizer.export_cluster_data(stock_id, timeframe_id, config_id, filename)
                
                # Update status
                stock_info = self.visualizer.get_stock_info(stock_id)
                timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
                self.status_var.set(f"Exported data for {stock_info['symbol']} ({timeframe_info['name']}) to {filename}")
                
                messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            self.status_var.set(f"Error exporting data: {e}")
            messagebox.showerror("Error", f"Error exporting data: {e}")

def main():
    root = tk.Tk()
    app = PatternDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
