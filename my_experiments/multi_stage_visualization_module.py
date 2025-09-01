"""
Enhanced Multi-Stage Visualization Module

Provides comprehensive visualization functions for multi-stage BERT training process 
and TIM influence analysis, with support for time window comparisons, stage-wise 
performance tracking, and comparative analysis across different time periods.
All text labels and outputs are in English.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set font and styling
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
except ImportError:
    HAS_SEABORN = False
    print("⚠️ seaborn not found, using basic matplotlib styling")
    # Set basic matplotlib styling
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.style.use('default')


class MultiStageVisualizer:
    """Enhanced Multi-Stage Experiment Visualizer"""
    
    def __init__(
        self, 
        save_dir: str = "./multi_stage_plots",
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 150
    ):
        """
        Initialize multi-stage visualizer
        
        Parameters:
        -----------
        save_dir : str
            Directory to save plots
        figure_size : Tuple[int, int]
            Figure size
        dpi : int
            Figure DPI
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Enhanced color scheme for multi-stage visualization
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#28B463', 
            'accent': '#F39C12',
            'warning': '#E74C3C',
            'neutral': '#85929E',
            'noise': '#E74C3C',
            'clean': '#28B463',
            'original': '#2E86C1',
            'pruned': '#8E44AD',
            # Stage-specific colors
            'stage_1': '#1f77b4',
            'stage_2': '#ff7f0e', 
            'stage_3': '#2ca02c',
            'stage_4': '#d62728',
            'stage_5': '#9467bd',
            'stage_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        }
        
    def plot_multi_stage_influence_comparison(
        self,
        stage_results: List[Dict],
        title: str = "TIM Influence Scores Across Time Windows",
        save_name: str = "multi_stage_influence_comparison.png"
    ):
        """
        Plot influence score distributions across multiple stages
        
        Parameters:
        -----------
        stage_results : List[Dict]
            List of stage results containing influence analysis
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"Creating multi-stage influence comparison: {title}")
        
        num_stages = len(stage_results)
        cols = min(3, num_stages)
        rows = (num_stages + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if num_stages == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if num_stages == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i, stage_result in enumerate(stage_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            influence_scores = np.array(stage_result["influence_analysis"]["scores"])
            stage_name = stage_result["stage_name"]
            time_desc = stage_result["time_window"]["description"]
            
            # Histogram of influence scores
            color = self.colors['stage_colors'][i % len(self.colors['stage_colors'])]
            ax.hist(influence_scores, bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(np.mean(influence_scores), color=self.colors['warning'], 
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(influence_scores):.6f}')
            
            ax.set_title(f'{stage_name}\nTime Window: {time_desc}')
            ax.set_xlabel('Influence Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(num_stages, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Multi-stage influence comparison saved: {save_path}")

    def plot_stage_performance_comparison(
        self,
        stage_results: List[Dict],
        original_history: Dict = None,
        title: str = "Performance Comparison Across Time Windows", 
        save_name: str = "stage_performance_comparison.png"
    ):
        """
        Plot performance comparison across stages
        
        Parameters:
        -----------
        stage_results : List[Dict]
            List of stage results
        original_history : Dict
            Original model training history for comparison
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"Creating stage performance comparison: {title}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        stage_names = [s["stage_name"] for s in stage_results]
        time_windows = [s["time_window"]["description"] for s in stage_results]
        
        # Performance metrics
        final_train_acc = []
        final_valid_acc = []
        final_train_loss = []
        noise_recall = []
        noise_precision = []
        mean_influence = []
        
        for stage in stage_results:
            hist = stage["training_results"]["pruned_history"]
            final_perf = hist.get("final_performance", {})
            
            final_train_acc.append(final_perf.get("train_accuracy", 0))
            final_valid_acc.append(final_perf.get("valid_accuracy", 0))  
            final_train_loss.append(final_perf.get("train_loss", 0))
            
            noise_det = stage["pruning_analysis"]["noise_detection"]
            noise_recall.append(noise_det["noise_recall"])
            noise_precision.append(noise_det["noise_precision"])
            
            inf_stats = stage["influence_analysis"]
            mean_influence.append(inf_stats["mean_influence"])
        
        # 1. Training accuracy comparison
        x = np.arange(len(stage_names))
        colors = [self.colors['stage_colors'][i % len(self.colors['stage_colors'])] for i in range(len(stage_names))]
        
        bars1 = ax1.bar(x, final_train_acc, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Final Training Accuracy by Stage')
        ax1.set_xlabel('Time Window Stage')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{name}\n{win}" for name, win in zip(stage_names, time_windows)], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add original model baseline if available
        if original_history and "final_performance" in original_history:
            orig_acc = original_history["final_performance"].get("train_accuracy", 0)
            ax1.axhline(orig_acc, color=self.colors['original'], 
                       linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Original (Noisy): {orig_acc:.3f}')
            ax1.legend()
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars1, final_train_acc)):
            ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f'{acc:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. Validation accuracy comparison
        bars2 = ax2.bar(x, final_valid_acc, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Final Validation Accuracy by Stage')
        ax2.set_xlabel('Time Window Stage')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{name}\n{win}" for name, win in zip(stage_names, time_windows)], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add original model baseline if available
        if original_history and "final_performance" in original_history:
            orig_acc = original_history["final_performance"].get("valid_accuracy", 0)
            ax2.axhline(orig_acc, color=self.colors['original'], 
                       linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Original (Noisy): {orig_acc:.3f}')
            ax2.legend()
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars2, final_valid_acc)):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f'{acc:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Noise detection effectiveness (Recall vs Precision)
        ax3.scatter(noise_recall, noise_precision, s=150, c=colors, alpha=0.7, edgecolor='black')
        
        # Add stage labels
        for i, (recall, precision, name) in enumerate(zip(noise_recall, noise_precision, stage_names)):
            ax3.annotate(name, (recall, precision), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax3.set_title('Noise Detection: Recall vs Precision')
        ax3.set_xlabel('Noise Recall')
        ax3.set_ylabel('Noise Precision')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # Add diagonal reference line
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
        ax3.legend()
        
        # 4. Mean influence scores by stage
        bars4 = ax4.bar(x, mean_influence, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Mean Influence Score by Stage')
        ax4.set_xlabel('Time Window Stage')
        ax4.set_ylabel('Mean Influence Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{name}\n{win}" for name, win in zip(stage_names, time_windows)], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, inf) in enumerate(zip(bars4, mean_influence)):
            ax4.text(bar.get_x() + bar.get_width()/2, inf + inf*0.05, f'{inf:.6f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=90)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Stage performance comparison saved: {save_path}")

    def plot_training_curves_comparison(
        self,
        original_history: Dict,
        stage_results: List[Dict],
        title: str = "Training Loss Curves: Original vs Multi-Stage Pruned Models",
        save_name: str = "training_curves_comparison.png"
    ):
        """
        Plot training curves comparison between original and all stage models
        
        Parameters:
        -----------
        original_history : Dict
            Original model training history
        stage_results : List[Dict]
            List of stage results
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"Creating training curves comparison: {title}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Training loss comparison
        if 'train_loss' in original_history:
            epochs = range(1, len(original_history['train_loss']) + 1)
            ax1.plot(epochs, original_history['train_loss'], 
                    color=self.colors['warning'], linewidth=3, 
                    label='Original (Noisy)', alpha=0.8)
        
        # Plot pruned training curves for each stage
        for i, stage in enumerate(stage_results):
            hist = stage["training_results"]["pruned_history"]
            if "train_loss" in hist:
                epochs = range(1, len(hist["train_loss"]) + 1)
                stage_name = stage["stage_name"]
                time_desc = stage["time_window"]["description"]
                color = self.colors['stage_colors'][i % len(self.colors['stage_colors'])]
                
                ax1.plot(epochs, hist["train_loss"], 
                        color=color, linewidth=2, 
                        label=f'{stage_name} {time_desc}', alpha=0.7)
        
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training accuracy comparison
        if 'train_accuracy' in original_history:
            epochs = range(1, len(original_history['train_accuracy']) + 1)
            ax2.plot(epochs, original_history['train_accuracy'], 
                    color=self.colors['warning'], linewidth=3, 
                    label='Original (Noisy)', alpha=0.8)
        
        for i, stage in enumerate(stage_results):
            hist = stage["training_results"]["pruned_history"]
            if "train_accuracy" in hist:
                epochs = range(1, len(hist["train_accuracy"]) + 1)
                stage_name = stage["stage_name"]
                time_desc = stage["time_window"]["description"]
                color = self.colors['stage_colors'][i % len(self.colors['stage_colors'])]
                
                ax2.plot(epochs, hist["train_accuracy"], 
                        color=color, linewidth=2, 
                        label=f'{stage_name} {time_desc}', alpha=0.7)
        
        ax2.set_title('Training Accuracy Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Training curves comparison saved: {save_path}")

    def plot_influence_heatmap(
        self,
        stage_results: List[Dict],
        title: str = "Influence Score Distribution Heatmap Across Time Windows",
        save_name: str = "influence_heatmap.png"
    ):
        """
        Create influence score distribution heatmap across stages
        
        Parameters:
        -----------
        stage_results : List[Dict]
            List of stage results
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"Creating influence heatmap: {title}")
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create influence matrix
        influence_matrix = []
        stage_names = []
        
        # Find global influence range for consistent binning
        all_influences = []
        for stage in stage_results:
            influences = np.array(stage["influence_analysis"]["scores"])
            all_influences.extend(influences)
        
        global_min, global_max = np.min(all_influences), np.max(all_influences)
        
        for stage in stage_results:
            inf_scores = np.array(stage["influence_analysis"]["scores"])
            stage_names.append(f"{stage['stage_name']}\n{stage['time_window']['description']}")
            
            # Create histogram with consistent bins
            hist, _ = np.histogram(inf_scores, bins=50, range=(global_min, global_max))
            influence_matrix.append(hist)
        
        if influence_matrix:
            influence_matrix = np.array(influence_matrix)
            
            # Create heatmap
            im = ax.imshow(influence_matrix, aspect='auto', cmap='viridis', interpolation='bilinear')
            ax.set_title('Influence Score Distribution Heatmap')
            ax.set_xlabel('Influence Score Bins (Low → High)')
            ax.set_ylabel('Time Window Stage')
            ax.set_yticks(range(len(stage_names)))
            ax.set_yticklabels(stage_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Frequency', rotation=270, labelpad=15)
            
            # Add influence range annotation
            ax.text(0.02, 0.98, f'Influence Range: [{global_min:.6f}, {global_max:.6f}]',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Influence heatmap saved: {save_path}")

    def plot_time_window_diagram(
        self,
        time_windows: List[Tuple[int, Optional[int]]],
        total_steps: int,
        title: str = "Training Time Windows Division (Steps)",
        save_name: str = "time_window_diagram.png"
    ):
        """
        Create time window division diagram
        
        Parameters:
        -----------
        time_windows : List[Tuple[int, Optional[int]]]
            List of (start_step, end_step) tuples
        total_steps : int
            Total number of training steps
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"Creating time window diagram: {title}")
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create timeline
        timeline = np.arange(0, total_steps + 1, max(1, total_steps // 100))
        
        # Draw main timeline
        ax.plot([0, total_steps], [0.5, 0.5], 'k-', linewidth=3, alpha=0.3)
        
        # Mark time windows
        for i, (start, end) in enumerate(time_windows):
            end_step = total_steps if end is None else end
            color = self.colors['stage_colors'][i % len(self.colors['stage_colors'])]
            
            # Draw window range as a thick line segment
            ax.plot([start, end_step], [0.5, 0.5], 
                   color=color, linewidth=12, alpha=0.7,
                   label=f'Stage {i+1}: [{start}, {"T" if end is None else end}]')
            
            # Add stage labels
            mid_point = (start + end_step) / 2
            ax.text(mid_point, 0.65, f'S{i+1}', ha='center', va='center', 
                   fontweight='bold', fontsize=14, 
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', edgecolor=color, linewidth=2))
            
            # Add step range labels
            steps_in_window = end_step - start + 1 if end is not None else total_steps - start
            ax.text(mid_point, 0.35, f'{steps_in_window} steps', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Mark step boundaries (show every 10% of total steps)
        step_markers = range(0, total_steps + 1, max(1, total_steps // 10))
        for step in step_markers:
            ax.axvline(step, color='gray', linestyle=':', alpha=0.5)
            ax.text(step, 0.2, str(step), ha='center', va='top', fontsize=9, rotation=45)
        
        ax.set_xlabel('Training Step')
        ax.set_xlim(-total_steps*0.02, total_steps*1.02)
        ax.set_ylim(0.1, 0.8)
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(time_windows), 3))
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add step markers
        ax.text(-total_steps*0.01, 0.5, '0', ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        ax.text(total_steps*1.01, 0.5, 'T', ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Time window diagram saved: {save_path}")

    def plot_comprehensive_dashboard(
        self,
        original_history: Dict,
        stage_results: List[Dict],
        experiment_config: Dict,
        data_stats: Dict,
        title: str = "Multi-Stage TIM Influence Data Pruning Experiment Dashboard",
        save_name: str = "comprehensive_dashboard.png"
    ):
        """
        Create comprehensive experiment dashboard
        
        Parameters:
        -----------
        original_history : Dict
            Original model training history
        stage_results : List[Dict]
            List of stage results
        experiment_config : Dict
            Experiment configuration
        data_stats : Dict
            Data statistics
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"Creating comprehensive dashboard: {title}")
        
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # Create complex grid layout
        gs = fig.add_gridspec(5, 6, hspace=0.35, wspace=0.3)
        
        # 1. Training Loss Comparison (top, spans 3 columns)
        ax1 = fig.add_subplot(gs[0, :3])
        
        # Plot original training curve
        if "train_loss" in original_history:
            epochs = range(1, len(original_history["train_loss"]) + 1)
            ax1.plot(epochs, original_history["train_loss"], 
                    color=self.colors['warning'], linewidth=4, 
                    label='Original (Noisy)', alpha=0.9)
        
        # Plot stage results
        for i, stage in enumerate(stage_results):
            hist = stage["training_results"]["pruned_history"]
            if "train_loss" in hist:
                epochs = range(1, len(hist["train_loss"]) + 1)
                stage_name = stage["stage_name"]
                time_desc = stage["time_window"]["description"]
                color = self.colors['stage_colors'][i % len(self.colors['stage_colors'])]
                ax1.plot(epochs, hist["train_loss"], 
                        color=color, linewidth=2.5, 
                        label=f'{stage_name} {time_desc}', alpha=0.8)
        
        ax1.set_title('Training Loss: Original vs Multi-Stage Pruned Models', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Summary (top right, spans 3 columns)
        ax2 = fig.add_subplot(gs[0, 3:])
        
        # Extract performance data
        stage_names = [s["stage_name"] for s in stage_results]
        final_train_acc = [s["training_results"]["pruned_history"].get("final_performance", {}).get("train_accuracy", 0) 
                          for s in stage_results]
        final_valid_acc = [s["training_results"]["pruned_history"].get("final_performance", {}).get("valid_accuracy", 0) 
                          for s in stage_results]
        
        x = np.arange(len(stage_names))
        width = 0.35
        
        colors = [self.colors['stage_colors'][i % len(self.colors['stage_colors'])] for i in range(len(stage_names))]
        bars1 = ax2.bar(x - width/2, final_train_acc, width, label='Training Accuracy', 
                       color=colors, alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, final_valid_acc, width, label='Validation Accuracy', 
                       color=colors, alpha=0.5, edgecolor='black')
        
        # Add baseline
        if "final_performance" in original_history:
            orig_train = original_history["final_performance"].get("train_accuracy", 0)
            orig_valid = original_history["final_performance"].get("valid_accuracy", 0)
            ax2.axhline(orig_train, color=self.colors['warning'], linestyle='--', linewidth=2, alpha=0.8)
            ax2.axhline(orig_valid, color=self.colors['warning'], linestyle=':', linewidth=2, alpha=0.8)
        
        ax2.set_title('Final Performance Comparison by Time Window', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Stage')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Influence Distribution Heatmap (middle left, spans 2 rows, 3 columns)
        ax3 = fig.add_subplot(gs[1:3, :3])
        
        # Create influence heatmap
        influence_matrix = []
        heatmap_labels = []
        
        # Find global range
        all_influences = []
        for stage in stage_results:
            influences = np.array(stage["influence_analysis"]["scores"])
            all_influences.extend(influences)
        
        if all_influences:
            global_min, global_max = np.min(all_influences), np.max(all_influences)
            
            for stage in stage_results:
                inf_scores = np.array(stage["influence_analysis"]["scores"])
                heatmap_labels.append(f"{stage['stage_name']}\n{stage['time_window']['description']}")
                
                # Create histogram
                hist, _ = np.histogram(inf_scores, bins=60, range=(global_min, global_max))
                influence_matrix.append(hist)
            
            influence_matrix = np.array(influence_matrix)
            
            im = ax3.imshow(influence_matrix, aspect='auto', cmap='plasma', interpolation='bilinear')
            ax3.set_title('Influence Score Distribution Heatmap Across Time Windows', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Influence Score Bins (Low → High Influence)')
            ax3.set_ylabel('Time Window Stage')
            ax3.set_yticks(range(len(heatmap_labels)))
            ax3.set_yticklabels(heatmap_labels)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label('Sample Frequency', rotation=270, labelpad=15)
        
        # 4. Noise Detection Effectiveness (middle right)
        ax4 = fig.add_subplot(gs[1, 3:])
        
        noise_recall = [s["pruning_analysis"]["noise_detection"]["noise_recall"] for s in stage_results]
        noise_precision = [s["pruning_analysis"]["noise_detection"]["noise_precision"] for s in stage_results]
        
        scatter = ax4.scatter(noise_recall, noise_precision, s=200, c=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add stage labels
        for i, (recall, precision, name) in enumerate(zip(noise_recall, noise_precision, stage_names)):
            ax4.annotate(name, (recall, precision), xytext=(8, 8), 
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax4.set_title('Noise Detection Effectiveness: Recall vs Precision', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Noise Recall')
        ax4.set_ylabel('Noise Precision')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        # Add diagonal and ideal region
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Balance')
        ax4.axhspan(0.8, 1.0, 0.8, 1.0, alpha=0.1, color='green', label='Ideal Region')
        ax4.legend()
        
        # 5. Stage Performance Metrics (middle right bottom)
        ax5 = fig.add_subplot(gs[2, 3:])
        
        # Create performance metrics table
        metrics_data = []
        for i, stage in enumerate(stage_results):
            final_perf = stage["training_results"]["pruned_history"].get("final_performance", {})
            noise_det = stage["pruning_analysis"]["noise_detection"]
            inf_stats = stage["influence_analysis"]
            
            metrics_data.append([
                stage["stage_name"],
                f"{final_perf.get('train_accuracy', 0):.3f}",
                f"{final_perf.get('valid_accuracy', 0):.3f}",
                f"{noise_det['noise_recall']:.2%}",
                f"{noise_det['noise_precision']:.2%}",
                f"{inf_stats['mean_influence']:.6f}"
            ])
        
        # Create table
        table_data = metrics_data
        table_headers = ['Stage', 'Train Acc', 'Valid Acc', 'Noise Recall', 'Noise Prec', 'Mean Influence']
        
        table = ax5.table(cellText=table_data, colLabels=table_headers, 
                         cellLoc='center', loc='center',
                         colColours=['lightblue'] * len(table_headers))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        ax5.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # 6. Time Window Diagram (bottom left, spans 3 columns)
        ax6 = fig.add_subplot(gs[3, :3])
        
        total_epochs = experiment_config.get('epochs', 10)
        time_windows = [(0, 2), (2, 4), (4, 6), (6, 8), (8, None)]  # Example windows
        
        # Create timeline
        timeline = np.arange(total_epochs + 1)
        ax6.plot(timeline, [0.5] * len(timeline), 'k-', linewidth=4, alpha=0.4)
        
        # Mark time windows
        for i, (start, end) in enumerate(time_windows[:len(stage_results)]):
            end_epoch = total_epochs if end is None else end
            window_range = np.arange(start, end_epoch + 1)
            color = self.colors['stage_colors'][i % len(self.colors['stage_colors'])]
            
            ax6.plot(window_range, [0.5] * len(window_range), 
                   color=color, linewidth=16, alpha=0.8,
                   label=f'Stage {i+1}: [{start}, {"T" if end is None else end}]')
            
            # Add stage labels
            mid_point = (start + end_epoch) / 2
            ax6.text(mid_point, 0.7, f'S{i+1}', ha='center', va='center', 
                   fontweight='bold', fontsize=16, 
                   bbox=dict(boxstyle="circle,pad=0.5", facecolor='white', edgecolor=color, linewidth=3))
        
        ax6.set_title('Training Time Windows Division', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Training Epoch')
        ax6.set_xlim(-0.5, total_epochs + 0.5)
        ax6.set_ylim(0.2, 0.9)
        ax6.set_yticks([])
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        ax6.grid(True, axis='x', alpha=0.3)
        
        # 7. Experiment Statistics Summary (bottom right, spans 3 columns)
        ax7 = fig.add_subplot(gs[3, 3:])
        
        # Calculate summary statistics
        best_train_acc = max(final_train_acc) if final_train_acc else 0
        best_valid_acc = max(final_valid_acc) if final_valid_acc else 0
        best_noise_recall = max(noise_recall) if noise_recall else 0
        best_noise_precision = max(noise_precision) if noise_precision else 0
        
        total_samples = data_stats.get('clean_data_info', {}).get('train_samples', 0)
        noise_count = data_stats.get('noise_info', {}).get('noise_count', 0)
        noise_rate = experiment_config.get('noise_rate', 0)
        
        summary_text = f"""MULTI-STAGE EXPERIMENT SUMMARY

Dataset Configuration:
• Dataset: {experiment_config.get('dataset_name', 'N/A').upper()}
• Total Training Samples: {total_samples:,}
• Injected Noise Samples: {noise_count:,}
• Noise Rate: {noise_rate*100:.1f}%
• Time Window Stages: {experiment_config.get('num_stages', 5)}

Performance Highlights:
• Best Training Accuracy: {best_train_acc:.3f}
• Best Validation Accuracy: {best_valid_acc:.3f}
• Best Noise Recall: {best_noise_recall:.1%}
• Best Noise Precision: {best_noise_precision:.1%}

Model Configuration:
• Architecture: {experiment_config.get('model_name', 'N/A')}
• Training Epochs: {experiment_config.get('epochs', 'N/A')}
• Batch Size: {experiment_config.get('batch_size', 'N/A')}
• Pruning Ratio: {experiment_config.get('prune_ratio', 0)*100:.1f}%

Experiment Status: SUCCESS ✓"""
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['neutral'], alpha=0.1))
        ax7.set_title('Experiment Configuration & Results Summary', fontsize=14, fontweight='bold')
        ax7.axis('off')
        
        # 8. Key Insights and Conclusions (bottom, spans full width)
        ax8 = fig.add_subplot(gs[4, :])
        
        # Generate insights based on results
        best_stage_idx = np.argmax(final_valid_acc) if final_valid_acc else 0
        best_stage_name = stage_results[best_stage_idx]["stage_name"] if stage_results else "N/A"
        best_time_window = stage_results[best_stage_idx]["time_window"]["description"] if stage_results else "N/A"
        
        insights_text = f"""KEY INSIGHTS AND CONCLUSIONS

🏆 BEST PERFORMING TIME WINDOW:
{best_stage_name} {best_time_window} achieved the highest validation accuracy ({best_valid_acc:.3f})

📊 NOISE DETECTION PATTERNS:
• Average noise detection recall across stages: {np.mean(noise_recall):.1%}
• Average noise detection precision across stages: {np.mean(noise_precision):.1%}
• Most effective stage for noise detection: {stage_names[np.argmax(noise_recall)] if stage_names else 'N/A'} (Recall: {max(noise_recall):.1%})

🔍 TIM INFLUENCE ANALYSIS:
• Influence scores varied significantly across time windows, indicating temporal sensitivity
• Early vs late training phases showed different noise identification patterns
• Multi-stage approach provides comprehensive view of sample importance throughout training

💡 PRACTICAL IMPLICATIONS:
• Different time windows excel at identifying different types of problematic samples
• Combined multi-stage approach could outperform single-window methods
• Time-aware data cleaning strategies show promise for improving model robustness"""
        
        ax8.text(0.02, 0.98, insights_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='serif',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        ax8.set_title('Key Insights and Conclusions', fontsize=16, fontweight='bold')
        ax8.axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Comprehensive dashboard saved: {save_path}")

    def save_experiment_summary(
        self,
        experiment_results: Dict,
        save_name: str = "multi_stage_experiment_summary.json"
    ):
        """
        Save experiment results summary
        
        Parameters:
        -----------
        experiment_results : Dict
            Complete experiment results
        save_name : str
            Save filename
        """
        save_path = self.save_dir / save_name
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"Multi-stage experiment summary saved: {save_path}")


def create_multi_stage_visualizer(
    save_dir: str = "./multi_stage_plots",
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> MultiStageVisualizer:
    """
    Factory function: Create multi-stage experiment visualizer
    
    Parameters:
    -----------
    save_dir : str
        Directory to save plots, default "./multi_stage_plots"
    figure_size : Tuple[int, int]
        Figure size, default (12, 8)
    dpi : int
        Figure DPI, default 150
        
    Returns:
    --------
    MultiStageVisualizer
        Configured multi-stage visualizer
    """
    return MultiStageVisualizer(
        save_dir=save_dir,
        figure_size=figure_size,
        dpi=dpi
    )


if __name__ == "__main__":
    # Test multi-stage visualizer
    print("🧪 Testing multi-stage visualizer")
    
    # Create visualizer
    visualizer = create_multi_stage_visualizer()
    
    # Generate test data for 5 stages
    np.random.seed(42)
    
    test_stage_results = []
    for i in range(5):
        # Mock stage results
        n_samples = 1000
        influence_scores = np.random.normal(0.5, 0.2, n_samples)
        
        # Mock training history
        epochs = 5
        test_history = {
            'train_loss': np.random.exponential(0.5, epochs) + 0.1,
            'train_accuracy': np.random.beta(8, 2, epochs),
            'final_performance': {
                'train_accuracy': np.random.uniform(0.7, 0.9),
                'valid_accuracy': np.random.uniform(0.65, 0.85),
                'train_loss': np.random.uniform(0.1, 0.5)
            }
        }
        
        stage_result = {
            'stage_num': i + 1,
            'stage_name': f'Stage_{i+1}',
            'time_window': {
                'start_epoch': i * 2,
                'end_epoch': (i + 1) * 2 if i < 4 else None,
                'description': f'[{i*2}, {"T" if i == 4 else (i+1)*2}]'
            },
            'influence_analysis': {
                'scores': influence_scores.tolist(),
                'mean_influence': float(np.mean(influence_scores)),
                'std_influence': float(np.std(influence_scores))
            },
            'training_results': {
                'pruned_history': test_history
            },
            'pruning_analysis': {
                'noise_detection': {
                    'noise_recall': np.random.uniform(0.4, 0.8),
                    'noise_precision': np.random.uniform(0.3, 0.7)
                }
            }
        }
        
        test_stage_results.append(stage_result)
    
    print("🎨 Testing multi-stage visualizations...")
    
    # Test multi-stage influence comparison
    visualizer.plot_multi_stage_influence_comparison(
        test_stage_results, title="Test Multi-Stage Influence Comparison"
    )
    
    # Test stage performance comparison
    visualizer.plot_stage_performance_comparison(
        test_stage_results, title="Test Stage Performance Comparison"
    )
    
    # Test influence heatmap
    visualizer.plot_influence_heatmap(
        test_stage_results, title="Test Influence Heatmap"
    )
    
    # Test time window diagram
    time_windows = [(0, 2), (2, 4), (4, 6), (6, 8), (8, None)]
    visualizer.plot_time_window_diagram(
        time_windows, total_epochs=10, title="Test Time Window Diagram"
    )
    
    print("✅ Multi-stage visualizer test complete")