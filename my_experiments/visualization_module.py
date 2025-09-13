"""
Visualization module

Provides visualization functions for BERT training process and TIM influence analysis,
supporting loss curves, influence distribution, and comparative analysis.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
    # Set font and styling
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
except ImportError:
    HAS_SEABORN = False
    print("⚠️ seaborn not found, using basic matplotlib styling")
    # Set basic matplotlib styling
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.style.use("default")


class ExperimentVisualizer:
    """Experiment Visualizer"""

    def __init__(
        self,
        save_dir: str = "./experiment_plots",
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 150,
    ):
        """
        Initialize visualizer

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

        # Color scheme
        self.colors = {
            "primary": "#2E86C1",
            "secondary": "#28B463",
            "accent": "#F39C12",
            "warning": "#E74C3C",
            "neutral": "#85929E",
            "noise": "#E74C3C",
            "clean": "#28B463",
            "original": "#2E86C1",
            "pruned": "#8E44AD",
        }

    def plot_training_curves(
        self,
        history: Dict,
        title: str = "Training Curves",
        save_name: str = "training_curves.png",
    ):
        """
        Plot training loss and accuracy curves

        Parameters:
        -----------
        history : Dict
            Training history data
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"📊 Creating training curves: {title}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Epoch-level loss and accuracy
        if "train_loss" in history and "train_accuracy" in history:
            epochs = range(1, len(history["train_loss"]) + 1)

            # Training loss
            ax1.plot(
                epochs,
                history["train_loss"],
                color=self.colors["primary"],
                linewidth=2,
                label="Training Loss",
            )
            if "valid_loss" in history and len(history["valid_loss"]) > 0:
                valid_steps = np.linspace(1, len(epochs), len(history["valid_loss"]))
                ax1.plot(
                    valid_steps,
                    history["valid_loss"],
                    color=self.colors["accent"],
                    linewidth=2,
                    label="Validation Loss",
                )

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss Curves")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Training accuracy
            ax2.plot(
                epochs,
                history["train_accuracy"],
                color=self.colors["secondary"],
                linewidth=2,
                label="Training Accuracy",
            )
            if "valid_accuracy" in history and len(history["valid_accuracy"]) > 0:
                valid_steps = np.linspace(
                    1, len(epochs), len(history["valid_accuracy"])
                )
                ax2.plot(
                    valid_steps,
                    history["valid_accuracy"],
                    color=self.colors["accent"],
                    linewidth=2,
                    label="Validation Accuracy",
                )

            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Accuracy Curves")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 2. Step-level detailed curves (if available)
        if "step_losses" in history:
            steps = range(1, len(history["step_losses"]) + 1)
            ax3.plot(
                steps,
                history["step_losses"],
                color=self.colors["primary"],
                linewidth=1,
                alpha=0.7,
            )
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Loss")
            ax3.set_title("Step-level Loss")
            ax3.grid(True, alpha=0.3)

        if "step_accuracies" in history:
            steps = range(1, len(history["step_accuracies"]) + 1)
            ax4.plot(
                steps,
                history["step_accuracies"],
                color=self.colors["secondary"],
                linewidth=1,
                alpha=0.7,
            )
            ax4.set_xlabel("Training Step")
            ax4.set_ylabel("Accuracy")
            ax4.set_title("Step-level Accuracy")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

        print(f"💾 Training curves saved: {save_path}")

    def plot_influence_distribution(
        self,
        influence_scores: np.ndarray,
        noise_indices: Optional[np.ndarray] = None,
        title: str = "Influence Score Distribution",
        save_name: str = "influence_distribution.png",
    ):
        """
        Plot influence score distribution

        Parameters:
        -----------
        influence_scores : np.ndarray
            Influence scores
        noise_indices : Optional[np.ndarray]
            Noise sample indices
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"📊 Creating influence distribution: {title}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Influence score histogram
        ax1.hist(
            influence_scores,
            bins=50,
            color=self.colors["primary"],
            alpha=0.7,
            edgecolor="black",
        )
        ax1.axvline(
            np.mean(influence_scores),
            color=self.colors["warning"],
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(influence_scores):.6f}",
        )
        ax1.set_xlabel("Influence Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Influence Score Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Influence ranking plot
        sorted_indices = np.argsort(influence_scores)
        ranked_scores = influence_scores[sorted_indices]

        ax2.plot(
            range(len(ranked_scores)),
            ranked_scores,
            color=self.colors["primary"],
            linewidth=2,
        )
        ax2.set_xlabel("Sample Rank (by Influence)")
        ax2.set_ylabel("Influence Score")
        ax2.set_title("Influence Ranking Curve")
        ax2.grid(True, alpha=0.3)

        # 3. If noise information available, plot noise vs clean comparison
        if noise_indices is not None:
            clean_indices = np.setdiff1d(
                np.arange(len(influence_scores)), noise_indices
            )

            # Plot influence for noise vs clean samples
            ax3.hist(
                influence_scores[noise_indices],
                bins=25,
                alpha=0.7,
                color=self.colors["noise"],
                label=f"Noise samples (n={len(noise_indices)})",
            )
            ax3.hist(
                influence_scores[clean_indices],
                bins=25,
                alpha=0.7,
                color=self.colors["clean"],
                label=f"Clean samples (n={len(clean_indices)})",
            )
            ax3.set_xlabel("Influence Score")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Noise vs Clean Sample Influence")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. Noise sample ranking positions
            noise_ranks = []
            for noise_idx in noise_indices:
                rank = np.where(sorted_indices == noise_idx)[0][0]
                rank_percentile = rank / len(influence_scores)
                noise_ranks.append(rank_percentile)

            ax4.hist(
                noise_ranks,
                bins=20,
                color=self.colors["noise"],
                alpha=0.7,
                edgecolor="black",
            )
            ax4.axvline(
                np.mean(noise_ranks),
                color=self.colors["warning"],
                linestyle="--",
                linewidth=2,
                label=f"Mean rank percentile: {np.mean(noise_ranks):.3f}",
            )
            ax4.set_xlabel("Rank Percentile")
            ax4.set_ylabel("Number of Noise Samples")
            ax4.set_title("Noise Sample Rank Distribution")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # If no noise info, show influence statistics
            stats_text = f"""Influence Statistics:
Mean: {np.mean(influence_scores):.6f}
Std: {np.std(influence_scores):.6f}
Min: {np.min(influence_scores):.6f}
Max: {np.max(influence_scores):.6f}
Samples: {len(influence_scores)}"""

            ax3.text(
                0.1,
                0.5,
                stats_text,
                transform=ax3.transAxes,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=self.colors["neutral"],
                    alpha=0.3,
                ),
            )
            ax3.set_title("Influence Statistics")
            ax3.axis("off")

            # Show top/bottom 20 influence scores
            top_k = min(20, len(influence_scores))
            most_influential = np.argsort(influence_scores)[-top_k:][::-1]
            least_influential = np.argsort(influence_scores)[:top_k]

            ax4.barh(
                range(top_k),
                influence_scores[most_influential],
                color=self.colors["secondary"],
                alpha=0.7,
                label="Most Influential",
            )
            ax4.barh(
                range(top_k, 2 * top_k),
                influence_scores[least_influential],
                color=self.colors["warning"],
                alpha=0.7,
                label="Least Influential",
            )
            ax4.set_xlabel("Influence Score")
            ax4.set_ylabel("Sample Index")
            ax4.set_title(f"Top/Bottom {top_k} Influential Samples")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

        print(f"💾 Influence distribution saved: {save_path}")

    def plot_comparative_analysis(
        self,
        original_history: Dict,
        pruned_history: Dict,
        influence_stats: Dict,
        title: str = "Before vs After Pruning Analysis",
        save_name: str = "comparative_analysis.png",
    ):
        """
        Plot comparative analysis before/after pruning

        Parameters:
        -----------
        original_history : Dict
            Original (noisy) training history
        pruned_history : Dict
            Pruned training history
        influence_stats : Dict
            Influence statistics
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"📊 Creating comparative analysis: {title}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Training loss comparison
        if "train_loss" in original_history and "train_loss" in pruned_history:
            orig_epochs = range(1, len(original_history["train_loss"]) + 1)
            pruned_epochs = range(1, len(pruned_history["train_loss"]) + 1)

            ax1.plot(
                orig_epochs,
                original_history["train_loss"],
                color=self.colors["original"],
                linewidth=2,
                label="Original (Noisy)",
            )
            ax1.plot(
                pruned_epochs,
                pruned_history["train_loss"],
                color=self.colors["pruned"],
                linewidth=2,
                label="After Pruning",
            )

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss")
            ax1.set_title("Training Loss Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Training accuracy comparison
        if "train_accuracy" in original_history and "train_accuracy" in pruned_history:
            ax2.plot(
                orig_epochs,
                original_history["train_accuracy"],
                color=self.colors["original"],
                linewidth=2,
                label="Original (Noisy)",
            )
            ax2.plot(
                pruned_epochs,
                pruned_history["train_accuracy"],
                color=self.colors["pruned"],
                linewidth=2,
                label="After Pruning",
            )

            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Training Accuracy")
            ax2.set_title("Training Accuracy Comparison")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Final performance comparison
        orig_final = original_history.get("final_performance", {})
        pruned_final = pruned_history.get("final_performance", {})

        metrics = ["train_accuracy", "valid_accuracy"]
        orig_values = [orig_final.get(m, 0) for m in metrics]
        pruned_values = [pruned_final.get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax3.bar(
            x - width / 2,
            orig_values,
            width,
            label="Original (Noisy)",
            color=self.colors["original"],
            alpha=0.7,
        )
        ax3.bar(
            x + width / 2,
            pruned_values,
            width,
            label="After Pruning",
            color=self.colors["pruned"],
            alpha=0.7,
        )

        ax3.set_xlabel("Metric")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Final Performance Comparison")
        ax3.set_xticks(x)
        ax3.set_xticklabels(["Train Accuracy", "Valid Accuracy"])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for i, (orig, pruned) in enumerate(zip(orig_values, pruned_values)):
            ax3.text(
                i - width / 2,
                orig + 0.01,
                f"{orig:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax3.text(
                i + width / 2,
                pruned + 0.01,
                f"{pruned:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Experiment setup and improvement summary
        summary_text = f"""Experiment Setup:
• Dataset: {influence_stats.get('total_samples', 'N/A')} training samples
• Noise rate: {influence_stats.get('noise_rate', 0)*100:.1f}%
• Pruned samples: {influence_stats.get('noise_count', 'N/A')}

Performance Improvement:
• Train accuracy: {orig_final.get('train_accuracy', 0):.3f} → {pruned_final.get('train_accuracy', 0):.3f} ({pruned_final.get('train_accuracy', 0) - orig_final.get('train_accuracy', 0):+.3f})
• Valid accuracy: {orig_final.get('valid_accuracy', 0):.3f} → {pruned_final.get('valid_accuracy', 0):.3f} ({pruned_final.get('valid_accuracy', 0) - orig_final.get('valid_accuracy', 0):+.3f})
• Training time: {orig_final.get('total_time', 0):.1f}s → {pruned_final.get('total_time', 0):.1f}s

TIM Influence Analysis:
• Mean influence: {influence_stats.get('mean_influence', 0):.6f}
• Influence std: {influence_stats.get('std_influence', 0):.6f}"""

        if "noise_analysis" in influence_stats:
            noise_analysis = influence_stats["noise_analysis"]
            summary_text += f"""
• Noise mean influence: {noise_analysis.get('noise_samples', {}).get('mean_influence', 0):.6f}
• Clean mean influence: {noise_analysis.get('clean_samples', {}).get('mean_influence', 0):.6f}
• Noise mean rank percentile: {noise_analysis.get('mean_noise_rank_percentile', 0):.3f}"""

        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=self.colors["neutral"], alpha=0.2
            ),
        )
        ax4.set_title("Experiment Summary")
        ax4.axis("off")

        plt.tight_layout()

        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

        print(f"💾 Comparative analysis saved: {save_path}")

    def plot_pruning_analysis(
        self,
        influence_scores: np.ndarray,
        prune_indices: np.ndarray,
        keep_indices: np.ndarray,
        noise_indices: Optional[np.ndarray] = None,
        title: str = "Data Pruning Analysis",
        save_name: str = "pruning_analysis.png",
    ):
        """
        Plot data pruning analysis

        Parameters:
        -----------
        influence_scores : np.ndarray
            Influence scores
        prune_indices : np.ndarray
            Pruned sample indices
        keep_indices : np.ndarray
            Kept sample indices
        noise_indices : Optional[np.ndarray]
            Noise sample indices
        title : str
            Plot title
        save_name : str
            Save filename
        """
        print(f"📊 Creating pruning analysis: {title}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Pruned vs kept sample influence comparison
        ax1.hist(
            influence_scores[prune_indices],
            bins=30,
            alpha=0.7,
            color=self.colors["warning"],
            label=f"Pruned samples (n={len(prune_indices)})",
        )
        ax1.hist(
            influence_scores[keep_indices],
            bins=30,
            alpha=0.7,
            color=self.colors["secondary"],
            label=f"Kept samples (n={len(keep_indices)})",
        )
        ax1.set_xlabel("Influence Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Pruned vs Kept Sample Influence Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Influence ranking and pruning decision
        sorted_indices = np.argsort(influence_scores)
        ranks = np.arange(len(influence_scores))

        # Mark pruned and kept samples
        prune_mask = np.isin(sorted_indices, prune_indices)
        keep_mask = np.isin(sorted_indices, keep_indices)

        ax2.scatter(
            ranks[prune_mask],
            influence_scores[sorted_indices[prune_mask]],
            color=self.colors["warning"],
            alpha=0.7,
            s=20,
            label="Pruned samples",
        )
        ax2.scatter(
            ranks[keep_mask],
            influence_scores[sorted_indices[keep_mask]],
            color=self.colors["secondary"],
            alpha=0.7,
            s=20,
            label="Kept samples",
        )

        # Add pruning threshold line
        threshold_rank = len(prune_indices)
        ax2.axvline(
            threshold_rank,
            color=self.colors["accent"],
            linestyle="--",
            linewidth=2,
            label="Pruning threshold",
        )

        ax2.set_xlabel("Influence Rank")
        ax2.set_ylabel("Influence Score")
        ax2.set_title("Influence Ranking and Pruning Decision")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. If noise info available, analyze pruning effectiveness
        if noise_indices is not None:
            # Calculate noise capture ratio
            pruned_noise = np.intersect1d(prune_indices, noise_indices)
            noise_recall = (
                len(pruned_noise) / len(noise_indices) if len(noise_indices) > 0 else 0
            )
            noise_precision = (
                len(pruned_noise) / len(prune_indices) if len(prune_indices) > 0 else 0
            )

            # Create confusion matrix style visualization
            categories = ["Pruned samples", "Kept samples"]
            noise_counts = [
                len(np.intersect1d(prune_indices, noise_indices)),
                len(np.intersect1d(keep_indices, noise_indices)),
            ]
            clean_counts = [
                len(prune_indices) - noise_counts[0],
                len(keep_indices) - noise_counts[1],
            ]

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax3.bar(
                x - width / 2,
                noise_counts,
                width,
                label="Noise samples",
                color=self.colors["noise"],
                alpha=0.7,
            )
            bars2 = ax3.bar(
                x + width / 2,
                clean_counts,
                width,
                label="Clean samples",
                color=self.colors["clean"],
                alpha=0.7,
            )

            ax3.set_xlabel("Pruning Decision")
            ax3.set_ylabel("Sample Count")
            ax3.set_title(
                f"Noise Capture Effectiveness (Recall: {noise_recall:.2%}, Precision: {noise_precision:.2%})"
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # 4. Pruning statistics summary
        stats_text = f"""Pruning Statistics:
Total samples: {len(influence_scores)}
Pruned samples: {len(prune_indices)} ({len(prune_indices)/len(influence_scores)*100:.1f}%)
Kept samples: {len(keep_indices)} ({len(keep_indices)/len(influence_scores)*100:.1f}%)

Influence threshold: {influence_scores[sorted_indices[len(prune_indices)-1]]:.6f}

Pruned sample influence:
• Mean: {np.mean(influence_scores[prune_indices]):.6f}
• Std: {np.std(influence_scores[prune_indices]):.6f}

Kept sample influence:
• Mean: {np.mean(influence_scores[keep_indices]):.6f}
• Std: {np.std(influence_scores[keep_indices]):.6f}"""

        if noise_indices is not None:
            stats_text += f"""

Noise Detection Effectiveness:
• Noise samples: {len(noise_indices)}
• Captured noise: {len(np.intersect1d(prune_indices, noise_indices))}
• Noise recall: {noise_recall:.2%}
• Pruning precision: {noise_precision:.2%}"""

        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=self.colors["neutral"], alpha=0.2
            ),
        )
        ax4.set_title("Pruning Statistics Summary")
        ax4.axis("off")

        plt.tight_layout()

        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

        print(f"💾 Pruning analysis saved: {save_path}")

    def save_experiment_summary(
        self, experiment_results: Dict, save_name: str = "experiment_summary.json"
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

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"💾 Experiment summary saved: {save_path}")


def create_visualizer(
    save_dir: str = "./experiment_plots",
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> ExperimentVisualizer:
    """
    Factory function: Create experiment visualizer

    Parameters:
    -----------
    save_dir : str
        Directory to save plots, default "./experiment_plots"
    figure_size : Tuple[int, int]
        Figure size, default (12, 8)
    dpi : int
        Figure DPI, default 150

    Returns:
    --------
    ExperimentVisualizer
        Configured visualizer
    """
    return ExperimentVisualizer(save_dir=save_dir, figure_size=figure_size, dpi=dpi)


if __name__ == "__main__":
    # Test visualizer
    print("🧪 Testing visualizer")

    # Create visualizer
    visualizer = create_visualizer()

    # Generate test data
    np.random.seed(42)

    # Mock training history
    epochs = 5
    test_history = {
        "train_loss": np.random.exponential(0.5, epochs) + 0.1,
        "train_accuracy": np.random.beta(8, 2, epochs),
        "valid_loss": np.random.exponential(0.6, epochs // 2) + 0.2,
        "valid_accuracy": np.random.beta(7, 3, epochs // 2),
        "step_losses": np.random.exponential(0.5, epochs * 20) + 0.1,
        "step_accuracies": np.random.beta(8, 2, epochs * 20),
    }

    # Mock influence data
    n_samples = 100
    influence_scores = np.random.normal(0.5, 0.2, n_samples)
    noise_indices = np.random.choice(n_samples, 30, replace=False)

    print("🎨 Testing training curve plotting...")
    visualizer.plot_training_curves(test_history, title="Test Training Curves")

    print("🎨 Testing influence distribution plotting...")
    visualizer.plot_influence_distribution(
        influence_scores, noise_indices, title="Test Influence Distribution"
    )

    print("✅ Visualizer test complete")
