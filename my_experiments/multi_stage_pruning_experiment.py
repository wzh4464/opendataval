#!/usr/bin/env python3
"""
Multi-Stage TIM Influence Data Pruning Experiment

This experiment divides the training process into 5 time windows and performs 
TIM influence computation and data cleaning for each stage:
- Stage 1: [0, t1] 
- Stage 2: [t1, t2]
- Stage 3: [t2, t3] 
- Stage 4: [t3, t4]
- Stage 5: [t4, T]

For each stage, we:
1. Compute TIM influence scores for that time window
2. Identify and prune low-influence samples  
3. Retrain on cleaned data
4. Compare performance across all stages
"""

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from my_experiments.bert_training_module import create_bert_trainer
from my_experiments.noise_data_module import create_noise_processor
from my_experiments.tim_influence_module import create_tim_calculator
from my_experiments.multi_stage_visualization_module import create_multi_stage_visualizer


class MultiStagePruningExperiment:
    """Multi-Stage TIM Data Pruning Experiment"""

    def __init__(
        self,
        # Data configuration
        dataset_name: str = "imdb",
        train_count: int = 1000,
        valid_count: int = 200,
        test_count: int = 200,
        noise_rate: float = 0.3,
        # Model configuration
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        num_train_layers: int = 2,
        # Training configuration
        epochs: int = 10,  # More epochs to divide into 5 stages
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        # Multi-stage TIM configuration
        num_stages: int = 5,
        tim_batch_size: int = 8,
        regularization: float = 0.01,
        finite_diff_eps: float = 1e-5,
        # Pruning configuration
        prune_ratio: float = 0.3,
        # Experiment configuration
        random_state: int = 42,
        device: str = "auto",
        output_dir: str = "./multi_stage_results",
        save_plots: bool = True,
    ):
        """
        Initialize multi-stage pruning experiment

        Parameters:
        -----------
        dataset_name : str
            Dataset name, default "imdb"
        train_count : int
            Number of training samples, default 1000
        valid_count : int
            Number of validation samples, default 200
        test_count : int
            Number of test samples, default 200
        noise_rate : float
            Label noise ratio, default 0.3 (30%)
        model_name : str
            BERT model name, default "distilbert-base-uncased"
        num_classes : int
            Number of classes, default 2
        dropout_rate : float
            Dropout rate, default 0.2
        num_train_layers : int
            Number of fine-tuning layers, default 2
        epochs : int
            Total training epochs, default 10
        batch_size : int
            Batch size, default 16
        learning_rate : float
            Learning rate, default 2e-5
        num_stages : int
            Number of time window stages, default 5
        tim_batch_size : int
            TIM batch size, default 8
        regularization : float
            L2 regularization, default 0.01
        finite_diff_eps : float
            Finite difference parameter, default 1e-5
        prune_ratio : float
            Pruning ratio, default 0.3 (30%)
        random_state : int
            Random seed, default 42
        device : str
            Computing device, default "auto"
        output_dir : str
            Output directory, default "./multi_stage_results"
        save_plots : bool
            Whether to save plots, default True
        """

        # Save all configuration
        self.config = {
            "dataset_name": dataset_name,
            "train_count": train_count,
            "valid_count": valid_count,
            "test_count": test_count,
            "noise_rate": noise_rate,
            "model_name": model_name,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "num_train_layers": num_train_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_stages": num_stages,
            "tim_batch_size": tim_batch_size,
            "regularization": regularization,
            "finite_diff_eps": finite_diff_eps,
            "prune_ratio": prune_ratio,
            "random_state": random_state,
            "device": device,
            "output_dir": output_dir,
            "save_plots": save_plots,
        }

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate time windows for stages (based on steps, not epochs)
        self.time_windows = self._calculate_time_windows(
            total_epochs=epochs, 
            num_stages=num_stages,
            batch_size=batch_size,
            train_count=train_count
        )
        
        # Initialize components
        self.data_processor = None
        self.bert_trainer = None
        self.visualizer = None

        # Experiment results storage
        self.results = {
            "config": self.config,
            "time_windows": self.time_windows,
            "start_time": None,
            "end_time": None,
            "status": "initialized",
            "data_stats": {},
            "original_training": {},
            "stage_results": {},  # Results for each stage
            "comparative_analysis": {},
            "error_log": [],
        }

    def _calculate_time_windows(self, total_epochs: int, num_stages: int, batch_size: int = 16, train_count: int = 1000) -> List[Tuple[int, Optional[int]]]:
        """
        Calculate time windows for each stage based on training STEPS (not epochs)
        
        Parameters:
        -----------
        total_epochs : int
            Total number of training epochs
        num_stages : int
            Number of stages to divide training into
        batch_size : int
            Batch size for calculating steps per epoch
        train_count : int
            Number of training samples
            
        Returns:
        --------
        List[Tuple[int, Optional[int]]]
            List of (start_step, end_step) tuples for each stage
        """
        # Calculate total training steps
        steps_per_epoch = (train_count + batch_size - 1) // batch_size  # Ceiling division
        total_steps = total_epochs * steps_per_epoch
        
        print(f"📊 Time window calculation:")
        print(f"   Training samples: {train_count}")
        print(f"   Batch size: {batch_size}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total epochs: {total_epochs}")
        print(f"   Total steps: {total_steps}")
        
        # Divide steps evenly across stages
        steps_per_stage = total_steps // num_stages
        remainder_steps = total_steps % num_stages
        
        time_windows = []
        current_step = 0
        
        for stage in range(num_stages):
            start_step = current_step
            
            # Add remainder steps to early stages
            stage_steps = steps_per_stage + (1 if stage < remainder_steps else 0)
            end_step = current_step + stage_steps - 1  # End step is inclusive
            
            # For the last stage, end at T (None)
            if stage == num_stages - 1:
                time_windows.append((start_step, None))  # [t4, T]
            else:
                time_windows.append((start_step, end_step))
                
            current_step = end_step + 1  # Next stage starts after current end
            
        # Print time windows for verification
        print(f"   Time windows (steps):")
        for i, (start, end) in enumerate(time_windows, 1):
            end_desc = "T" if end is None else str(end)
            steps_in_window = (total_steps - start) if end is None else (end - start + 1)
            epochs_equiv = steps_in_window / steps_per_epoch
            print(f"     Stage {i}: [{start}, {end_desc}] - {steps_in_window} steps (~{epochs_equiv:.1f} epochs)")
            
        return time_windows

    def setup_components(self):
        """Setup experiment components"""
        print("⚙️  Setting up experiment components...")

        try:
            # 1. Data processor
            self.data_processor = create_noise_processor(
                dataset_name=self.config["dataset_name"],
                train_count=self.config["train_count"],
                valid_count=self.config["valid_count"],
                test_count=self.config["test_count"],
                noise_rate=self.config["noise_rate"],
                random_state=self.config["random_state"],
            )

            # 2. BERT trainer
            self.bert_trainer = create_bert_trainer(
                model_name=self.config["model_name"],
                num_classes=self.config["num_classes"],
                dropout_rate=self.config["dropout_rate"],
                num_train_layers=self.config["num_train_layers"],
                device=self.config["device"],
                random_state=self.config["random_state"],
            )

            # 3. Multi-stage visualizer
            if self.config["save_plots"]:
                self.visualizer = create_multi_stage_visualizer(
                    save_dir=str(self.output_dir / "plots")
                )

            print("✅ Component setup complete")

        except Exception as e:
            error_msg = f"Component setup failed: {e}"
            print(f"❌ {error_msg}")
            self.results["error_log"].append(error_msg)
            raise

    def prepare_data(self) -> Dict:
        """Prepare experiment data"""
        print("🔄 Preparing experiment data...")

        try:
            # 1. Load clean data
            clean_data = self.data_processor.load_clean_data()

            # 2. Inject label noise
            noisy_data, noise_indices = self.data_processor.inject_label_noise()

            # 3. Get noise statistics
            noise_stats = self.data_processor.get_noise_statistics()

            # Save data statistics
            self.results["data_stats"] = {
                "clean_data_info": {
                    "train_samples": len(clean_data["y_train"]),
                    "valid_samples": len(clean_data["y_valid"]),
                    "test_samples": len(clean_data["y_test"]),
                },
                "noise_info": noise_stats,
            }

            print("✅ Data preparation complete")
            print(f"   Total training samples: {len(noisy_data['y_train'])}")
            print(f"   Noise samples: {len(noise_indices)} ({len(noise_indices) / len(noisy_data['y_train']) * 100:.1f}%)")

            return noisy_data, noise_indices

        except Exception as e:
            error_msg = f"Data preparation failed: {e}"
            print(f"❌ {error_msg}")
            self.results["error_log"].append(error_msg)
            raise

    def train_original_model(self, data: Dict) -> Dict:
        """Train original (noisy) model"""
        print("🚀 Training original (noisy) model...")

        try:
            # Create model
            original_model = self.bert_trainer.create_model()

            # Save initial state for consistent initialization
            initial_state = self.bert_trainer.save_model_state(original_model)

            # Train model
            training_history = self.bert_trainer.train_model(
                model=original_model,
                data=data,
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                learning_rate=self.config["learning_rate"],
            )

            # Save training history
            self.bert_trainer.save_training_history(
                training_history, str(self.output_dir / "original_training")
            )

            # Save results
            self.results["original_training"] = {
                "history": training_history,
                "model_path": str(self.output_dir / "original_model.pt"),
                "initial_state_path": str(self.output_dir / "initial_state.pt"),
            }

            # Save model and initial state
            torch.save(original_model.state_dict(), self.output_dir / "original_model.pt")
            torch.save(initial_state, self.output_dir / "initial_state.pt")

            print("✅ Original model training complete")

            return original_model, initial_state, training_history

        except Exception as e:
            error_msg = f"Original model training failed: {e}"
            print(f"❌ {error_msg}")
            self.results["error_log"].append(error_msg)
            raise

    def run_stage_experiment(
        self, 
        stage_num: int, 
        time_window: Tuple[int, Optional[int]], 
        original_model, 
        original_data: Dict,
        initial_state: Dict
    ) -> Dict:
        """
        Run experiment for a specific stage
        
        Parameters:
        -----------
        stage_num : int
            Stage number (1-5)
        time_window : Tuple[int, Optional[int]]
            Time window (start_epoch, end_epoch)
        original_model : torch.nn.Module
            Original trained model
        original_data : Dict
            Original noisy data
        initial_state : Dict
            Initial model state for consistent initialization
            
        Returns:
        --------
        Dict
            Stage experiment results
        """
        start_epoch, end_epoch = time_window
        stage_name = f"Stage_{stage_num}"
        end_desc = "T" if end_epoch is None else str(end_epoch)
        
        print(f"\n🔬 Running {stage_name}: Time window [{start_epoch}, {end_desc}]")
        print("=" * 60)

        try:
            # 1. Create TIM calculator for this stage
            tim_calculator = create_tim_calculator(
                t1=start_epoch,  # start_step
                t2=end_epoch,    # end_step  
                num_epochs=self.config["epochs"],
                batch_size=self.config["tim_batch_size"],
                regularization=self.config["regularization"],
                finite_diff_eps=self.config["finite_diff_eps"],
                random_state=self.config["random_state"],
            )

            # 2. Compute influence scores for this time window
            print(f"📊 Computing TIM influence for [{start_epoch}, {end_desc}]...")
            influence_scores = tim_calculator.compute_influence(original_model, original_data)

            # 3. Analyze influence scores
            noise_indices = self.data_processor.noise_indices
            influence_analysis = tim_calculator.analyze_influence_scores(
                influence_scores, original_data["y_train"], noise_indices
            )

            # 4. Select samples to prune based on influence scores
            prune_indices, keep_indices = tim_calculator.select_bottom_k_samples(
                influence_scores, k_ratio=self.config["prune_ratio"]
            )

            # 5. Prune data
            pruned_data, remaining_indices = self.data_processor.prune_data_by_indices(
                prune_indices
            )

            # 6. Train model on pruned data
            print(f"🚀 Training {stage_name} model on pruned data...")
            pruned_model = self.bert_trainer.create_model()
            pruned_model = self.bert_trainer.load_model_state(pruned_model, initial_state)

            pruned_history = self.bert_trainer.train_model(
                model=pruned_model,
                data=pruned_data,
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                learning_rate=self.config["learning_rate"],
            )

            # 7. Analyze pruning effectiveness
            pruned_noise = np.intersect1d(prune_indices, noise_indices)
            noise_recall = (
                len(pruned_noise) / len(noise_indices) if len(noise_indices) > 0 else 0
            )
            noise_precision = (
                len(pruned_noise) / len(prune_indices) if len(prune_indices) > 0 else 0
            )

            # 8. Compile stage results
            stage_results = {
                "stage_num": stage_num,
                "stage_name": stage_name,
                "time_window": {
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "description": f"[{start_epoch}, {end_desc}]"
                },
                "influence_analysis": {
                    "scores": influence_scores.tolist(),
                    "statistics": influence_analysis,
                    "mean_influence": float(np.mean(influence_scores)),
                    "std_influence": float(np.std(influence_scores)),
                    "min_influence": float(np.min(influence_scores)),
                    "max_influence": float(np.max(influence_scores)),
                },
                "pruning_analysis": {
                    "prune_indices": prune_indices.tolist(),
                    "keep_indices": keep_indices.tolist(),
                    "remaining_indices": remaining_indices.tolist(),
                    "original_samples": len(original_data["y_train"]),
                    "pruned_samples": len(prune_indices),
                    "remaining_samples": len(remaining_indices),
                    "prune_ratio": len(prune_indices) / len(original_data["y_train"]),
                    "noise_detection": {
                        "total_noise": len(noise_indices),
                        "captured_noise": len(pruned_noise),
                        "noise_recall": noise_recall,
                        "noise_precision": noise_precision,
                    },
                },
                "training_results": {
                    "pruned_history": pruned_history,
                    "model_path": str(self.output_dir / f"{stage_name.lower()}_model.pt"),
                },
                "status": "success"
            }

            # Save stage model
            torch.save(pruned_model.state_dict(), self.output_dir / f"{stage_name.lower()}_model.pt")

            # Save stage results
            stage_dir = self.output_dir / stage_name.lower()
            stage_dir.mkdir(exist_ok=True)
            
            tim_calculator.save_influence_results(
                influence_scores, influence_analysis, str(stage_dir / "influence_analysis")
            )
            
            self.bert_trainer.save_training_history(
                pruned_history, str(stage_dir / "training_history")
            )

            print(f"✅ {stage_name} experiment complete")
            print(f"   Influence range: [{np.min(influence_scores):.6f}, {np.max(influence_scores):.6f}]")
            print(f"   Pruned samples: {len(prune_indices)} ({len(prune_indices) / len(original_data['y_train']) * 100:.1f}%)")
            print(f"   Noise recall: {noise_recall:.2%}, Precision: {noise_precision:.2%}")

            return stage_results

        except Exception as e:
            error_msg = f"{stage_name} experiment failed: {e}"
            print(f"❌ {error_msg}")
            self.results["error_log"].append(error_msg)
            
            return {
                "stage_num": stage_num,
                "stage_name": stage_name,
                "time_window": {
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "description": f"[{start_epoch}, {end_desc}]"
                },
                "status": "failed",
                "error": str(e)
            }

    def create_multi_stage_visualizations(self):
        """Create comprehensive visualizations across all stages"""
        if not self.config["save_plots"] or self.visualizer is None:
            print("⏭️  Skipping visualization (save_plots=False)")
            return

        print("🎨 Creating multi-stage visualizations...")

        try:
            successful_stages = [s for s in self.results["stage_results"].values() if s["status"] == "success"]
            original_history = self.results.get("original_training", {}).get("history", {})
            
            if not successful_stages:
                print("⚠️  No successful stages to visualize")
                return

            # 1. Multi-stage influence comparison
            self.visualizer.plot_multi_stage_influence_comparison(
                successful_stages,
                title="TIM Influence Scores Across Time Windows",
                save_name="multi_stage_influence_comparison.png"
            )
            
            # 2. Stage performance comparison
            self.visualizer.plot_stage_performance_comparison(
                successful_stages,
                original_history=original_history,
                title="Performance Comparison Across Time Windows", 
                save_name="stage_performance_comparison.png"
            )
            
            # 3. Training curves comparison
            self.visualizer.plot_training_curves_comparison(
                original_history,
                successful_stages,
                title="Training Loss Curves: Original vs Multi-Stage Pruned Models",
                save_name="training_curves_comparison.png"
            )
            
            # 4. Influence heatmap
            self.visualizer.plot_influence_heatmap(
                successful_stages,
                title="Influence Score Distribution Heatmap Across Time Windows",
                save_name="influence_heatmap.png"
            )
            
            # 5. Time window diagram
            # Calculate total steps for diagram
            steps_per_epoch = (self.config["train_count"] + self.config["batch_size"] - 1) // self.config["batch_size"]
            total_steps = self.config["epochs"] * steps_per_epoch
            
            self.visualizer.plot_time_window_diagram(
                self.time_windows,
                total_steps,
                title="Training Time Windows Division (Steps)",
                save_name="time_window_diagram.png"
            )
            
            # 6. Comprehensive dashboard
            self.visualizer.plot_comprehensive_dashboard(
                original_history,
                successful_stages,
                self.config,
                self.results.get("data_stats", {}),
                title="Multi-Stage TIM Influence Data Pruning Experiment Dashboard",
                save_name="comprehensive_dashboard.png"
            )

            print("✅ Multi-stage visualizations complete")

        except Exception as e:
            error_msg = f"Visualization creation failed: {e}"
            print(f"⚠️  {error_msg}")
            self.results["error_log"].append(error_msg)


    def run_complete_experiment(self) -> Dict:
        """Run complete multi-stage pruning experiment"""
        print("🧪 Starting Multi-Stage TIM Influence Data Pruning Experiment")
        print("=" * 80)
        print(f"⏰ Time windows: {self.time_windows}")
        print("=" * 80)

        self.results["start_time"] = time.time()
        self.results["status"] = "running"

        try:
            # 1. Setup components
            self.setup_components()

            # 2. Prepare data
            noisy_data, noise_indices = self.prepare_data()

            # 3. Train original model
            original_model, initial_state, original_history = self.train_original_model(noisy_data)

            # 4. Run experiments for each stage
            self.results["stage_results"] = {}
            
            for stage_num, time_window in enumerate(self.time_windows, 1):
                stage_results = self.run_stage_experiment(
                    stage_num=stage_num,
                    time_window=time_window,
                    original_model=original_model,
                    original_data=noisy_data,
                    initial_state=initial_state
                )
                
                self.results["stage_results"][f"stage_{stage_num}"] = stage_results

            # 5. Generate comparative analysis
            self.generate_comparative_analysis()

            # 6. Create multi-stage visualizations
            self.create_multi_stage_visualizations()

            # 7. Save experiment results
            self.save_experiment_results()

            self.results["status"] = "success"
            self.results["end_time"] = time.time()

            print("🎉 Multi-stage experiment complete!")
            self.print_experiment_summary()

            return self.results

        except Exception as e:
            self.results["status"] = "failed"
            self.results["end_time"] = time.time()
            error_msg = f"Experiment failed: {e}"
            print(f"💥 {error_msg}")
            self.results["error_log"].append(error_msg)
            traceback.print_exc()
            return self.results

    def generate_comparative_analysis(self):
        """Generate comparative analysis across all stages"""
        print("📈 Generating comparative analysis...")

        try:
            successful_stages = [s for s in self.results["stage_results"].values() if s["status"] == "success"]
            
            if not successful_stages:
                print("⚠️  No successful stages for comparison")
                return

            # Performance comparison
            performance_comparison = {}
            best_stage = None
            best_valid_acc = 0
            
            for stage in successful_stages:
                stage_name = stage["stage_name"]
                final_perf = stage["training_results"]["pruned_history"].get("final_performance", {})
                
                performance_comparison[stage_name] = {
                    "time_window": stage["time_window"]["description"],
                    "train_accuracy": final_perf.get("train_accuracy", 0),
                    "valid_accuracy": final_perf.get("valid_accuracy", 0),
                    "train_loss": final_perf.get("train_loss", 0),
                    "noise_recall": stage["pruning_analysis"]["noise_detection"]["noise_recall"],
                    "noise_precision": stage["pruning_analysis"]["noise_detection"]["noise_precision"],
                    "mean_influence": stage["influence_analysis"]["mean_influence"]
                }
                
                # Track best performing stage
                if final_perf.get("valid_accuracy", 0) > best_valid_acc:
                    best_valid_acc = final_perf.get("valid_accuracy", 0)
                    best_stage = stage_name

            # Stage ranking by different criteria
            stages_by_valid_acc = sorted(successful_stages, 
                                       key=lambda x: x["training_results"]["pruned_history"].get("final_performance", {}).get("valid_accuracy", 0), 
                                       reverse=True)
            
            stages_by_noise_recall = sorted(successful_stages,
                                          key=lambda x: x["pruning_analysis"]["noise_detection"]["noise_recall"],
                                          reverse=True)

            self.results["comparative_analysis"] = {
                "performance_comparison": performance_comparison,
                "best_stage": best_stage,
                "best_validation_accuracy": best_valid_acc,
                "stage_rankings": {
                    "by_validation_accuracy": [s["stage_name"] for s in stages_by_valid_acc],
                    "by_noise_recall": [s["stage_name"] for s in stages_by_noise_recall]
                },
                "summary_statistics": {
                    "total_successful_stages": len(successful_stages),
                    "mean_validation_accuracy": np.mean([s["training_results"]["pruned_history"].get("final_performance", {}).get("valid_accuracy", 0) 
                                                        for s in successful_stages]),
                    "mean_noise_recall": np.mean([s["pruning_analysis"]["noise_detection"]["noise_recall"] 
                                                 for s in successful_stages]),
                    "mean_noise_precision": np.mean([s["pruning_analysis"]["noise_detection"]["noise_precision"] 
                                                    for s in successful_stages])
                }
            }

            print("✅ Comparative analysis complete")

        except Exception as e:
            error_msg = f"Comparative analysis failed: {e}"
            print(f"⚠️  {error_msg}")
            self.results["error_log"].append(error_msg)

    def save_experiment_results(self):
        """Save complete experiment results"""
        print("💾 Saving experiment results...")

        try:
            # Save main results file
            with open(self.output_dir / "multi_stage_experiment_results.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

            # Save configuration file
            with open(self.output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            print(f"✅ Experiment results saved to: {self.output_dir}")

        except Exception as e:
            error_msg = f"Results saving failed: {e}"
            print(f"⚠️  {error_msg}")
            self.results["error_log"].append(error_msg)

    def print_experiment_summary(self):
        """Print experiment summary"""
        print("\n" + "=" * 80)
        print("📊 MULTI-STAGE EXPERIMENT SUMMARY")
        print("=" * 80)

        # Basic information
        print(f"📁 Results Directory: {self.output_dir}")
        print(f"⏱️  Total Runtime: {self.results.get('end_time', 0) - self.results.get('start_time', 0):.1f} seconds")
        print(f"🎯 Experiment Status: {self.results.get('status', 'unknown').upper()}")

        # Data information
        data_stats = self.results.get("data_stats", {})
        if data_stats:
            noise_info = data_stats.get("noise_info", {})
            print(f"\n📊 DATASET INFORMATION:")
            print(f"   Dataset: {self.config['dataset_name'].upper()}")
            print(f"   Training Samples: {noise_info.get('total_samples', 'N/A')}")
            print(f"   Noise Samples: {noise_info.get('noise_count', 'N/A')} ({noise_info.get('noise_rate', 0) * 100:.1f}%)")

        # Time windows
        print(f"\n⏰ TIME WINDOWS:")
        for i, (start, end) in enumerate(self.time_windows, 1):
            end_desc = "T" if end is None else str(end)
            print(f"   Stage {i}: [{start}, {end_desc}]")

        # Stage results
        successful_stages = [s for s in self.results.get("stage_results", {}).values() if s["status"] == "success"]
        failed_stages = [s for s in self.results.get("stage_results", {}).values() if s["status"] == "failed"]

        print(f"\n🚀 STAGE RESULTS:")
        print(f"   Successful Stages: {len(successful_stages)}/{len(self.time_windows)}")
        print(f"   Failed Stages: {len(failed_stages)}")

        if successful_stages:
            print(f"\n🏆 PERFORMANCE BY STAGE:")
            for stage in successful_stages:
                stage_name = stage["stage_name"]
                time_desc = stage["time_window"]["description"]
                final_perf = stage["training_results"]["pruned_history"].get("final_performance", {})
                noise_det = stage["pruning_analysis"]["noise_detection"]

                print(f"\n   {stage_name} - Time Window {time_desc}:")
                print(f"     Training Accuracy: {final_perf.get('train_accuracy', 0):.3f}")
                print(f"     Validation Accuracy: {final_perf.get('valid_accuracy', 0):.3f}")
                print(f"     Noise Recall: {noise_det['noise_recall']:.2%}")
                print(f"     Noise Precision: {noise_det['noise_precision']:.2%}")

        # Best performing stage
        comparative = self.results.get("comparative_analysis", {})
        if comparative:
            print(f"\n🥇 BEST PERFORMING STAGE: {comparative.get('best_stage', 'N/A')}")
            print(f"   Best Validation Accuracy: {comparative.get('best_validation_accuracy', 0):.3f}")
            
            summary_stats = comparative.get("summary_statistics", {})
            print(f"\n📈 OVERALL STATISTICS:")
            print(f"   Mean Validation Accuracy: {summary_stats.get('mean_validation_accuracy', 0):.3f}")
            print(f"   Mean Noise Recall: {summary_stats.get('mean_noise_recall', 0):.2%}")
            print(f"   Mean Noise Precision: {summary_stats.get('mean_noise_precision', 0):.2%}")

        # Error log
        if self.results.get("error_log"):
            print(f"\n⚠️  ERROR LOG:")
            for error in self.results["error_log"]:
                print(f"   • {error}")

        print("=" * 80)


def create_multi_stage_experiment(
    # Data configuration
    dataset_name: str = "imdb",
    train_count: int = 2000,
    valid_count: int = 400,
    test_count: int = 400,
    noise_rate: float = 0.3,
    # Model configuration
    model_name: str = "distilbert-base-uncased",
    # Training configuration
    epochs: int = 10,
    batch_size: int = 32,
    # Multi-stage configuration
    num_stages: int = 5,
    # Experiment configuration
    output_dir: str = "./multi_stage_results",
    random_state: int = 42,
) -> MultiStagePruningExperiment:
    """
    Factory function: Create multi-stage pruning experiment

    Parameters:
    -----------
    dataset_name : str
        Dataset name, default "imdb"
    train_count : int
        Training sample count, default 2000
    valid_count : int
        Validation sample count, default 400
    test_count : int
        Test sample count, default 400
    noise_rate : float
        Noise ratio, default 0.3
    model_name : str
        Model name, default "distilbert-base-uncased"
    epochs : int
        Training epochs, default 10
    batch_size : int
        Batch size, default 32
    num_stages : int
        Number of time window stages, default 5
    output_dir : str
        Output directory, default "./multi_stage_results"
    random_state : int
        Random seed, default 42

    Returns:
    --------
    MultiStagePruningExperiment
        Configured experiment object
    """
    return MultiStagePruningExperiment(
        dataset_name=dataset_name,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        noise_rate=noise_rate,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        num_stages=num_stages,
        output_dir=output_dir,
        random_state=random_state,
    )


def main():
    """Main function - Run complete multi-stage experiment"""
    print("🧪 Multi-Stage TIM Influence Data Pruning Experiment")
    print("=" * 80)

    # Create experiment
    experiment = create_multi_stage_experiment()

    # Run experiment
    results = experiment.run_complete_experiment()

    # Return results
    return results


if __name__ == "__main__":
    success = main()
    if success and success.get("status") == "success":
        print("\n🎉 Multi-stage experiment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Multi-stage experiment failed")
        sys.exit(1)