#!/usr/bin/env python3
"""
Quick runner for multi-stage TIM influence data pruning experiment

This script runs the complete 5-stage experiment with optimized settings for 
demonstration and testing purposes.
"""

import sys
import time
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from my_experiments.multi_stage_pruning_experiment import create_multi_stage_experiment


def main():
    """Run multi-stage experiment with configurable settings"""
    
    parser = argparse.ArgumentParser(description="Run multi-stage TIM experiment")
    parser.add_argument("--n_tr", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=200, help="Number of validation/test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("🚀 Multi-Stage TIM Influence Data Pruning Experiment")
    print("=" * 80)
    print("This experiment divides training into 5 time windows based on TRAINING STEPS:")
    print("Stage 1: [0, t1] - Early training steps")  
    print("Stage 2: [t1, t2] - Initial convergence steps")
    print("Stage 3: [t2, t3] - Mid training steps")
    print("Stage 4: [t3, t4] - Late training steps") 
    print("Stage 5: [t4, T] - Final convergence steps")
    print("\nIMPORTANT: Time windows are based on training STEPS, not epochs!")
    print("Each step corresponds to one batch gradient update.")
    print("\nFor each stage, we:")
    print("1. Compute TIM influence scores for that time window")
    print("2. Prune low-influence samples")
    print("3. Retrain on cleaned data")
    print("4. Compare performance across all stages")
    print("=" * 80)
    
    # Create experiment with provided settings
    experiment = create_multi_stage_experiment(
        # Data configuration
        dataset_name="imdb",
        train_count=args.n_tr,
        valid_count=args.n_val,
        test_count=args.n_val,
        noise_rate=0.3,      # 30% label noise
        
        # Model configuration
        pretrained_model_name="distilbert-base-uncased",
        
        # Training configuration
        epochs=10,           # 10 epochs total, 2 epochs per stage
        batch_size=16,
        learning_rate=2e-5,
        
        # Multi-stage configuration  
        num_stages=5,        # 5 time windows
        
        # Pruning & relabel configuration
        prune_ratio=0.1,
        relabel_ratio=0.3,
        
        # Experiment configuration
        output_dir="./multi_stage_demo_results",
        random_state=args.seed,
    )
    
    print(f"\n⚙️  EXPERIMENT CONFIGURATION:")
    print(f"   Dataset: {experiment.config['dataset_name'].upper()}")
    print(f"   Training samples: {experiment.config['train_count']:,}")
    print(f"   Noise rate: {experiment.config['noise_rate']*100:.1f}%")
    print(f"   Total epochs: {experiment.config['epochs']}")
    print(f"   Time windows: {experiment.config['num_stages']}")
    print(f"   Pruning ratio: {experiment.config['prune_ratio']*100:.1f}%")
    print(f"   Output directory: {experiment.config['output_dir']}")
    
    print(f"\n⏰ TIME WINDOW BREAKDOWN (STEPS):")
    # Calculate steps per epoch for display
    batch_size = experiment.config['batch_size']
    train_count = experiment.config['train_count'] 
    steps_per_epoch = (train_count + batch_size - 1) // batch_size
    
    for i, (start, end) in enumerate(experiment.time_windows, 1):
        end_desc = "T" if end is None else str(end)
        if end is None:
            total_steps = experiment.config['epochs'] * steps_per_epoch
            steps_in_window = total_steps - start
        else:
            steps_in_window = end - start + 1
        epochs_equiv = steps_in_window / steps_per_epoch
        print(f"   Stage {i}: [{start}, {end_desc}] - {steps_in_window} steps (~{epochs_equiv:.1f} epochs)")
    
    # Confirm start
    print(f"\n🔍 EXPERIMENT OVERVIEW:")
    print("This multi-stage experiment will:")
    print("• Train an original model on noisy data")
    print("• For each of 5 time windows, compute TIM influence scores")
    print("• Prune low-influence samples (likely noise) from each stage")
    print("• Retrain models on cleaned data")
    print("• Generate comprehensive visualizations comparing all stages")
    print("• Create performance analysis across different time periods")
    
    print(f"\n📊 EXPECTED OUTPUTS:")
    print("• Multi-stage influence score comparisons")
    print("• Performance metrics across time windows")
    print("• Noise detection effectiveness analysis")
    print("• Training curves for original vs pruned models") 
    print("• Comprehensive experiment dashboard")
    print("• Time window division diagram")
    
    try:
        # Start experiment
        start_time = time.time()
        
        print(f"\n🧪 Starting experiment...")
        print("This may take several minutes depending on your hardware.")
        print("Progress will be shown for each stage.\n")
        
        # Run the complete experiment
        results = experiment.run_complete_experiment()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print final summary
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print("=" * 80)
        print(f"⏱️  Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📁 Results saved to: {experiment.output_dir}")
        print(f"🎨 Plots saved to: {experiment.output_dir / 'plots'}")
        print(f"🎯 Status: {results.get('status', 'unknown').upper()}")
        
        # Show key results if successful
        if results.get("status") == "success":
            successful_stages = [s for s in results.get("stage_results", {}).values() 
                               if s["status"] == "success"]
            
            print(f"\n📈 KEY RESULTS:")
            print(f"   Successful stages: {len(successful_stages)}/5")
            
            if successful_stages:
                # Find best performing stage
                best_valid_acc = 0
                best_stage = None
                
                for stage in successful_stages:
                    final_perf = stage["training_results"]["pruned_history"].get("final_performance", {})
                    valid_acc = final_perf.get("valid_accuracy", 0)
                    
                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_stage = stage["stage_name"]
                        best_time_window = stage["time_window"]["description"]
                
                if best_stage:
                    print(f"   Best performing stage: {best_stage} {best_time_window}")
                    print(f"   Best validation accuracy: {best_valid_acc:.3f}")
                    
                # Show original vs best improvement
                orig_perf = results.get("original_training", {}).get("history", {}).get("final_performance", {})
                orig_valid_acc = orig_perf.get("valid_accuracy", 0)
                
                if orig_valid_acc > 0:
                    improvement = best_valid_acc - orig_valid_acc
                    print(f"   Original model accuracy: {orig_valid_acc:.3f}")
                    print(f"   Improvement: {improvement:+.3f}")
        
        print(f"\n📋 GENERATED FILES:")
        print(f"   • multi_stage_experiment_results.json - Complete results")
        print(f"   • experiment_config.json - Configuration used")
        print(f"   • plots/multi_stage_influence_comparison.png")
        print(f"   • plots/stage_performance_comparison.png")
        print(f"   • plots/training_curves_comparison.png")
        print(f"   • plots/influence_heatmap.png")
        print(f"   • plots/time_window_diagram.png")
        print(f"   • plots/comprehensive_dashboard.png")
        
        print(f"\n✅ Multi-stage experiment demonstration complete!")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)