#!/usr/bin/env python3
"""
快速测试脚本 - 运行小规模噪声剪枝实验

这个脚本用于快速验证整个实验流程是否正常工作，使用小规模数据。
"""

from my_experiments.noise_pruning_experiment import create_experiment

def main():
    """Run quick test"""
    print("🧪 Quick Test - Small Scale Noise Pruning Experiment")
    print("=" * 50)
    
    # Create small-scale experiment for testing
    experiment = create_experiment(
        dataset_name="imdb",
        train_count=50,  # Very small data for quick testing
        valid_count=20,
        test_count=20,
        noise_rate=0.3,
        epochs=2,  # Few epochs
        tim_epochs=1,  # Minimal TIM training
        batch_size=8,  # Small batch size
        output_dir="./quick_test_results"
    )
    
    # Run experiment
    results = experiment.run_complete_experiment()
    
    # Check results
    if results.get('status') == 'success':
        print("\n🎉 Quick test completed successfully!")
        print(f"📁 Results saved to: {experiment.output_dir}")
        return True
    else:
        print("\n❌ Quick test failed")
        print(f"❗ Errors: {results.get('error_log', [])}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)