#!/usr/bin/env python3
"""
快速测试脚本 - 运行小规模噪声剪枝实验

这个脚本用于快速验证整个实验流程是否正常工作，使用小规模数据。
"""

from my_experiments.noise_pruning_experiment import create_experiment

def main():
    """运行快速测试"""
    print("🧪 开始快速测试噪声剪枝实验")
    print("=" * 50)
    
    # 创建小规模实验用于测试
    experiment = create_experiment(
        dataset_name="imdb",
        train_count=50,  # 很小的数据量用于快速测试
        valid_count=20,
        test_count=20,
        noise_rate=0.3,
        epochs=2,  # 少量epoch
        tim_epochs=1,  # 最小TIM训练
        batch_size=8,  # 小批次
        output_dir="./quick_test_results"
    )
    
    # 运行实验
    results = experiment.run_complete_experiment()
    
    # 检查结果
    if results.get('status') == 'success':
        print("\n🎉 快速测试成功完成！")
        print(f"📁 结果保存在: {experiment.output_dir}")
        return True
    else:
        print("\n❌ 快速测试失败")
        print(f"❗ 错误: {results.get('error_log', [])}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)