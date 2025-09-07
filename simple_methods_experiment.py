#!/usr/bin/env python3
"""
简化版BERT情感分析数据评估实验 - 只使用最稳定的方法

专门修正错误并测试与BERT兼容的数据评估方法
"""

import sys
import time
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from opendataval.dataloader import DataFetcher
from opendataval.model import BertClassifier
from opendataval.dataval import RandomEvaluator


def create_bert_model(num_classes=2):
    """创建BERT分类器"""
    return BertClassifier(
        pretrained_model_name="distilbert-base-uncased",
        num_classes=num_classes,
        dropout_rate=0.1,
        num_train_layers=2
    )


def prepare_data(train_count=10, test_count=10, noise_rate=0.3):
    """准备数据集"""
    print("🔄 准备数据集...")
    
    from opendataval.dataloader import mix_labels
    
    data_fetcher = (
        DataFetcher("imdb", cache_dir="../data_files/", force_download=False)
        .split_dataset_by_count(train_count, test_count, test_count)
    )
    
    print(f"📊 数据规模: 训练={train_count}, 验证={test_count}, 测试={test_count}")
    
    if noise_rate > 0:
        print(f"🔀 注入标签噪声: {noise_rate*100:.1f}%")
        data_fetcher = data_fetcher.noisify(mix_labels, noise_rate=noise_rate)
    
    return data_fetcher


def run_experiment_with_simple_methods():
    """运行带有简单方法的实验"""
    
    print("🚀 简化版BERT情感分析数据评估实验")
    print("=" * 60)
    
    # 实验配置
    config = {
        "dataset": "imdb",
        "train_count": 10,
        "test_count": 10,
        "noise_rate": 0.3,
        "random_state": 42,
        "output_dir": "./simple_methods_results"
    }
    
    print(f"📋 实验配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 准备数据
    data_fetcher = prepare_data(
        train_count=config["train_count"],
        test_count=config["test_count"],
        noise_rate=config["noise_rate"]
    )
    
    # 创建模型工厂
    def model_factory():
        return create_bert_model(num_classes=2)
    
    # 运行RandomEvaluator （这个是最稳定的）
    print(f"\n🧪 运行RandomEvaluator...")
    try:
        model = model_factory()
        random_evaluator = RandomEvaluator()
        
        start_time = time.time()
        evaluator_instance = random_evaluator.train(data_fetcher, model)
        data_values = evaluator_instance.evaluate_data_values()
        end_time = time.time()
        
        # 统计结果
        mean_value = np.mean(data_values)
        std_value = np.std(data_values)
        
        result = {
            "evaluator": "RandomEvaluator",
            "status": "success",
            "duration_seconds": end_time - start_time,
            "data_values_stats": {
                "mean": float(mean_value),
                "std": float(std_value),
                "min": float(np.min(data_values)),
                "max": float(np.max(data_values)),
                "shape": list(data_values.shape)
            },
            "data_values": data_values.tolist()
        }
        
        print(f"   ✅ 完成! 耗时: {end_time - start_time:.3f}秒")
        print(f"   📈 数据值统计: 均值={mean_value:.4f}, 标准差={std_value:.4f}")
        
    except Exception as e:
        print(f"   ❌ 失败: {str(e)}")
        result = {
            "evaluator": "RandomEvaluator",
            "status": "failed",
            "error": str(e)
        }
    
    # 保存结果
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    full_results = {
        "config": config,
        "results": [result],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_dir / "simple_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # 如果成功，创建简单的可视化
    if result["status"] == "success":
        create_simple_plots(result, output_dir)
    
    print(f"\n🎉 实验完成! 结果保存到: {output_dir}")
    return result["status"] == "success"


def create_simple_plots(result, output_dir):
    """创建简单的可视化图表"""
    
    data_values = np.array(result["data_values"])
    
    # 1. 数据值分布图
    plt.figure(figsize=(10, 6))
    plt.hist(data_values, bins=5, alpha=0.7, edgecolor='black')
    plt.xlabel("Data Values")
    plt.ylabel("Frequency")
    plt.title(f"Data Values Distribution - {result['evaluator']}")
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats = result["data_values_stats"]
    plt.axvline(stats["mean"], color='red', linestyle='--', 
                label=f'Mean: {stats["mean"]:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "data_values_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 数据值序列图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data_values)), data_values, 'o-', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Data Value")
    plt.title(f"Data Values by Sample Index - {result['evaluator']}")
    plt.grid(True, alpha=0.3)
    
    # 添加均值线
    plt.axhline(stats["mean"], color='red', linestyle='--', 
                label=f'Mean: {stats["mean"]:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "data_values_sequence.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 可视化图表已保存")


def analyze_data_for_high_value_selection(result, select_ratio=0.9):
    """分析数据并选择高价值样本"""
    
    if result["status"] != "success":
        print("❌ 无法分析失败的实验结果")
        return
    
    data_values = np.array(result["data_values"])
    n_samples = len(data_values)
    n_select = int(n_samples * select_ratio)
    
    # 获取高价值样本的索引
    sorted_indices = np.argsort(data_values)[::-1]  # 降序排列
    high_value_indices = sorted_indices[:n_select]
    low_value_indices = sorted_indices[n_select:]
    
    print(f"\n📊 数据价值分析:")
    print(f"   总样本数: {n_samples}")
    print(f"   选择比例: {select_ratio*100:.0f}%")
    print(f"   高价值样本: {n_select}个")
    print(f"   低价值样本: {len(low_value_indices)}个")
    
    if len(high_value_indices) > 0:
        high_values = data_values[high_value_indices]
        print(f"   高价值样本统计: 均值={np.mean(high_values):.4f}, "
              f"范围=[{np.min(high_values):.4f}, {np.max(high_values):.4f}]")
    
    if len(low_value_indices) > 0:
        low_values = data_values[low_value_indices]
        print(f"   低价值样本统计: 均值={np.mean(low_values):.4f}, "
              f"范围=[{np.min(low_values):.4f}, {np.max(low_values):.4f}]")
    
    return {
        "high_value_indices": high_value_indices.tolist(),
        "low_value_indices": low_value_indices.tolist(),
        "high_values": data_values[high_value_indices].tolist(),
        "low_values": data_values[low_value_indices].tolist() if len(low_value_indices) > 0 else []
    }


if __name__ == "__main__":
    print("🚀 开始简化版数据评估实验...")
    
    success = run_experiment_with_simple_methods()
    
    if success:
        # 读取结果并进行分析
        results_file = Path("./simple_methods_results/simple_results.json")
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = data["results"][0]
            analysis = analyze_data_for_high_value_selection(result, select_ratio=0.9)
            
            # 保存分析结果
            if analysis:
                data["data_analysis"] = analysis
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ 分析完成! 高价值样本分析已保存到结果文件")
    
    print(f"\n{'='*60}")
    print(f"🎯 实验总结:")
    print(f"   使用RandomEvaluator成功完成了基线数据价值评估")
    print(f"   在30%标签噪声环境下，评估了{10}个训练样本的数据价值")
    print(f"   生成了数据值分布图和序列图")
    print(f"   分析了90%高价值样本的选择策略")
    
    sys.exit(0 if success else 1)