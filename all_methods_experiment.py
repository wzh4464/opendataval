#!/usr/bin/env python3
"""
BERT情感分析实验 - 运行所有数据评估方法（除TIM外）

使用OpenDataVal的所有数据评估方法对BERT情感分析进行数据价值评估实验。
实验配置：训练集2048，测试集256，种子42，30%标签翻转
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from opendataval.dataloader import DataFetcher
from opendataval.model import BertClassifier
from opendataval.dataval import (
    LeaveOneOut,
    RandomEvaluator,
    LossValEvaluator
)


def create_bert_model(num_classes=2, device="auto"):
    """创建BERT分类器"""
    return BertClassifier(
        pretrained_model_name="distilbert-base-uncased",
        num_classes=num_classes,
        dropout_rate=0.1,
        num_train_layers=2
    )


def prepare_data(train_count=2048, test_count=256, noise_rate=0.3, random_state=42):
    """准备数据集"""
    print("🔄 准备数据集...")
    
    # 加载IMDB数据集 - 使用链式调用方式
    from opendataval.dataloader import mix_labels
    
    data_fetcher = (
        DataFetcher("imdb", cache_dir="../data_files/", force_download=False)
        .split_dataset_by_count(train_count, test_count, test_count)
    )
    
    print(f"📊 数据规模: 训练={train_count}, 验证={test_count}, 测试={test_count}")
    
    # 注入标签噪声
    if noise_rate > 0:
        print(f"🔀 注入标签噪声: {noise_rate*100:.1f}%")
        data_fetcher = data_fetcher.noisify(mix_labels, noise_rate=noise_rate)
    
    return data_fetcher


def get_evaluator_configs():
    """获取快速数据评估方法配置（除TIM和Shapley外）"""
    evaluators_config = {
        # 简单且快速的方法
        "RandomEvaluator": {
            "class": RandomEvaluator,
            "kwargs": {},
            "description": "Random baseline evaluator"
        },
        
        "LeaveOneOut": {
            "class": LeaveOneOut,
            "kwargs": {},
            "description": "Leave-One-Out evaluation"
        },
        
        "LossValEvaluator": {
            "class": LossValEvaluator,
            "kwargs": {
                "is_classification": True,  # 添加必需的参数
            },
            "description": "Loss Value evaluator"
        },
    }
    
    return evaluators_config


def run_evaluator(evaluator_name, evaluator_config, data_fetcher, model, metric="accuracy"):
    """运行单个评估器"""
    print(f"\n🧪 运行评估器: {evaluator_name}")
    print(f"   描述: {evaluator_config['description']}")
    
    try:
        # 创建评估器实例
        evaluator_class = evaluator_config["class"]
        evaluator_kwargs = evaluator_config.get("kwargs", {})
        
        evaluator = evaluator_class(**evaluator_kwargs)
        
        # 开始计时
        start_time = time.time()
        
        # 训练评估器
        print(f"   ⚡ 开始训练...")
        # 修正metric参数 - 使用函数而不是字符串
        from sklearn.metrics import accuracy_score
        eval_metric = accuracy_score if metric == "accuracy" else metric
        evaluator_instance = evaluator.train(data_fetcher, model, metric=eval_metric)
        
        # 评估数据值
        print(f"   📊 计算数据值...")
        data_values = evaluator_instance.evaluate_data_values()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 统计结果
        mean_value = np.mean(data_values)
        std_value = np.std(data_values)
        min_value = np.min(data_values)
        max_value = np.max(data_values)
        
        result = {
            "evaluator": evaluator_name,
            "description": evaluator_config['description'],
            "status": "success",
            "duration_seconds": duration,
            "data_values_stats": {
                "mean": float(mean_value),
                "std": float(std_value),
                "min": float(min_value),
                "max": float(max_value),
                "shape": data_values.shape
            },
            "data_values": data_values.tolist(),  # 保存完整的数值
        }
        
        print(f"   ✅ 完成! 耗时: {duration:.1f}秒")
        print(f"   📈 数据值统计: 均值={mean_value:.4f}, 标准差={std_value:.4f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ 失败: {str(e)}")
        print(f"   🔍 错误详情: {traceback.format_exc()}")
        
        return {
            "evaluator": evaluator_name,
            "description": evaluator_config['description'],
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def select_high_value_samples(data_values, select_ratio=0.9):
    """选择高价值样本（90%）"""
    n_samples = len(data_values)
    n_select = int(n_samples * select_ratio)
    
    # 获取数据值的排序索引（降序）
    sorted_indices = np.argsort(data_values)[::-1]
    
    # 选择前90%的高价值样本
    selected_indices = sorted_indices[:n_select]
    
    return selected_indices


def retrain_with_selected_data(data_fetcher, selected_indices, model_class, metric="accuracy"):
    """使用选中的数据重新训练模型"""
    # 创建新的数据获取器，只包含选中的样本
    # 注意：这里需要根据实际的DataFetcher API进行调整
    
    # 获取训练数据
    x_train, y_train = data_fetcher.x_train, data_fetcher.y_train
    x_valid, y_valid = data_fetcher.x_valid, data_fetcher.y_valid
    
    # 选择高价值样本
    x_train_selected = x_train[selected_indices]
    y_train_selected = y_train[selected_indices]
    
    # 创建新模型
    model = model_class()
    
    # 训练模型（这里需要根据实际的BERT模型训练API调整）
    # 简化版本，实际需要更完整的训练循环
    model.fit(x_train_selected, y_train_selected, 
              validation_data=(x_valid, y_valid),
              epochs=3,  # 减少训练轮数以加快速度
              batch_size=32)
    
    # 评估性能
    train_score = model.score(x_train_selected, y_train_selected)
    valid_score = model.score(x_valid, y_valid)
    
    return {
        "train_score": train_score,
        "valid_score": valid_score,
        "selected_samples": len(selected_indices),
        "original_samples": len(x_train)
    }


def create_comparison_plots(results, output_dir="./results"):
    """Create comparison plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter successful results
    successful_results = [r for r in results if r["status"] == "success"]
    
    if not successful_results:
        print("⚠️  No successful results for plotting")
        return
    
    # 1. Runtime comparison plot
    plt.figure(figsize=(12, 6))
    evaluator_names = [r["evaluator"] for r in successful_results]
    durations = [r["duration_seconds"] for r in successful_results]
    
    plt.barh(evaluator_names, durations)
    plt.xlabel("Runtime (seconds)")
    plt.title("Runtime Comparison of Data Evaluation Methods")
    plt.tight_layout()
    plt.savefig(output_path / "runtime_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Data values distribution plot
    plt.figure(figsize=(15, 10))
    n_methods = len(successful_results)
    cols = 3
    rows = (n_methods + cols - 1) // cols
    
    for i, result in enumerate(successful_results):
        plt.subplot(rows, cols, i + 1)
        data_values = np.array(result["data_values"])
        plt.hist(data_values, bins=30, alpha=0.7)
        plt.title(f'{result["evaluator"]}\n'
                 f'Mean: {result["data_values_stats"]["mean"]:.3f}')
        plt.xlabel("Data Values")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(output_path / "data_values_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Data values statistics comparison plot
    plt.figure(figsize=(12, 8))
    means = [r["data_values_stats"]["mean"] for r in successful_results]
    stds = [r["data_values_stats"]["std"] for r in successful_results]
    
    x_pos = np.arange(len(evaluator_names))
    plt.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5)
    plt.xticks(x_pos, evaluator_names, rotation=45, ha='right')
    plt.ylabel("Data Values")
    plt.title("Data Values Statistics Comparison (Mean ± Std)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "data_values_stats_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to: {output_path}")


def main():
    """主函数"""
    print("🚀 BERT情感分析数据评估实验 - 所有方法（除TIM外）")
    print("=" * 80)
    
    # 实验配置
    config = {
        "dataset": "imdb",
        "train_count": 10,
        "test_count": 10,
        "noise_rate": 0.3,
        "random_state": 42,
        "metric": "accuracy",
        "output_dir": "./all_methods_results_fixed"
    }
    
    print(f"📋 实验配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 准备数据
    print(f"\n🔧 准备实验数据...")
    data_fetcher = prepare_data(
        train_count=config["train_count"],
        test_count=config["test_count"],
        noise_rate=config["noise_rate"],
        random_state=config["random_state"]
    )
    
    # 创建模型工厂函数
    def model_factory():
        return create_bert_model(num_classes=2)
    
    # 获取评估器配置
    evaluators_config = get_evaluator_configs()
    
    print(f"\n🧪 将运行 {len(evaluators_config)} 个数据评估方法:")
    for name, config_item in evaluators_config.items():
        print(f"   • {name}: {config_item['description']}")
    
    # 运行实验
    results = []
    total_start_time = time.time()
    
    for evaluator_name, evaluator_config in evaluators_config.items():
        try:
            # 为每个评估器创建新的模型实例
            model = model_factory()
            
            # 运行评估器
            result = run_evaluator(
                evaluator_name, 
                evaluator_config, 
                data_fetcher, 
                model, 
                metric=config["metric"]
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ 评估器 {evaluator_name} 运行失败: {e}")
            results.append({
                "evaluator": evaluator_name,
                "status": "failed",
                "error": str(e)
            })
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 保存结果
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # 保存完整结果
    with open(output_dir / "all_methods_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "results": results,
            "total_duration_seconds": total_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)
    
    # 生成对比图表
    create_comparison_plots(results, output_dir)
    
    # 打印总结
    print(f"\n🎉 实验完成!")
    print("=" * 80)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"⏱️  总运行时间: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
    print(f"✅ 成功的方法: {len(successful)}/{len(results)}")
    print(f"❌ 失败的方法: {len(failed)}")
    
    if successful:
        print(f"\n📊 成功方法结果:")
        for result in successful:
            stats = result["data_values_stats"]
            print(f"   • {result['evaluator']}: "
                  f"均值={stats['mean']:.4f}, "
                  f"用时={result['duration_seconds']:.1f}s")
    
    if failed:
        print(f"\n⚠️  失败方法:")
        for result in failed:
            print(f"   • {result['evaluator']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 结果保存位置: {output_dir}")
    print(f"   • all_methods_results.json - 完整结果数据")
    print(f"   • runtime_comparison.png - 运行时间对比")
    print(f"   • data_values_distribution.png - 数据值分布")
    print(f"   • data_values_stats_comparison.png - 数据值统计对比")
    
    return len(successful) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)