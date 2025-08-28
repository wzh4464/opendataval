#!/usr/bin/env python3
"""
统一的BERT情感分析和数据估值实验 (含TIM方法)
支持多种设备（CPU/CUDA/MPS），命令行参数配置
使用预训练DistilBERT模型进行微调，包含新的TIM方法
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opendataval.dataval import AME, DataOob, RandomEvaluator, InfluenceFunction, TimInfluence
from opendataval.experiment import ExperimentMediator


def get_device_config(device: str = "auto") -> dict:
    """获取设备相关配置"""
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    # 设备特定优化
    if device == "mps":
        # Apple Silicon MPS优化
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        torch.set_num_threads(8)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🍎 Apple Silicon MPS优化已启用")
        
    elif device == "cuda":
        # CUDA优化
        torch.backends.cudnn.benchmark = True
        print(f"🚀 CUDA优化已启用 (设备数量: {torch.cuda.device_count()})")
        
    else:
        # CPU优化
        torch.set_num_threads(os.cpu_count() or 4)
        print("💻 CPU模式已启用")
    
    return {"device": device}


def get_experiment_config(args) -> dict:
    """根据参数获取实验配置"""
    
    # 根据设备调整默认配置
    if args.device == "mps":
        default_batch_size = 8
        default_num_models = 15
        default_epochs = 3
    elif args.device == "cuda":
        default_batch_size = 16
        default_num_models = 25
        default_epochs = 3
    else:
        default_batch_size = 4
        default_num_models = 10
        default_epochs = 2
    
    config = {
        'dataset_name': args.dataset,
        'train_count': args.train_samples,
        'valid_count': args.valid_samples,
        'test_count': args.test_samples,
        'model_name': 'BertClassifier',
        'train_kwargs': {
            'epochs': args.epochs or default_epochs,
            'batch_size': args.batch_size or default_batch_size,
            'lr': args.learning_rate,
        },
        'evaluator_config': {
            'num_models': args.num_models or default_num_models,
        },
        'device': args.device,
        'tim_config': {
            'num_epochs': args.tim_epochs,
            'regularization': args.tim_reg,
            'window_type': args.tim_window_type,
            'start_step': args.tim_start_step,
            'end_step': args.tim_end_step,
        }
    }
    
    return config


def create_evaluators(config: dict, methods: List[str]) -> List:
    """创建数据估值方法评估器，包括TIM"""
    evaluators = []
    num_models = config['evaluator_config']['num_models']
    
    for method in methods:
        if method == "random":
            evaluators.append(RandomEvaluator())
        elif method == "dataoob":
            evaluators.append(DataOob(num_models=num_models))
        elif method == "ame":
            evaluators.append(AME(num_models=num_models))
        elif method == "influence":
            # 影响函数在某些设备上可能不稳定
            if config['device'] != 'mps':
                evaluators.append(InfluenceFunction())
            else:
                print("⚠️ 跳过影响函数方法（MPS设备不稳定）")
        elif method == "tim":
            # 新的TIM方法 - 支持任意时间区间
            tim_kwargs = {
                'time_window_type': config['tim_config']['window_type'],
                'num_epochs': config['tim_config']['num_epochs'],
                'regularization': config['tim_config']['regularization']
            }
            if config['tim_config']['start_step'] is not None:
                tim_kwargs['start_step'] = config['tim_config']['start_step']
            if config['tim_config']['end_step'] is not None:
                tim_kwargs['end_step'] = config['tim_config']['end_step']
                
            evaluators.append(TimInfluence(**tim_kwargs))
        else:
            print(f"⚠️ 未知的评估方法: {method}")
    
    return evaluators


def run_experiment(args) -> dict:
    """运行完整的BERT情感分析实验"""
    
    print("=" * 60)
    print("🤖 BERT情感分析与数据估值实验 (含TIM方法)")
    print(f"📅 实验时间: {datetime.now()}")
    print("=" * 60)
    
    # 1. 设备配置
    device_config = get_device_config(args.device)
    actual_device = device_config['device']
    
    # 2. 实验配置
    config = get_experiment_config(args)
    config['device'] = actual_device  # 使用实际检测到的设备
    
    print(f"\n📋 实验配置:")
    print(f"   🖥️  设备: {actual_device.upper()}")
    print(f"   📚 数据集: {config['dataset_name']}")
    print(f"   🏷️  模型: {config['model_name']} (微调DistilBERT)")
    print(f"   🔢 训练样本: {config['train_count']}")
    print(f"   📦 批次大小: {config['train_kwargs']['batch_size']}")
    print(f"   🔄 训练轮次: {config['train_kwargs']['epochs']}")
    print(f"   📈 学习率: {config['train_kwargs']['lr']}")
    print(f"   🎯 评估方法: {', '.join(args.methods)}")
    if "tim" in args.methods:
        print(f"   ⏰ TIM回溯轮次: {config['tim_config']['num_epochs']}")
        print(f"   🔧 TIM正则化: {config['tim_config']['regularization']}")
    
    # 3. 设置实验环境
    print(f"\n🔧 设置实验环境...")
    try:
        exper_med = ExperimentMediator.model_factory_setup(
            dataset_name=config['dataset_name'],
            train_count=config['train_count'],
            valid_count=config['valid_count'],
            test_count=config['test_count'],
            model_name=config['model_name'],
            train_kwargs=config['train_kwargs'],
            metric_name='accuracy',
            device=actual_device,
        )
        print(f"✅ 实验环境设置成功")
        # 获取基线准确率（如果可用）
        try:
            baseline_accuracy = getattr(exper_med, 'model_metric', 0.0)
        except:
            baseline_accuracy = 0.0
        print(f"📊 基线模型准确率: {baseline_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ 实验环境设置失败: {e}")
        return None
    
    # 4. 创建评估器
    print(f"\n🧮 初始化数据估值方法...")
    evaluators = create_evaluators(config, args.methods)
    
    if not evaluators:
        print("❌ 没有有效的评估方法")
        return None
    
    print(f"✅ 创建了 {len(evaluators)} 个评估器")
    for evaluator in evaluators:
        eval_name = type(evaluator).__name__
        if eval_name == "TimInfluence":
            print(f"   🆕 TIM (Time-varying Influence): {evaluator.num_epochs} epochs")
        else:
            print(f"   📊 {eval_name}")
    
    # 5. 计算数据估值
    print(f"\n⏳ 开始计算数据估值...")
    start_time = datetime.now()
    
    try:
        eval_med = exper_med.compute_data_values(evaluators)
        elapsed_time = datetime.now() - start_time
        print(f"✅ 数据估值完成，总耗时: {elapsed_time}")
        
    except Exception as e:
        print(f"❌ 数据估值失败: {e}")
        return None
    
    # 6. 分析结果
    print(f"\n📊 分析结果...")
    results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'device': actual_device,
            'baseline_accuracy': float(baseline_accuracy),
            'elapsed_time': str(elapsed_time),
            'tim_enabled': "tim" in args.methods,
        },
        'config': config,
        'evaluators': {}
    }
    
    # 保存数据估值结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "my_experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    for evaluator in eval_med.data_evaluators:
        method_name = type(evaluator).__name__
        data_values = evaluator.data_values
        
        # 统计信息
        stats = {
            'mean': float(data_values.mean()),
            'std': float(data_values.std()),
            'min': float(data_values.min()),
            'max': float(data_values.max()),
            'median': float(np.median(data_values)),
            'shape': list(data_values.shape)
        }
        
        results['evaluators'][method_name] = stats
        
        # 保存数值数组
        values_file = f"{results_dir}/data_values_{method_name}_{timestamp}.npy"
        np.save(values_file, data_values)
        
        print(f"\n📈 {method_name}:")
        print(f"   均值: {stats['mean']:.6f}")
        print(f"   标准差: {stats['std']:.6f}")
        print(f"   范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"   中位数: {stats['median']:.6f}")
        
        if method_name == "TimInfluence":
            print(f"   🔍 TIM特征: 基于最后{evaluator.num_epochs}轮训练的影响")
    
    # 7. 保存配置和结果
    config_file = f"{results_dir}/config_{timestamp}.json"
    results_file = f"{results_dir}/results_{timestamp}.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果文件:")
    print(f"   📄 配置: {config_file}")
    print(f"   📄 结果: {results_file}")
    print(f"   📄 数据估值: {results_dir}/data_values_*_{timestamp}.npy")
    
    # 8. 实验总结
    print(f"\n🎯 实验总结:")
    print(f"   🖥️  设备: {actual_device.upper()}")
    print(f"   🤖 模型: DistilBERT微调")
    print(f"   📊 基线准确率: {baseline_accuracy:.4f}")
    print(f"   ⏱️  总耗时: {elapsed_time}")
    print(f"   📈 评估方法: {len(evaluators)}种")
    print(f"   🔢 训练样本: {config['train_count']}")
    
    if "tim" in args.methods:
        print(f"   🆕 TIM方法已启用: 时间窗口为最后{config['tim_config']['num_epochs']}轮")
    
    if actual_device == "mps":
        print(f"\n🍎 在Apple Silicon上成功运行BERT微调！")
    elif actual_device == "cuda":
        print(f"\n🚀 在GPU上成功运行BERT微调！")
    else:
        print(f"\n💻 在CPU上成功运行BERT微调！")
    
    return results


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="BERT情感分析与数据估值实验 (含TIM方法)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础配置
    parser.add_argument(
        "--device", 
        choices=["auto", "cpu", "cuda", "mps"], 
        default="auto",
        help="计算设备 (auto=自动检测最佳设备)"
    )
    
    parser.add_argument(
        "--dataset",
        default="imdb",
        help="数据集名称"
    )
    
    # 数据配置
    parser.add_argument(
        "--train-samples",
        type=int,
        default=200,
        help="训练样本数量"
    )
    
    parser.add_argument(
        "--valid-samples", 
        type=int,
        default=100,
        help="验证样本数量"
    )
    
    parser.add_argument(
        "--test-samples",
        type=int, 
        default=100,
        help="测试样本数量"
    )
    
    # 训练配置
    parser.add_argument(
        "--epochs",
        type=int,
        help="训练轮次 (默认根据设备自适应)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批次大小 (默认根据设备自适应)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="学习率 (BERT微调推荐值)"
    )
    
    # 评估配置  
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["random", "dataoob", "ame", "influence", "tim"],
        default=["random", "dataoob", "tim"],
        help="数据估值方法 (包含新的TIM方法)"
    )
    
    parser.add_argument(
        "--num-models",
        type=int,
        help="评估器使用的模型数量 (默认根据设备自适应)"
    )
    
    # TIM特定配置
    parser.add_argument(
        "--tim-epochs",
        type=int,
        default=3,
        help="TIM方法回溯的epoch数量"
    )
    
    parser.add_argument(
        "--tim-reg",
        type=float,
        default=0.01,
        help="TIM方法的L2正则化参数"
    )
    
    parser.add_argument(
        "--tim-window-type",
        choices=["last_epochs", "custom_range", "full"],
        default="last_epochs",
        help="TIM时间窗口类型"
    )
    
    parser.add_argument(
        "--tim-start-step",
        type=int,
        help="TIM自定义时间窗口起始步骤"
    )
    
    parser.add_argument(
        "--tim-end-step", 
        type=int,
        help="TIM自定义时间窗口结束步骤"
    )
    
    # 输出配置
    parser.add_argument(
        "--output-dir",
        default="my_experiments/results",
        help="输出目录"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="详细输出"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        print(f"🔍 命令行参数: {vars(args)}")
        
        results = run_experiment(args)
        
        if results:
            print(f"\n✅ 实验成功完成！")
            if "tim" in args.methods:
                print(f"🎉 TIM (Time-varying Influence Measurement) 方法测试完成")
            print(f"🎉 BERT微调情感分析实验结束")
        else:
            print(f"\n❌ 实验失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 实验被用户中断")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n💥 实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)