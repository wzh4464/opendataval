#!/usr/bin/env python3
"""
现代化累积差分数据价值评估CLI

支持多种模型、数据集和评估方法的组合实验。
使用 uv 管理依赖：uv run python -m experiments.run_experiment --help

特性：
- 配置文件支持
- 多种模型类型 (BERT, MLP, LogisticRegression)
- 多种数据集 (IMDB, 其他NLP/图像/表格数据)
- 多种评估方法 (LAVA, KNNShapley, InfluenceFunction)
- 现代化日志和进度显示
- 结果验证和统计分析
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.cumulative_differential import CumulativeDifferentialEvaluator, create_evaluator_from_config
from experiments.utils import (
    select_device, set_random_seeds, ModelFactory, DataProcessor,
    BertEmbeddingWrapper, ExperimentLogger, validate_csv_output, compute_statistics
)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """加载配置文件"""
    default_config = {
        "dataset": {
            "name": "imdb",
            "train_count": 1000,
            "valid_count": 200,
            "test_count": 200,
            "add_noise": None
        },
        "model": {
            "type": "bert",
            "pretrained_model": "distilbert-base-uncased",
            "kwargs": {}
        },
        "training": {
            "epochs": 5,
            "batch_size": 16,
            "lr": 2e-5,
            "save_every": 1
        },
        "evaluator": {
            "name": "lava",
            "kwargs": {
                "embedding_mode": "pooled"
            }
        },
        "experiment": {
            "output_dir": "./results/cumulative_differential",
            "output_prefix": "experiment",
            "device": "auto",
            "seed": 42,
            "skip_missing_checkpoints": True,
            "verify_telescope": True
        }
    }

    if config_path and config_path.exists():
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                user_config = yaml.safe_load(f)
            else:  # json
                user_config = json.load(f)

        # 合并配置
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = merge_dict(base[key], value)
                else:
                    base[key] = value
            return base

        default_config = merge_dict(default_config, user_config)

    return default_config


def create_model_with_data_compatibility(
    model_config: Dict[str, Any],
    fetcher_info: Dict[str, Any]
) -> torch.nn.Module:
    """根据数据集信息创建兼容的模型"""
    model_type = model_config["type"]

    if model_type == "bert":
        return ModelFactory.create_model(
            model_type=model_type,
            output_dim=fetcher_info["num_classes"],
            pretrained_model_name=model_config.get("pretrained_model", "distilbert-base-uncased"),
            **model_config.get("kwargs", {})
        )
    else:
        return ModelFactory.create_model(
            model_type=model_type,
            input_dim=fetcher_info["input_dim"],
            output_dim=fetcher_info["num_classes"],
            **model_config.get("kwargs", {})
        )


def run_experiment(config: Dict[str, Any], logger: ExperimentLogger):
    """执行完整的累积差分实验"""

    # 设置设备和随机种子
    device = select_device(config["experiment"]["device"])
    set_random_seeds(config["experiment"]["seed"])

    logger.log(f"🚀 开始累积差分数据价值评估实验")
    logger.log(f"设备: {device}")
    logger.log(f"配置: {config}")

    # 1. 准备数据
    logger.log("📂 准备数据...")
    dataset_config = config["dataset"]
    x_train, y_train, x_valid, y_valid, x_test, y_test, fetcher = DataProcessor.prepare_data(
        dataset_name=dataset_config["name"],
        train_count=dataset_config["train_count"],
        valid_count=dataset_config["valid_count"],
        test_count=dataset_config["test_count"],
        random_state=config["experiment"]["seed"],
        add_noise=dataset_config.get("add_noise")
    )

    # 获取数据信息
    fetcher_info = {
        "num_classes": fetcher.label_dim[0],
        "input_dim": getattr(fetcher, 'feature_dim', [None])[0],
        "is_text": dataset_config["name"] in ["imdb"]  # 可扩展
    }
    logger.log(f"数据信息: {fetcher_info}")

    # 处理文本数据
    if fetcher_info["is_text"]:
        x_train = DataProcessor.process_text_data(x_train)
        x_valid = DataProcessor.process_text_data(x_valid)
        y_train = DataProcessor.convert_labels(y_train)
        y_valid = DataProcessor.convert_labels(y_valid)

    # 2. 创建模型
    logger.log("🤖 创建模型...")
    model = create_model_with_data_compatibility(config["model"], fetcher_info)
    model.to(device)
    logger.log(f"模型类型: {config['model']['type']}")

    # 3. 创建累积差分评估器
    logger.log("⚙️ 创建累积差分评估器...")
    evaluator_config = config["evaluator"]

    # 根据评估器类型设置参数
    evaluator_kwargs = {
        "device": device,
        "random_state": config["experiment"]["seed"],
    }

    # 添加评估器特定参数
    evaluator_kwargs.update(evaluator_config.get("kwargs", {}))

    evaluator_class = create_evaluator_from_config(evaluator_config)

    cd_evaluator = CumulativeDifferentialEvaluator(
        evaluator_class=evaluator_class,
        evaluator_kwargs=evaluator_kwargs,
        device=device,
        random_state=config["experiment"]["seed"],
    )

    # 设置数据和检查点管理器
    cd_evaluator.setup_data(x_train, y_train, x_valid, y_valid)
    cd_evaluator.setup_checkpoint_manager(model)

    # 4. 训练模型并保存检查点
    logger.log("🏋️ 训练模型...")
    training_config = config["training"]

    with tqdm(total=training_config["epochs"], desc="训练进度") as pbar:
        class ProgressCallback:
            def __init__(self, pbar, logger):
                self.pbar = pbar
                self.logger = logger

            def on_epoch_end(self, epoch):
                self.pbar.update(1)
                self.logger.log(f"完成 epoch {epoch + 1}")

        # 注意：这里简化了进度回调，实际实现中需要修改训练函数
        trained_model = cd_evaluator.train_with_checkpoints(
            model=model,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            lr=training_config["lr"],
            save_every=training_config["save_every"],
        )
        pbar.n = training_config["epochs"]
        pbar.refresh()

    # 5. 计算累积差分
    logger.log("📊 计算累积差分影响力...")
    available_epochs = cd_evaluator.checkpoint_manager.available_epochs()
    logger.log(f"可用检查点: {available_epochs}")

    with tqdm(total=len(available_epochs), desc="计算影响力") as pbar:
        cumulative_diffs = cd_evaluator.compute_cumulative_differential(
            epochs=available_epochs,
            skip_missing=config["experiment"]["skip_missing_checkpoints"]
        )
        pbar.n = len(available_epochs)
        pbar.refresh()

    # 6. 验证望远镜求和
    if config["experiment"]["verify_telescope"]:
        logger.log("🔍 验证望远镜求和...")
        final_epoch = max(available_epochs)
        is_valid = cd_evaluator.verify_telescope_sum(cumulative_diffs, final_epoch)
        logger.log(f"望远镜求和验证: {'通过' if is_valid else '失败'}")

    # 7. 保存结果
    logger.log("💾 保存结果...")
    output_dir = Path(config["experiment"]["output_dir"])
    output_prefix = config["experiment"]["output_prefix"]
    dataset_name = config["dataset"]["name"]
    evaluator_name = config["evaluator"]["name"]
    seed = config["experiment"]["seed"]

    output_file = output_dir / f"{output_prefix}_{dataset_name}_{evaluator_name}_seed{seed}.csv"

    metadata = {
        "config": config,
        "device": str(device),
        "available_epochs": available_epochs,
        "fetcher_info": fetcher_info,
    }

    cd_evaluator.save_to_csv(cumulative_diffs, output_file, metadata)

    # 8. 验证输出文件
    logger.log("✅ 验证输出文件...")
    is_valid_csv = validate_csv_output(output_file, list(cumulative_diffs.keys()))

    # 9. 生成统计报告
    logger.log("📈 生成统计报告...")
    stats_report = {}
    for epoch, diff_data in cumulative_diffs.items():
        stats_report[f"epoch_{epoch}"] = compute_statistics(diff_data)

    # 保存统计报告
    stats_file = output_dir / f"{output_prefix}_{dataset_name}_{evaluator_name}_seed{seed}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_report, f, indent=2)

    logger.log(f"📊 统计报告保存至: {stats_file}")

    # 10. 实验总结
    logger.log("🎉 实验完成!")
    logger.log(f"结果文件: {output_file}")
    logger.log(f"统计报告: {stats_file}")
    logger.log(f"处理样本数: {len(next(iter(cumulative_diffs.values())))}")
    logger.log(f"处理轮数: {len(cumulative_diffs)}")

    return {
        "output_file": output_file,
        "stats_file": stats_file,
        "is_valid": is_valid_csv,
        "telescope_valid": is_valid if config["experiment"]["verify_telescope"] else None,
        "stats": stats_report
    }


def main():
    """主CLI入口"""
    parser = argparse.ArgumentParser(
        description="现代化累积差分数据价值评估CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config", "-c", type=Path,
        help="配置文件路径 (.json 或 .yaml)"
    )

    # 基础配置覆盖
    parser.add_argument("--dataset", help="数据集名称")
    parser.add_argument("--model", help="模型类型")
    parser.add_argument("--evaluator", help="评估器类型")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--device", help="计算设备")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--output-dir", type=Path, help="输出目录")

    # 实用选项
    parser.add_argument("--dry-run", action="store_true", help="显示配置但不运行实验")
    parser.add_argument("--list-models", action="store_true", help="列出支持的模型")
    parser.add_argument("--list-evaluators", action="store_true", help="列出支持的评估器")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 列出可用选项
    if args.list_models:
        print("支持的模型类型:")
        for model in ModelFactory.get_supported_models():
            print(f"  - {model}")
        return

    if args.list_evaluators:
        print("支持的评估器类型:")
        evaluators = ["lava", "knnshapley", "influence"]
        for evaluator in evaluators:
            print(f"  - {evaluator}")
        return

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.model:
        config["model"]["type"] = args.model
    if args.evaluator:
        config["evaluator"]["name"] = args.evaluator
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["experiment"]["device"] = args.device
    if args.seed:
        config["experiment"]["seed"] = args.seed
    if args.output_dir:
        config["experiment"]["output_dir"] = str(args.output_dir)

    # 创建输出目录和日志
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(output_dir)
    logger.save_config(config)

    # 显示配置
    if args.verbose or args.dry_run:
        print("🔧 实验配置:")
        print(json.dumps(config, indent=2))
        print()

    if args.dry_run:
        print("🏃‍♂️ 模拟运行模式，实际不执行实验")
        return

    try:
        # 运行实验
        results = run_experiment(config, logger)

        # 保存日志
        logger.save_logs()

        # 显示最终结果
        print("\n" + "="*60)
        print("🎊 实验完成!")
        print(f"📄 输出文件: {results['output_file']}")
        print(f"📊 统计报告: {results['stats_file']}")
        print(f"✅ CSV验证: {'通过' if results['is_valid'] else '失败'}")
        if results['telescope_valid'] is not None:
            print(f"🔭 望远镜求和: {'通过' if results['telescope_valid'] else '失败'}")
        print("="*60)

    except Exception as e:
        logger.log(f"❌ 实验失败: {e}", "ERROR")
        logger.save_logs()
        print(f"\n❌ 实验失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()