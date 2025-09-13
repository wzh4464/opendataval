#!/usr/bin/env python3
"""
累积差分数据价值评估框架

实现将"只有全局分数"的估值方法转换为逐epoch的累积差分输出。

核心思想：
- I(e): 将第e个epoch的检查点当作"最终模型"来计算全局影响力向量
- ΔI(e) = I(e) - I(e-1): 第e轮的新增贡献 (I(-1) = 0向量)
- 输出CSV: 每列influence_epoch_e填ΔI(e)
- 望远镜求和: 所有列相加应等于最终影响力I(E)

支持所有全局分数估值方法：LAVA、KNNShapley、InfluenceFunction等
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher
from opendataval.dataval.api import DataEvaluator
from opendataval.dataval.lava import LavaEvaluator
from opendataval.dataval.knnshap import KNNShapley
from opendataval.dataval.influence import InfluenceFunction
from opendataval.model import Model, BertClassifier
from opendataval.util import set_random_seed


class ModelCheckpointManager:
    """管理训练过程中的模型检查点"""

    def __init__(self, base_model: Model, device: torch.device):
        self.base_model = base_model
        self.device = device
        self.checkpoints: Dict[int, Dict[str, Any]] = {}

    def save_checkpoint(self, epoch: int, model_state: Dict[str, Any]):
        """保存指定epoch的模型状态"""
        # 深拷贝模型状态到CPU以节省GPU内存
        checkpoint = {k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in model_state.items()}
        self.checkpoints[epoch] = checkpoint

    def load_checkpoint(self, epoch: int) -> Model:
        """加载指定epoch的模型检查点"""
        if epoch not in self.checkpoints:
            raise ValueError(f"Checkpoint for epoch {epoch} not found")

        # 克隆基础模型并加载状态
        model = self.base_model.clone()
        model.load_state_dict(self.checkpoints[epoch])
        model.to(self.device)
        return model

    def has_checkpoint(self, epoch: int) -> bool:
        """检查是否存在指定epoch的检查点"""
        return epoch in self.checkpoints

    def available_epochs(self) -> List[int]:
        """返回所有可用的epoch"""
        return sorted(self.checkpoints.keys())


class CumulativeDifferentialEvaluator:
    """累积差分数据价值评估器"""

    def __init__(
        self,
        evaluator_class: Type[DataEvaluator],
        evaluator_kwargs: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        random_state: Optional[RandomState] = None,
    ):
        self.evaluator_class = evaluator_class
        self.evaluator_kwargs = evaluator_kwargs
        self.device = device
        self.random_state = check_random_state(random_state)

        # 存储历史影响力计算结果
        self.influence_history: Dict[int, np.ndarray] = {}
        self.checkpoint_manager: Optional[ModelCheckpointManager] = None

    def setup_data(
        self,
        x_train: Union[torch.Tensor, List],
        y_train: torch.Tensor,
        x_valid: Union[torch.Tensor, List],
        y_valid: torch.Tensor,
    ):
        """设置训练和验证数据"""
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def setup_checkpoint_manager(self, base_model: Model):
        """设置检查点管理器"""
        self.checkpoint_manager = ModelCheckpointManager(base_model, self.device)

    def train_with_checkpoints(
        self,
        model: Model,
        epochs: int,
        batch_size: int = 32,
        lr: float = 1e-3,
        save_every: int = 1,
        **train_kwargs
    ) -> Model:
        """训练模型并保存检查点"""
        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager not initialized")

        print(f"🚀 开始训练 {epochs} 轮，每 {save_every} 轮保存检查点")

        # 保存初始状态 (epoch -1, 实际上是epoch 0的初始状态)
        self.checkpoint_manager.save_checkpoint(-1, model.state_dict())

        # 准备训练数据
        if isinstance(self.x_train, list):  # 文本数据
            x_train_tensor = self.y_train  # 占位符，实际使用原始文本
        else:
            x_train_tensor = self.x_train

        # 训练循环
        for epoch in range(epochs):
            print(f"  📈 训练 Epoch {epoch + 1}/{epochs}")

            # 执行一轮训练
            if hasattr(model, 'fit_epoch'):
                # 如果模型支持单epoch训练
                model.fit_epoch(self.x_train, self.y_train,
                               batch_size=batch_size, lr=lr, **train_kwargs)
            else:
                # 否则训练1个epoch
                model.fit(self.x_train, self.y_train,
                         epochs=1, batch_size=batch_size, lr=lr, **train_kwargs)

            # 保存检查点
            if (epoch + 1) % save_every == 0:
                self.checkpoint_manager.save_checkpoint(epoch, model.state_dict())
                print(f"    💾 保存检查点: epoch {epoch}")

        print("✅ 训练完成")
        return model

    def compute_influence_at_epoch(self, epoch: int) -> np.ndarray:
        """计算指定epoch检查点的影响力分数"""
        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager not initialized")

        if epoch in self.influence_history:
            return self.influence_history[epoch]

        print(f"  🧮 计算 epoch {epoch} 的影响力...")

        # 加载指定epoch的模型检查点
        if epoch == -1:
            # 特殊情况：初始模型 (未训练)
            model_at_epoch = self.checkpoint_manager.load_checkpoint(-1)
        else:
            model_at_epoch = self.checkpoint_manager.load_checkpoint(epoch)

        # 创建评估器实例
        evaluator = self.evaluator_class(**self.evaluator_kwargs)

        # 设置数据
        evaluator.input_data(self.x_train, self.y_train, self.x_valid, self.y_valid)

        # 特殊处理不同类型的评估器
        if hasattr(evaluator, 'embedding_model'):
            # ModelLessMixin (LAVA, KNNShapley)
            if isinstance(model_at_epoch, BertClassifier):
                # 对于BERT模型，需要包装为嵌入模型
                from my_experiments.run_lava_bert import BertEmbeddingWrapper
                embedding_model = BertEmbeddingWrapper(model_at_epoch, mode="pooled")
                evaluator.embedding_model = embedding_model
        elif hasattr(evaluator, 'pred_model'):
            # ModelMixin (InfluenceFunction)
            evaluator.pred_model = model_at_epoch

        # 训练并计算数据价值
        evaluator.train_data_values()
        influence_scores = evaluator.evaluate_data_values()

        # 缓存结果
        self.influence_history[epoch] = influence_scores
        return influence_scores

    def compute_cumulative_differential(
        self,
        epochs: List[int],
        skip_missing: bool = True
    ) -> Dict[int, np.ndarray]:
        """计算累积差分影响力"""
        print(f"📊 计算累积差分影响力: epochs {epochs}")

        # 确保epochs排序
        epochs = sorted(epochs)

        cumulative_diffs = {}
        prev_influence = None

        for epoch in epochs:
            # 检查检查点是否存在
            if not self.checkpoint_manager.has_checkpoint(epoch):
                if skip_missing:
                    print(f"⚠️  跳过缺失的检查点: epoch {epoch}")
                    continue
                else:
                    raise ValueError(f"Missing checkpoint for epoch {epoch}")

            # 计算当前epoch的影响力
            current_influence = self.compute_influence_at_epoch(epoch)

            # 计算差分
            if prev_influence is None:
                # 第一个epoch：ΔI(e) = I(e) - 0
                diff = current_influence.copy()
            else:
                # 后续epoch：ΔI(e) = I(e) - I(e-1)
                diff = current_influence - prev_influence

            cumulative_diffs[epoch] = diff
            prev_influence = current_influence.copy()

            print(f"    ✓ epoch {epoch}: 差分统计 mean={diff.mean():.6f}, std={diff.std():.6f}")

        return cumulative_diffs

    def save_to_csv(
        self,
        cumulative_diffs: Dict[int, np.ndarray],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """将累积差分结果保存为CSV文件"""
        print(f"💾 保存结果到: {output_path}")

        if not cumulative_diffs:
            raise ValueError("No cumulative differential data to save")

        # 获取数据维度
        sample_diff = next(iter(cumulative_diffs.values()))
        n_samples = len(sample_diff)
        epochs = sorted(cumulative_diffs.keys())

        # 写入CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 写入header
            headers = [f'influence_epoch_{e}' for e in epochs]
            writer.writerow(headers)

            # 写入数据 (按样本行，按epoch列)
            for i in range(n_samples):
                row = [cumulative_diffs[epoch][i] for epoch in epochs]
                writer.writerow(row)

        # 保存元数据
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            metadata_full = {
                'epochs': epochs,
                'n_samples': n_samples,
                'evaluator_class': self.evaluator_class.__name__,
                'evaluator_kwargs': self.evaluator_kwargs,
                **metadata
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata_full, f, indent=2, default=str)

        print(f"✅ 保存完成: {len(epochs)} 个epoch, {n_samples} 个样本")

    def verify_telescope_sum(
        self,
        cumulative_diffs: Dict[int, np.ndarray],
        final_epoch: int,
        tolerance: float = 1e-6
    ) -> bool:
        """验证望远镜求和：所有差分相加应等于最终影响力"""
        print(f"🔍 验证望远镜求和 (tolerance={tolerance})")

        # 计算最终影响力
        final_influence = self.compute_influence_at_epoch(final_epoch)

        # 累加所有差分
        epochs = sorted(cumulative_diffs.keys())
        summed_diffs = np.zeros_like(final_influence)
        for epoch in epochs:
            summed_diffs += cumulative_diffs[epoch]

        # 比较
        diff = np.abs(final_influence - summed_diffs)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        is_valid = max_diff < tolerance

        print(f"  📏 最大差异: {max_diff:.10f}")
        print(f"  📊 平均差异: {mean_diff:.10f}")
        print(f"  {'✅' if is_valid else '❌'} 望远镜求和{'通过' if is_valid else '失败'}")

        return is_valid


def create_evaluator_from_config(config: Dict[str, Any]) -> Type[DataEvaluator]:
    """根据配置创建评估器类"""
    evaluator_map = {
        'lava': LavaEvaluator,
        'knnshapley': KNNShapley,
        'influence': InfluenceFunction,
    }

    evaluator_name = config['name'].lower()
    if evaluator_name not in evaluator_map:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")

    return evaluator_map[evaluator_name]


def main():
    """主函数：累积差分数据价值评估CLI"""
    parser = argparse.ArgumentParser(
        description="累积差分数据价值评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集配置
    parser.add_argument("--dataset", default="imdb",
                       help="数据集名称")
    parser.add_argument("--train-count", type=int, default=1000,
                       help="训练样本数")
    parser.add_argument("--valid-count", type=int, default=200,
                       help="验证样本数")
    parser.add_argument("--test-count", type=int, default=200,
                       help="测试样本数")

    # 模型配置
    parser.add_argument("--model", default="bert",
                       choices=["bert", "mlp", "logistic"],
                       help="模型类型")
    parser.add_argument("--pretrained-model",
                       default="distilbert-base-uncased",
                       help="预训练模型名称(仅BERT)")

    # 训练配置
    parser.add_argument("--epochs", type=int, default=5,
                       help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="批量大小")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--save-every", type=int, default=1,
                       help="每几轮保存检查点")

    # 评估器配置
    parser.add_argument("--evaluator", default="lava",
                       choices=["lava", "knnshapley", "influence"],
                       help="数据价值评估方法")
    parser.add_argument("--embedding-mode", default="pooled",
                       choices=["pooled", "logits", "probs"],
                       help="嵌入模式(仅LAVA)")

    # 输出配置
    parser.add_argument("--output-dir",
                       default="./results/cumulative_differential",
                       help="输出目录")
    parser.add_argument("--output-prefix", default="experiment",
                       help="输出文件前缀")

    # 其他配置
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="设备选择")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--skip-missing-checkpoints", action="store_true",
                       help="跳过缺失的检查点")

    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"🚀 累积差分数据价值评估")
    print(f"  设备: {device}")
    print(f"  数据集: {args.dataset}")
    print(f"  模型: {args.model}")
    print(f"  评估器: {args.evaluator}")
    print(f"  训练轮数: {args.epochs}")

    # 设置随机种子
    set_random_seed(args.seed)

    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("📂 加载数据...")
    fetcher = DataFetcher.setup(
        dataset_name=args.dataset,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        random_state=args.seed,
    )
    x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints

    # 创建模型
    print("🤖 创建模型...")
    if args.model == "bert":
        model = BertClassifier(
            pretrained_model_name=args.pretrained_model,
            num_classes=fetcher.label_dim[0]
        ).to(device)
        # 处理文本数据格式
        if hasattr(x_train, 'dataset'):
            x_train = [x_train.dataset[i] for i in x_train.indices]
            x_valid = [x_valid.dataset[i] for i in x_valid.indices]
    else:
        raise NotImplementedError(f"Model {args.model} not implemented yet")

    # 创建累积差分评估器
    print("⚙️  创建累积差分评估器...")
    evaluator_kwargs = {}
    if args.evaluator == "lava":
        evaluator_kwargs = {
            "device": device,
            "random_state": args.seed,
        }
    elif args.evaluator == "knnshapley":
        evaluator_kwargs = {
            "random_state": args.seed,
        }
    elif args.evaluator == "influence":
        evaluator_kwargs = {}

    evaluator_class = create_evaluator_from_config({"name": args.evaluator})

    cd_evaluator = CumulativeDifferentialEvaluator(
        evaluator_class=evaluator_class,
        evaluator_kwargs=evaluator_kwargs,
        device=device,
        random_state=args.seed,
    )

    # 设置数据和检查点管理器
    cd_evaluator.setup_data(x_train, y_train, x_valid, y_valid)
    cd_evaluator.setup_checkpoint_manager(model)

    # 训练模型并保存检查点
    print("🏋️  训练模型...")
    trained_model = cd_evaluator.train_with_checkpoints(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
    )

    # 计算累积差分
    available_epochs = cd_evaluator.checkpoint_manager.available_epochs()
    print(f"📊 可用检查点: {available_epochs}")

    cumulative_diffs = cd_evaluator.compute_cumulative_differential(
        epochs=available_epochs,
        skip_missing=args.skip_missing_checkpoints
    )

    # 验证望远镜求和
    final_epoch = max(available_epochs)
    cd_evaluator.verify_telescope_sum(cumulative_diffs, final_epoch)

    # 保存结果
    output_file = output_dir / f"{args.output_prefix}_{args.dataset}_{args.evaluator}_seed{args.seed}.csv"
    metadata = {
        "args": vars(args),
        "device": str(device),
        "final_epoch": final_epoch,
        "available_epochs": available_epochs,
    }

    cd_evaluator.save_to_csv(cumulative_diffs, output_file, metadata)

    print("🎉 累积差分评估完成!")


if __name__ == "__main__":
    main()