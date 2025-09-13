"""
BERT训练和评估模块

提供模块化的BERT训练、评估和性能监控功能, 支持相同初始化的对比实验。
"""

import copy
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from opendataval.model import BertClassifier


class BertTrainer:
    """BERT训练器"""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        num_train_layers: int = 2,
        device: str = "auto",
        random_state: int = 42,
    ):
        """
        初始化BERT训练器

        Parameters:
        -----------
        model_name : str
            预训练模型名称
        num_classes : int
            分类类别数
        dropout_rate : float
            Dropout率
        num_train_layers : int
            微调层数
        device : str
            设备选择 ('auto', 'cpu', 'cuda', 'mps')
        random_state : int
            随机种子
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_train_layers = num_train_layers
        self.random_state = random_state

        # 设置设备
        self.device = self._setup_device(device)

        # 设置随机种子
        self._set_random_seed(random_state)

        # 训练历史
        self.training_history = defaultdict(list)

    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
                print("🚀 使用CUDA GPU加速")
            elif torch.backends.mps.is_available():
                device_obj = torch.device("mps")
                print("🍎 使用Apple Silicon MPS加速")
                # MPS优化设置
                import os

                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            else:
                device_obj = torch.device("cpu")
                print("💻 使用CPU")
        else:
            device_obj = torch.device(device)

        return device_obj

    def _set_random_seed(self, seed: int):
        """设置随机种子确保可重现"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # 确保CuDNN的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_model(self) -> BertClassifier:
        """创建BERT模型"""
        print(f"🤖 创建BERT模型: {self.model_name}")

        model = BertClassifier(
            pretrained_model_name=self.model_name,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            num_train_layers=self.num_train_layers,
        )

        model = model.to(self.device)
        print(f"📍 模型设备: {self.device}")

        return model

    def save_model_state(self, model: BertClassifier) -> Dict:
        """保存模型状态 (用于相同初始化)"""
        return copy.deepcopy(model.state_dict())

    def load_model_state(self, model: BertClassifier, state_dict: Dict):
        """加载模型状态 (用于相同初始化)"""
        model.load_state_dict(state_dict)
        return model

    def train_model(
        self,
        model: BertClassifier,
        data: Dict,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        eval_steps: int = 50,
        save_steps: int = 100,
        log_steps: int = 10,
    ) -> Dict:
        """
        训练BERT模型

        Parameters:
        -----------
        model : BertClassifier
            要训练的模型
        data : Dict
            包含训练数据的字典 {'x_train', 'y_train', 'x_valid', 'y_valid'}
        epochs : int
            训练轮数
        batch_size : int
            批次大小
        learning_rate : float
            学习率
        warmup_steps : int
            预热步数
        max_grad_norm : float
            梯度裁剪阈值
        eval_steps : int
            评估间隔
        save_steps : int
            保存间隔
        log_steps : int
            日志间隔

        Returns:
        --------
        Dict: 训练历史和最终性能
        """
        print("🚀 开始训练BERT模型")
        print(f"   训练样本: {len(data['y_train'])}")
        print(f"   验证样本: {len(data['y_valid'])}")
        print(f"   轮数: {epochs}, 批次大小: {batch_size}")
        print(f"   学习率: {learning_rate}")

        # 准备数据
        x_train, y_train = data["x_train"], data["y_train"]
        x_valid, y_valid = data["x_valid"], data["y_valid"]

        # 确保标签在正确设备上
        y_train = y_train.to(self.device)
        y_valid = y_valid.to(self.device)

        # 设置优化器和调度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

        # 计算总步数
        steps_per_epoch = len(x_train) // batch_size + (
            1 if len(x_train) % batch_size > 0 else 0
        )
        total_steps = epochs * steps_per_epoch

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.1, total_iters=total_steps
        )

        # 训练历史记录
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "valid_loss": [],
            "valid_accuracy": [],
            "learning_rates": [],
            "timestamps": [],
            "step_losses": [],  # 每步的loss, 用于绘图
            "step_accuracies": [],  # 每步的accuracy
        }

        model.train()
        start_time = time.time()
        step = 0

        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            print(f"\n📍 Epoch {epoch + 1}/{epochs}")

            # 批次训练
            for batch_start in range(0, len(x_train), batch_size):
                batch_end = min(batch_start + batch_size, len(x_train))
                batch_x = x_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]

                # 前向传播
                optimizer.zero_grad()

                try:
                    # BERT模型需要tokenization
                    outputs = model.predict(batch_x)  # 使用predict方法

                    # 计算损失
                    loss = torch.nn.functional.cross_entropy(outputs, batch_y)

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # 优化步骤
                    optimizer.step()
                    scheduler.step()

                    # 记录损失
                    epoch_losses.append(loss.item())
                    history["step_losses"].append(loss.item())

                    # 计算准确率
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == batch_y).sum().item()
                    accuracy = correct / len(batch_y)

                    epoch_correct += correct
                    epoch_total += len(batch_y)
                    history["step_accuracies"].append(accuracy)

                    step += 1

                    # 日志记录
                    if step % log_steps == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(
                            f"   Step {step}/{total_steps}: Loss={loss.item():.4f}, "
                            f"Acc={accuracy:.4f}, LR={current_lr:.2e}"
                        )

                    # 验证评估
                    if step % eval_steps == 0:
                        valid_loss, valid_acc = self.evaluate_model(
                            model, x_valid, y_valid
                        )
                        history["valid_loss"].append(valid_loss)
                        history["valid_accuracy"].append(valid_acc)
                        print(
                            f"   🔍 验证 - Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.4f}"
                        )
                        model.train()  # 回到训练模式

                except Exception as e:
                    print(f"❌ 训练步骤失败: {e}")
                    continue

            # Epoch结束统计
            epoch_loss = np.mean(epoch_losses)
            epoch_acc = epoch_correct / epoch_total

            history["train_loss"].append(epoch_loss)
            history["train_accuracy"].append(epoch_acc)
            history["learning_rates"].append(scheduler.get_last_lr()[0])
            history["timestamps"].append(time.time() - start_time)

            print(
                f"✅ Epoch {epoch + 1} 完成 - "
                f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
            )

        # 最终评估
        final_train_loss, final_train_acc = self.evaluate_model(model, x_train, y_train)
        final_valid_loss, final_valid_acc = self.evaluate_model(model, x_valid, y_valid)

        total_time = time.time() - start_time

        print(f"\n🎉 训练完成! 总耗时: {total_time:.1f}秒")
        print("📊 最终性能:")
        print(
            f"   训练集 - Loss: {final_train_loss:.4f}, Accuracy: {final_train_acc:.4f}"
        )
        print(
            f"   验证集 - Loss: {final_valid_loss:.4f}, Accuracy: {final_valid_acc:.4f}"
        )

        # 添加最终性能到历史记录
        history["final_performance"] = {
            "train_loss": final_train_loss,
            "train_accuracy": final_train_acc,
            "valid_loss": final_valid_loss,
            "valid_accuracy": final_valid_acc,
            "total_time": total_time,
            "total_steps": step,
        }

        return history

    def evaluate_model(
        self,
        model: BertClassifier,
        x_data: List,
        y_data: torch.Tensor,
        batch_size: int = 32,
    ) -> Tuple[float, float]:
        """
        评估模型性能

        Parameters:
        -----------
        model : BertClassifier
            要评估的模型
        x_data : List
            输入数据
        y_data : torch.Tensor
            标签数据
        batch_size : int
            评估批次大小

        Returns:
        --------
        Tuple[float, float]: (平均损失, 准确率)
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        y_data = y_data.to(self.device)

        with torch.no_grad():
            for batch_start in range(0, len(x_data), batch_size):
                batch_end = min(batch_start + batch_size, len(x_data))
                batch_x = x_data[batch_start:batch_end]
                batch_y = y_data[batch_start:batch_end]

                try:
                    outputs = model.predict(batch_x)
                    loss = torch.nn.functional.cross_entropy(outputs, batch_y)

                    total_loss += loss.item() * len(batch_y)

                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == batch_y).sum().item()
                    total_samples += len(batch_y)

                except Exception as e:
                    print(f"⚠️  评估批次失败: {e}")
                    continue

        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy

    def save_training_history(self, history: Dict, save_path: str):
        """保存训练历史"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存为numpy格式便于分析
        np.savez(
            save_path / "training_history.npz",
            **{
                k: np.array(v) if isinstance(v, list) else v
                for k, v in history.items()
                if k != "final_performance"
            },
        )

        # 保存最终性能为JSON
        import json

        if "final_performance" in history:
            with open(save_path / "final_performance.json", "w") as f:
                json.dump(history["final_performance"], f, indent=2)

        print(f"💾 训练历史已保存到: {save_path}")


def create_bert_trainer(
    model_name: str = "distilbert-base-uncased",
    num_classes: int = 2,
    dropout_rate: float = 0.2,
    num_train_layers: int = 2,
    device: str = "auto",
    random_state: int = 42,
) -> BertTrainer:
    """
    工厂函数: 创建BERT训练器

    Parameters:
    -----------
    model_name : str
        预训练模型名称, 默认"distilbert-base-uncased"
    num_classes : int
        分类类别数, 默认2
    dropout_rate : float
        Dropout率, 默认0.2
    num_train_layers : int
        微调层数, 默认2
    device : str
        设备选择, 默认"auto"
    random_state : int
        随机种子, 默认42

    Returns:
    --------
    BertTrainer
        配置好的训练器
    """
    return BertTrainer(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        num_train_layers=num_train_layers,
        device=device,
        random_state=random_state,
    )


if __name__ == "__main__":
    # 测试训练器
    print("🧪 测试BERT训练器")

    # 创建训练器
    trainer = create_bert_trainer()

    # 创建模型
    model = trainer.create_model()

    print("✅ BERT训练器测试完成")
