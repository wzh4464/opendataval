"""
累积差分框架的工具模块

包含设备选择、模型工厂、数据处理等辅助功能。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from opendataval.dataloader import DataFetcher
from opendataval.model import BertClassifier, ClassifierMLP, LogisticRegression, Model


def select_device(preference: str = "auto") -> torch.device:
    """智能设备选择"""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        preference == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    elif preference == "cpu":
        return torch.device("cpu")

    # auto模式
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_random_seeds(seed: int):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ModelFactory:
    """模型工厂类, 支持创建不同类型的模型"""

    @staticmethod
    def create_model(
        model_type: str,
        input_dim: Optional[int] = None,
        output_dim: int = 2,
        pretrained_model_name: str = "distilbert-base-uncased",
        **model_kwargs,
    ) -> Model:
        """创建指定类型的模型

        Parameters
        ----------
        model_type : str
            模型类型: "bert", "mlp", "logistic"
        input_dim : int, optional
            输入维度 (对于bert模型不需要)
        output_dim : int
            输出维度 (类别数)
        pretrained_model_name : str
            预训练模型名称 (仅BERT)
        model_kwargs : dict
            其他模型参数

        Returns
        -------
        Model
            创建的模型实例
        """
        model_type = model_type.lower()

        if model_type == "bert":
            return BertClassifier(
                pretrained_model_name=pretrained_model_name,
                num_classes=output_dim,
                **model_kwargs,
            )
        elif model_type == "mlp":
            if input_dim is None:
                raise ValueError("input_dim required for MLP model")
            return ClassifierMLP(
                input_dim=input_dim, num_classes=output_dim, **model_kwargs
            )
        elif model_type == "logistic":
            if input_dim is None:
                raise ValueError("input_dim required for Logistic model")
            return LogisticRegression(
                input_dim=input_dim, num_classes=output_dim, **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_supported_models() -> List[str]:
        """返回支持的模型类型"""
        return ["bert", "mlp", "logistic"]


class DataProcessor:
    """数据处理工具类"""

    @staticmethod
    def prepare_data(
        dataset_name: str,
        train_count: int,
        valid_count: int,
        test_count: int,
        random_state: int = 42,
        add_noise: Optional[Dict[str, Any]] = None,
    ):
        """准备数据集

        Returns
        -------
        tuple
            (x_train, y_train, x_valid, y_valid, x_test, y_test, fetcher)
        """
        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            random_state=random_state,
            add_noise=add_noise,
        )

        x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints

        return x_train, y_train, x_valid, y_valid, x_test, y_test, fetcher

    @staticmethod
    def process_text_data(x_data):
        """处理文本数据格式"""
        if hasattr(x_data, "dataset"):
            # Subset对象, 提取实际数据
            return [x_data.dataset[i] for i in x_data.indices]
        return x_data

    @staticmethod
    def convert_labels(y_tensor: torch.Tensor) -> torch.Tensor:
        """将one-hot标签转换为类别索引"""
        if y_tensor.ndim > 1 and y_tensor.shape[1] > 1:
            return torch.argmax(y_tensor, dim=1).long()
        return y_tensor.long()


class BertEmbeddingWrapper(torch.nn.Module):
    """BERT嵌入包装器, 用于LAVA等模型无关评估方法"""

    def __init__(self, bert_model: BertClassifier, mode: str = "pooled"):
        super().__init__()
        assert mode in {"pooled", "logits", "probs"}
        self.bert_model = bert_model
        self.mode = mode

    @property
    def device(self) -> torch.device:
        return self.bert_model.bert.device

    def predict(self, x_dataset) -> torch.Tensor:
        """返回BERT嵌入或预测结果"""
        if len(x_dataset) == 0:
            return torch.zeros(0, 1, device=self.device)

        # 将文本转为input_ids和attention_mask
        bert_inputs = self.bert_model.tokenize(x_dataset)
        input_ids, attention_mask = [t.to(self.device) for t in bert_inputs.tensors]

        with torch.no_grad():
            if self.mode == "pooled":
                # 返回CLS token的隐藏状态
                hidden_states = self.bert_model.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
                pooled = hidden_states[:, 0]  # CLS token
                return pooled
            elif self.mode == "logits":
                # 返回分类器logits
                hidden_states = self.bert_model.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
                pooled = hidden_states[:, 0]
                pre = self.bert_model.classifier.pre_linear(pooled)
                act = self.bert_model.classifier.acti(pre)
                drop = self.bert_model.classifier.dropout(act)
                logits = self.bert_model.classifier.linear(drop)
                return logits
            else:  # probs
                return self.bert_model.predict(x_dataset)


class ExperimentLogger:
    """实验记录器"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        """记录日志消息"""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "level": level, "message": message}
        self.logs.append(log_entry)
        print(f"[{timestamp}] {level}: {message}")

    def save_logs(self, filename: str = "experiment.log"):
        """保存日志到文件"""
        log_path = self.output_dir / filename
        with open(log_path, "w") as f:
            for log in self.logs:
                f.write(f"[{log['timestamp']}] {log['level']}: {log['message']}\n")

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """保存实验配置"""
        config_path = self.output_dir / filename
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)


def validate_csv_output(csv_path: Path, expected_epochs: List[int]) -> bool:
    """验证CSV输出格式是否正确"""
    import csv

    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # 检查列名格式
            expected_headers = [f"influence_epoch_{e}" for e in expected_epochs]
            if header != expected_headers:
                print(f"❌ CSV头部不匹配: 期望 {expected_headers}, 实际 {header}")
                return False

            # 检查数据行数
            data_rows = list(reader)
            if len(data_rows) == 0:
                print("❌ CSV没有数据行")
                return False

            # 检查数据格式
            for i, row in enumerate(data_rows[:5]):  # 检查前5行
                if len(row) != len(expected_epochs):
                    print(
                        f"❌ 第{i+1}行列数不匹配: 期望 {len(expected_epochs)}, 实际 {len(row)}"
                    )
                    return False

                try:
                    [float(val) for val in row]
                except ValueError:
                    print(f"❌ 第{i+1}行包含非数值数据: {row}")
                    return False

        print(f"✅ CSV格式验证通过: {len(data_rows)} 行, {len(expected_epochs)} 列")
        return True

    except Exception as e:
        print(f"❌ CSV验证失败: {e}")
        return False


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """计算数据统计信息"""
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "count": len(data),
    }
