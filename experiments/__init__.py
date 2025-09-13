"""
累积差分数据价值评估框架

现代化的实验框架，支持将全局分数估值方法转换为逐epoch的累积差分输出。

主要模块：
- cumulative_differential: 核心累积差分评估器
- run_experiment: 现代化CLI接口
- utils: 工具函数和辅助类

使用方式：
```bash
# 使用配置文件
uv run python -m experiments.run_experiment --config experiments/configs/lava_bert_imdb.yaml

# 命令行参数
uv run python -m experiments.run_experiment --dataset imdb --model bert --evaluator lava --epochs 5

# 查看帮助
uv run python -m experiments.run_experiment --help
```
"""

from .cumulative_differential import CumulativeDifferentialEvaluator, ModelCheckpointManager
from .utils import (
    ModelFactory, DataProcessor, BertEmbeddingWrapper,
    ExperimentLogger, select_device, set_random_seeds
)

__version__ = "1.0.0"
__author__ = "OpenDataVal Team"

__all__ = [
    "CumulativeDifferentialEvaluator",
    "ModelCheckpointManager",
    "ModelFactory",
    "DataProcessor",
    "BertEmbeddingWrapper",
    "ExperimentLogger",
    "select_device",
    "set_random_seeds"
]