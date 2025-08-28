# CLAUDE_zh.md

此文件为Claude Code (claude.ai/code) 在此代码库中工作时提供中文指导。

## 项目概述

OpenDataVal是一个用于数据估值算法的统一基准库。这是一个Python库，为图像、自然语言处理和表格数据提供标准化数据集、数据估值方法和评估任务。

## 开发命令

### 安装

```bash
# 生产环境安装
make install

# 开发环境安装（包含pre-commit钩子）
make install-dev
```

### 测试和质量检查

```bash
# 运行测试并生成覆盖率报告
make coverage

# 使用ruff格式化代码
make format

# 清理临时文件
make clean
```

### 构建命令

```bash
# 更新依赖项
make build
```

### CLI工具使用

```bash
# 使用命令行工具
opendataval --file cli.csv -n [job_id] -o [path/to/output/]

# 或不安装直接运行
python opendataval --file cli.csv -n [job_id] -o [path/to/output/]
```

## 架构和组件

该库包含4个主要的交互组件：

### 1. DataFetcher (`opendataval.dataloader`) - 数据获取器

- 加载和预处理数据集
- 处理数据分割和噪声注入
- 可通过 `DataFetcher.datasets_available()` 查询可用数据集
- 支持图像/NLP数据集的嵌入向量

### 2. Model (`opendataval.model`) - 模型

- 提供可训练的预测模型
- 支持多种架构：LogisticRegression, MLP, LeNet, BERT
- 兼容scikit-learn模型
- 模型遵循PyTorch风格的API (`fit`, `predict`)

### 3. DataEvaluator (`opendataval.dataval`) - 数据评估器

- 实现不同的数据估值算法
- 可用方法：AME, CS-Shapley, DVRL, 影响函数, KNN-Shapley, LAVA, Leave-One-Out, Random, 基于体积的方法
- 所有评估器继承自 `DataEvaluator` 基类
- 通用模式：`evaluator.input_data().train_data_values().evaluate_data_values()`

### 4. ExperimentMediator (`opendataval.experiment`) - 实验协调器

- 跨多个数据评估器编排实验
- 处理模型训练和评估
- 提供绘图和非绘图实验方法
- 工厂方法：`ExperimentMediator.model_factory_setup()`

## 关键设计模式

### 数据处理流程

```python
# 标准工作流程
fetcher = DataFetcher(dataset_name='iris')
model = LogisticRegression(input_dim, output_dim)
evaluator = DataOob()

# 训练和评估
dataval = evaluator.train(fetcher, model, metric='accuracy')
data_values = dataval.evaluate_data_values()
```

### 实验设置

```python
# 使用ExperimentMediator
exper_med = ExperimentMediator.model_factory_setup(
    dataset_name='iris',
    train_count=50,
    valid_count=50,
    test_count=50,
    model_name='ClassifierMLP'
)
eval_med = exper_med.compute_data_values([DataOob()])
```

## 代码质量标准

- 使用ruff进行代码检查和格式化（行长度：88）
- 遵循numpy文档字符串约定
- 测试覆盖率目标：约75%
- Pre-commit钩子强制代码质量
- 支持Python 3.9-3.11

## 测试

- 测试文件位于 `test/` 目录
- 使用pytest进行覆盖率报告
- CI/CD在Ubuntu和Windows上运行测试
- 可运行单个测试文件：`pytest test/test_specific.py`

## 依赖项

关键依赖项版本固定以保证稳定性：

- PyTorch (~2.2.2) 用于深度学习模型
- scikit-learn (~1.3) 用于传统机器学习
- numpy (~1.26.4) 用于数值计算
- pandas 用于数据操作
- transformers (~4.38) 用于NLP模型

## 文档

- 基于Sphinx的文档：<https://opendataval.github.io>
- 每个主要包目录中的README文件提供具体指导
- `examples/` 目录中的示例演示使用模式

## 重要提示

- 建议保持CLAUDE.md为英文版本，因为：
  - 编程命令和API都是英文的
  - 模型对英文技术文档处理更准确
  - 国际协作更方便
- 此中文版本仅供理解参考
