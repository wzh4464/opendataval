# BERT + TIM 情感分析实验

## 文件说明

- `bert_sentiment_analysis.py`: 完整的BERT + TIM实验代码
- `test_bert_tim.py`: 基本功能测试脚本

## 修复的问题

1. **模型兼容性**: 修正为仅支持DistilBERT系列模型（OpenDataVal的BertClassifier基于DistilBERT架构）
2. **数据处理**: 修复了Subset对象的处理，确保数据格式正确
3. **TIM配置**: 设置t1=0, t2=T（完整训练过程）

## 支持的模型

- `distilbert-base-uncased` (66M参数)
- `distilbert-base-cased` (66M参数) 
- `distilbert-base-multilingual-cased` (134M参数) - 最大模型

## 在服务器上运行

```bash
# 基本测试
uv run python my_experiments/test_bert_tim.py

# 完整实验
uv run python my_experiments/bert_sentiment_analysis.py
```

## 实验配置

- **数据集**: IMDB电影评论情感分析
- **TIM时间窗口**: [0, T] 完整训练过程
- **训练样本**: 500（可调整）
- **批处理大小**: 8（适合BERT）
- **训练轮数**: 2（减少实验时间）

## 输出结果

实验将生成：
- JSON格式的详细结果文件
- 每个模型的影响力分数
- 按类别分析的统计信息
- 最有影响力和最无影响力样本的索引