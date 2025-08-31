"""
BERT情感分析实验 - 使用OpenDataVal TIM方法

使用Time-varying Influence Measurement (TIM)进行BERT情感分析微调的数据价值评估实验。
本实验设置 t1 = 0, t2 = T（完整训练过程），使用不同大小的BERT模型进行比较。

实验配置：
- 数据集: IMDB电影评论情感分析数据集
- 模型: 多种BERT模型大小选项（从DistilBERT到BERT-Large）
- 评估方法: TIM (Time-varying Influence Measurement)
- 时间窗口: [0, T] - 完整训练过程
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from opendataval.dataloader import DataFetcher
from opendataval.dataval.tim import TimInfluence
from opendataval.model import BertClassifier


def get_bert_model_configs() -> Dict[str, Dict]:
    """获取不同大小的BERT模型配置

    返回从小到大的BERT模型配置列表，包括参数规模信息。
    注意：OpenDataVal的BertClassifier基于DistilBERT架构，只能使用DistilBERT预训练模型。

    Returns:
        Dict[str, Dict]: 模型配置字典，键为模型名称，值为配置参数
    """
    return {
        # DistilBERT模型 (OpenDataVal支持)
        "distilbert-base-uncased": {
            "pretrained_model_name": "distilbert-base-uncased",
            "parameters": "66M",
            "description": "DistilBERT-Base (66M参数) - BERT的轻量级版本，速度快",
        },
        "distilbert-base-cased": {
            "pretrained_model_name": "distilbert-base-cased", 
            "parameters": "66M",
            "description": "DistilBERT-Base-Cased (66M参数) - 区分大小写版本",
        },
        # 多语言DistilBERT
        "distilbert-base-multilingual-cased": {
            "pretrained_model_name": "distilbert-base-multilingual-cased",
            "parameters": "134M", 
            "description": "DistilBERT多语言 (134M参数) - 支持多种语言",
        },
        # 注意：标准BERT模型与DistilBERT架构不兼容，已移除
        # 如需使用更大模型，需要修改BertClassifier类的实现
    }


class BertTimExperiment:
    """BERT + TIM 情感分析实验类"""

    def __init__(
        self,
        dataset_name: str = "imdb",
        train_count: int = 1000,
        valid_count: int = 200,
        test_count: int = 200,
        random_state: int = 42,
        output_dir: str = "./results",
    ):
        """
        初始化实验配置

        Parameters:
        -----------
        dataset_name : str
            数据集名称，默认"imdb"用于情感分析
        train_count : int
            训练样本数量
        valid_count : int
            验证样本数量
        test_count : int
            测试样本数量
        random_state : int
            随机种子
        output_dir : str
            结果输出目录
        """
        self.dataset_name = dataset_name
        self.train_count = train_count
        self.valid_count = valid_count
        self.test_count = test_count
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # 实验结果存储
        self.results = {}

    def prepare_data(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备IMDB情感分析数据"""
        print(f"🔄 加载数据集: {self.dataset_name}")
        print(
            f"📊 数据规模: 训练={self.train_count}, 验证={self.valid_count}, 测试={self.test_count}"
        )

        # 使用DataFetcher.setup加载IMDB数据集，并指定数据分割
        fetcher = DataFetcher.setup(
            dataset_name=self.dataset_name,
            train_count=self.train_count,
            valid_count=self.valid_count,
            test_count=self.test_count,
            random_state=self.random_state,
        )

        # 获取原始文本数据（不使用embedding）
        x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
        
        # 转换数据类型以确保兼容性
        if hasattr(x_train, 'dataset'):
            # 如果是Subset对象，提取实际数据
            x_train_data = [x_train.dataset[i] for i in x_train.indices]
            x_valid_data = [x_valid.dataset[i] for i in x_valid.indices] 
            x_test_data = [x_test.dataset[i] for i in x_test.indices]
        else:
            x_train_data, x_valid_data, x_test_data = x_train, x_valid, x_test
            
        # 确保标签是torch tensor格式，并转换one-hot为索引
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if not isinstance(y_valid, torch.Tensor):
            y_valid = torch.tensor(y_valid, dtype=torch.long)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.long)
            
        # 如果是one-hot编码，转换为索引格式
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = torch.argmax(y_train, dim=1)
            print(f"   转换one-hot标签为索引: {y_train[:5]}")
        if len(y_valid.shape) > 1 and y_valid.shape[1] > 1:
            y_valid = torch.argmax(y_valid, dim=1)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = torch.argmax(y_test, dim=1)

        print("✅ 数据加载完成")
        print(f"   训练集样本数: {len(x_train_data)}")
        print(f"   验证集样本数: {len(x_valid_data)}")
        print(f"   测试集样本数: {len(x_test_data)}")
        print(f"   类别数: {len(np.unique(y_train))}")
        
        # 返回处理后的数据
        return x_train_data, y_train, x_valid_data, y_valid, x_test_data, y_test

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def create_bert_model(self, model_config: Dict) -> BertClassifier:
        """创建BERT分类器模型"""
        model = BertClassifier(
            pretrained_model_name=model_config["pretrained_model_name"],
            num_classes=2,  # 二分类情感分析
            dropout_rate=0.2,
            num_train_layers=2,  # 微调最后2层
        )

        # 智能设备选择，优先GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("🚀 使用CUDA GPU加速")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🍎 使用Apple Silicon MPS加速")
            # MPS优化设置
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        else:
            device = torch.device("cpu")
            print("💻 使用CPU (未检测到GPU)")
            
        model = model.to(device)

        print(f"🤖 创建模型: {model_config['description']}")
        print(f"📍 设备: {device}")

        return model

    def setup_tim_evaluator(
        self, t1: int = 0, t2: int = None, num_epochs: int = 5, batch_size: int = 16
    ) -> TimInfluence:
        """
        设置TIM评估器

        Parameters:
        -----------
        t1 : int
            时间窗口开始步骤，默认0（从开始）
        t2 : int
            时间窗口结束步骤，None表示到结束（T）
        num_epochs : int
            训练轮数
        batch_size : int
            批处理大小
        """
        print("⚙️  设置TIM评估器")
        print(f"   时间窗口: t1={t1}, t2={'T(end)' if t2 is None else t2}")
        print(f"   训练配置: epochs={num_epochs}, batch_size={batch_size}")

        tim_evaluator = TimInfluence(
            start_step=t1,
            end_step=t2,
            time_window_type="full" if t1 == 0 and t2 is None else "custom_range",
            num_epochs=num_epochs,
            batch_size=batch_size,
            regularization=0.01,
            finite_diff_eps=1e-5,
            random_state=self.random_state,
        )

        return tim_evaluator

    def run_single_experiment(
        self, model_name: str, model_config: Dict, data: Tuple, tim_config: Dict = None
    ) -> Dict:
        """运行单个BERT+TIM实验"""
        x_train, y_train, x_valid, y_valid, x_test, y_test = data

        print("\n" + "=" * 60)
        print(f"🔬 开始实验: {model_name}")
        print(f"📝 {model_config['description']}")
        print("=" * 60)

        # 默认TIM配置
        if tim_config is None:
            tim_config = {
                "t1": 0,
                "t2": None,  # 到结束
                "num_epochs": 3,
                "batch_size": 8,  # BERT需要较小的batch size
            }

        try:
            # 1. 创建模型
            model = self.create_bert_model(model_config)

            # 2. 设置TIM评估器
            tim_evaluator = self.setup_tim_evaluator(**tim_config)

            # 3. 输入数据到TIM - 需要转换为tensor格式
            # TIM内部需要tensor数据，但我们有文本数据，需要先tokenize
            print("   🔄 对文本数据进行tokenization...")
            
            # 使用模型的tokenizer处理文本数据
            train_dataset = model.tokenize(x_train)
            valid_dataset = model.tokenize(x_valid)
            
            # 获取tokenized的tensor数据
            train_input_ids = train_dataset.tensors[0]
            train_attention_mask = train_dataset.tensors[1]  
            valid_input_ids = valid_dataset.tensors[0]
            valid_attention_mask = valid_dataset.tensors[1]
            
            # 确保数据在正确的设备上
            device = model.bert.device
            train_input_ids = train_input_ids.to(device)
            train_attention_mask = train_attention_mask.to(device)
            valid_input_ids = valid_input_ids.to(device)
            valid_attention_mask = valid_attention_mask.to(device)
            y_train = y_train.to(device)
            y_valid = y_valid.to(device)
            
            # 为TIM创建简化的tensor输入（转换为float以支持梯度计算）
            tim_evaluator.input_data(
                x_train=train_input_ids.float(),
                y_train=y_train,
                x_valid=valid_input_ids.float(),
                y_valid=y_valid
            )

            # 4. 创建TIM兼容的BERT包装器
            class BertTimWrapper(torch.nn.Module):
                """包装BERT模型以兼容TIM的tensor输入格式"""
                def __init__(self, bert_model, attention_mask):
                    super().__init__()
                    self.bert_model = bert_model
                    self.attention_mask = attention_mask.detach()  # 避免梯度问题
                    
                def forward(self, input_ids):
                    # TIM传递的是float tensor，我们需要转换为token IDs
                    batch_size = input_ids.shape[0]
                    device = input_ids.device
                    
                    # 使用对应的attention mask片段  
                    mask = self.attention_mask[:batch_size].to(device)
                    
                    # 将float tensor转为long token IDs
                    input_ids_long = input_ids.long()
                    
                    # 调用BERT并获取logits（不要softmax）
                    outputs = self.bert_model(input_ids_long, attention_mask=mask)
                    
                    # 移除最后的Softmax层，直接返回logits以便梯度传播
                    # BERT classifier的最后一层是softmax，我们需要raw logits
                    if hasattr(self.bert_model, 'classifier'):
                        # 获取分类器之前的hidden states
                        hidden_states = self.bert_model.bert(input_ids_long, attention_mask=mask)[0]
                        pooled_output = hidden_states[:, 0]  # [CLS] token
                        
                        # 只通过linear层，不要softmax
                        pre_linear = self.bert_model.classifier.pre_linear(pooled_output)
                        activated = self.bert_model.classifier.acti(pre_linear)
                        dropped = self.bert_model.classifier.dropout(activated)
                        logits = self.bert_model.classifier.linear(dropped)
                        
                        return logits  # 返回raw logits而不是softmax输出
                    else:
                        return outputs
                    
                def predict(self, input_ids):
                    """TIM调用的预测接口"""
                    with torch.enable_grad():
                        return self.forward(input_ids)
                    
                def parameters(self):
                    return self.bert_model.parameters()
                    
                def named_parameters(self):
                    return self.bert_model.named_parameters()
                    
                def zero_grad(self):
                    return self.bert_model.zero_grad()
                    
                def train(self):
                    self.bert_model.train()
                    return self
                    
                def eval(self):
                    self.bert_model.eval() 
                    return self
            
            # 创建包装器
            bert_wrapper = BertTimWrapper(model, train_attention_mask)
            tim_evaluator.pred_model = bert_wrapper

            # 5. 训练并记录状态
            print("\n🚀 开始TIM训练...")
            tim_evaluator.train_data_values(
                epochs=tim_config["num_epochs"],
                batch_size=tim_config["batch_size"],
                lr=2e-5,  # BERT推荐学习率
            )

            # 6. 计算影响力数据值
            print("\n📊 计算数据影响力...")
            data_values = tim_evaluator.evaluate_data_values()

            # 7. 分析结果
            results = self.analyze_results(
                model_name=model_name,
                data_values=data_values,
                tim_evaluator=tim_evaluator,
                y_train=y_train,
            )

            print(f"✅ 实验完成: {model_name}")
            return results

        except Exception as e:
            print(f"❌ 实验失败: {model_name}")
            print(f"   错误: {e!s}")
            return {"model_name": model_name, "status": "failed", "error": str(e)}

    def analyze_results(
        self,
        model_name: str,
        data_values: np.ndarray,
        tim_evaluator: TimInfluence,
        y_train: torch.Tensor,
    ) -> Dict:
        """分析TIM实验结果"""

        # 基础统计
        mean_influence = float(np.mean(data_values))
        std_influence = float(np.std(data_values))
        min_influence = float(np.min(data_values))
        max_influence = float(np.max(data_values))

        # 按类别分析影响力 - 确保tensor在CPU上
        if isinstance(y_train, torch.Tensor):
            y_train_np = y_train.cpu().numpy()
        else:
            y_train_np = y_train

        positive_indices = np.where(y_train_np == 1)[0]
        negative_indices = np.where(y_train_np == 0)[0]

        positive_influence = data_values[positive_indices]
        negative_influence = data_values[negative_indices]

        # 找出最有影响力的样本
        top_k = 10
        most_influential_indices = np.argsort(data_values)[-top_k:][::-1]
        least_influential_indices = np.argsort(data_values)[:top_k]

        results = {
            "model_name": model_name,
            "status": "success",
            "data_values": data_values.tolist(),
            "statistics": {
                "mean_influence": mean_influence,
                "std_influence": std_influence,
                "min_influence": min_influence,
                "max_influence": max_influence,
                "total_samples": len(data_values),
            },
            "class_analysis": {
                "positive_samples": {
                    "count": len(positive_influence),
                    "mean_influence": float(np.mean(positive_influence)),
                    "std_influence": float(np.std(positive_influence)),
                },
                "negative_samples": {
                    "count": len(negative_influence),
                    "mean_influence": float(np.mean(negative_influence)),
                    "std_influence": float(np.std(negative_influence)),
                },
            },
            "top_influential": {
                "indices": most_influential_indices.tolist(),
                "values": data_values[most_influential_indices].tolist(),
            },
            "least_influential": {
                "indices": least_influential_indices.tolist(),
                "values": data_values[least_influential_indices].tolist(),
            },
            "training_info": {
                "total_steps": tim_evaluator.total_steps,
                "steps_per_epoch": tim_evaluator.steps_per_epoch,
                "cached_intervals": len(tim_evaluator._influence_cache),
            },
        }

        # 打印关键结果
        print("\n📈 实验结果摘要:")
        print(f"   平均影响力: {mean_influence:.6f}")
        print(f"   影响力标准差: {std_influence:.6f}")
        print(f"   影响力范围: [{min_influence:.6f}, {max_influence:.6f}]")
        print(
            f"   正面样本平均影响力: {results['class_analysis']['positive_samples']['mean_influence']:.6f}"
        )
        print(
            f"   负面样本平均影响力: {results['class_analysis']['negative_samples']['mean_influence']:.6f}"
        )
        print(f"   训练总步数: {tim_evaluator.total_steps}")

        return results

    def save_results(self, filename: str = None):
        """保存实验结果到JSON文件"""
        import json

        if filename is None:
            filename = "bert_tim_experiment_results.json"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"💾 结果已保存到: {filepath}")

    def run_full_experiment_suite(self, selected_models: List[str] = None):
        """运行完整的BERT模型对比实验"""

        print("🔬 BERT + TIM 情感分析实验套件")
        print("=" * 80)

        # 获取模型配置
        model_configs = get_bert_model_configs()

        if selected_models is None:
            # 默认选择支持的DistilBERT模型
            selected_models = [
                "distilbert-base-uncased",  # 基础: 66M参数
                "distilbert-base-cased",    # 区分大小写: 66M参数  
                "distilbert-base-multilingual-cased",  # 多语言: 134M参数 (最大)
            ]

        print(f"📋 选择的模型: {selected_models}")

        # 准备数据（所有实验使用相同数据）
        data = self.prepare_data()

        # TIM配置 - 设置 t1=0, t2=T（完整训练过程）
        tim_config = {
            "t1": 0,  # 从训练开始
            "t2": None,  # 到训练结束（T）
            "num_epochs": 2,  # 减少epoch数以适应实验
            "batch_size": 8,  # 较小的batch size适合BERT
        }

        print(
            f"⚙️  TIM配置: t1={tim_config['t1']}, t2=T, epochs={tim_config['num_epochs']}"
        )

        # 运行每个模型的实验
        for model_name in selected_models:
            if model_name not in model_configs:
                print(f"⚠️  跳过未知模型: {model_name}")
                continue

            model_config = model_configs[model_name]

            # 运行实验
            result = self.run_single_experiment(
                model_name=model_name,
                model_config=model_config,
                data=data,
                tim_config=tim_config,
            )

            self.results[model_name] = result

        # 保存结果
        self.save_results()

        # 打印实验摘要
        self.print_experiment_summary()

    def print_experiment_summary(self):
        """打印实验结果摘要"""
        print("\n" + "=" * 80)
        print("📊 BERT + TIM 实验结果摘要")
        print("=" * 80)

        successful_results = {
            k: v for k, v in self.results.items() if v.get("status") == "success"
        }

        if not successful_results:
            print("❌ 没有成功的实验结果")
            return

        print(f"✅ 成功完成实验: {len(successful_results)}/{len(self.results)}")
        print()

        # 按影响力统计排序
        results_by_mean_influence = sorted(
            successful_results.items(),
            key=lambda x: x[1]["statistics"]["mean_influence"],
            reverse=True,
        )

        print("🏆 按平均影响力排名:")
        print("-" * 60)
        for i, (model_name, result) in enumerate(results_by_mean_influence, 1):
            stats = result["statistics"]
            print(f"{i}. {model_name}")
            print(f"   平均影响力: {stats['mean_influence']:.6f}")
            print(f"   标准差: {stats['std_influence']:.6f}")
            print(f"   训练步数: {result['training_info']['total_steps']}")
            print()


def main():
    """主函数 - 运行BERT TIM情感分析实验"""

    print("🚀 启动BERT + TIM情感分析实验")
    print("=" * 50)

    # 显示可用的BERT模型选项
    model_configs = get_bert_model_configs()
    print("📋 可用的BERT模型选项:")
    for model_name, config in model_configs.items():
        print(f"  • {model_name}: {config['description']}")
    print()

    # 创建实验实例
    experiment = BertTimExperiment(
        dataset_name="imdb",  # IMDB情感分析数据集
        train_count=500,  # 训练样本数（实验用较小数据集）
        valid_count=100,  # 验证样本数
        test_count=100,  # 测试样本数
        random_state=42,
        output_dir="./bert_tim_results",
    )

    # 选择要测试的模型（按推荐顺序，仅支持DistilBERT）
    selected_models = [
        "distilbert-base-uncased",  # 基础模型
        "distilbert-base-cased",    # 区分大小写
        "distilbert-base-multilingual-cased",  # 最大的多语言模型
    ]

    print("🎯 选择测试的模型（DistilBERT系列，按参数规模）:")
    for model in selected_models:
        print(f"  • {model}: {model_configs[model]['parameters']} 参数")
    print()
    
    print("ℹ️  说明: OpenDataVal的BertClassifier基于DistilBERT架构")
    print("   只支持DistilBERT系列预训练模型，不支持标准BERT/RoBERTa")
    print("   实验在GPU服务器上运行")
    print()

    # 运行实验套件
    experiment.run_full_experiment_suite(selected_models)

    print("🎉 DistilBERT + TIM 实验配置完成！")
    print("   已修复模型兼容性和数据处理问题")


if __name__ == "__main__":
    main()
