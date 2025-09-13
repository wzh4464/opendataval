"""
TIM影响力计算模块

基于之前的TIM实现, 提供模块化的影响力计算功能, 支持BERT模型的数据价值评估。
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from opendataval.dataval.tim import TimInfluence
from opendataval.model import BertClassifier


class BertTimInfluenceCalculator:
    """BERT + TIM 影响力计算器"""

    def __init__(
        self,
        t1: int = 0,
        t2: Optional[int] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        regularization: float = 0.01,
        finite_diff_eps: float = 1e-5,
        random_state: int = 42,
    ):
        """
        初始化TIM影响力计算器

        Parameters:
        -----------
        t1 : int
            Time window start STEP (not epoch), default 0
        t2 : Optional[int]
            Time window end STEP (not epoch), None means to end (T)
        num_epochs : int
            训练轮数, 默认3
        batch_size : int
            批次大小, 默认8 (BERT需要较小batch size)
        regularization : float
            L2正则化参数, 默认0.01
        finite_diff_eps : float
            有限差分参数, 默认1e-5
        random_state : int
            随机种子, 默认42
        """
        self.t1 = t1
        self.t2 = t2
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.finite_diff_eps = finite_diff_eps
        self.random_state = random_state

        # TIM评估器
        self.tim_evaluator = None
        self.influence_scores = None

    def setup_tim_evaluator(self) -> TimInfluence:
        """设置TIM评估器"""
        print("⚙️  Setting up TIM evaluator")
        print(
            f"   Time window (steps): t1={self.t1}, t2={'T(end)' if self.t2 is None else self.t2}"
        )
        print(
            f"   Training config: epochs={self.num_epochs}, batch_size={self.batch_size}"
        )

        self.tim_evaluator = TimInfluence(
            start_step=self.t1,
            end_step=self.t2,
            time_window_type=(
                "full" if self.t1 == 0 and self.t2 is None else "custom_range"
            ),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            regularization=self.regularization,
            finite_diff_eps=self.finite_diff_eps,
            random_state=self.random_state,
        )

        return self.tim_evaluator

    def create_bert_wrapper(self, model: BertClassifier, attention_mask: torch.Tensor):
        """创建TIM兼容的BERT包装器"""

        class BertTimWrapper(torch.nn.Module):
            """包装BERT模型以兼容TIM的tensor输入格式"""

            def __init__(self, bert_model, attention_mask):
                super().__init__()
                self.bert_model = bert_model
                self.attention_mask = attention_mask.detach()

            def forward(self, input_ids):
                # TIM传递的是float tensor, 转换为token IDs
                batch_size = input_ids.shape[0]
                device = input_ids.device

                # 使用对应的attention mask片段
                mask = self.attention_mask[:batch_size].to(device)

                # 将float tensor转为long token IDs
                input_ids_long = input_ids.long()

                # 调用BERT并获取logits (不要softmax)
                if hasattr(self.bert_model, "classifier"):
                    # 获取分类器之前的hidden states
                    hidden_states = self.bert_model.bert(
                        input_ids_long, attention_mask=mask
                    )[0]
                    pooled_output = hidden_states[:, 0]  # [CLS] token

                    # 只通过linear层, 不要softmax
                    pre_linear = self.bert_model.classifier.pre_linear(pooled_output)
                    activated = self.bert_model.classifier.acti(pre_linear)
                    dropped = self.bert_model.classifier.dropout(activated)
                    logits = self.bert_model.classifier.linear(dropped)

                    return logits  # 返回raw logits而不是softmax输出
                else:
                    return self.bert_model(input_ids_long, attention_mask=mask)

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

        return BertTimWrapper(model, attention_mask)

    def compute_influence(self, model: BertClassifier, data: Dict) -> np.ndarray:
        """
        计算训练数据的影响力分数

        Parameters:
        -----------
        model : BertClassifier
            训练好的BERT模型
        data : Dict
            包含训练和验证数据的字典

        Returns:
        --------
        np.ndarray: 影响力分数数组
        """
        print("🔄 计算TIM影响力分数")
        print(f"   训练样本数: {len(data['y_train'])}")
        print(f"   验证样本数: {len(data['y_valid'])}")

        # 1. 设置TIM评估器
        if self.tim_evaluator is None:
            self.setup_tim_evaluator()

        # 2. 准备数据
        x_train, y_train = data["x_train"], data["y_train"]
        x_valid, y_valid = data["x_valid"], data["y_valid"]

        # 获取模型设备
        device = next(model.parameters()).device
        y_train = y_train.to(device)
        y_valid = y_valid.to(device)

        # 3. Tokenization
        print("   🔄 对文本数据进行tokenization...")
        train_dataset = model.tokenize(x_train)
        valid_dataset = model.tokenize(x_valid)

        # 获取tokenized数据
        train_input_ids = train_dataset.tensors[0].to(device)
        train_attention_mask = train_dataset.tensors[1].to(device)
        valid_input_ids = valid_dataset.tensors[0].to(device)
        _ = valid_dataset.tensors[1].to(device)

        # 4. 为TIM准备tensor输入 (转换为float以支持梯度计算)
        self.tim_evaluator.input_data(
            x_train=train_input_ids.float(),
            y_train=y_train,
            x_valid=valid_input_ids.float(),
            y_valid=y_valid,
        )

        # 5. 创建BERT包装器
        bert_wrapper = self.create_bert_wrapper(model, train_attention_mask)
        self.tim_evaluator.pred_model = bert_wrapper

        # 6. 训练并记录状态
        print("   🚀 开始TIM训练...")
        self.tim_evaluator.train_data_values(
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=2e-5,  # BERT推荐学习率
        )

        # 7. 计算影响力数据值
        print("   📊 计算数据影响力...")
        influence_scores = self.tim_evaluator.evaluate_data_values()

        self.influence_scores = influence_scores

        print("✅ 影响力计算完成")
        print(f"   影响力分数形状: {influence_scores.shape}")
        print(f"   平均影响力: {np.mean(influence_scores):.6f}")
        print(f"   标准差: {np.std(influence_scores):.6f}")

        return influence_scores

    def analyze_influence_scores(
        self,
        influence_scores: np.ndarray,
        y_train: torch.Tensor,
        noise_indices: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        分析影响力分数

        Parameters:
        -----------
        influence_scores : np.ndarray
            影响力分数
        y_train : torch.Tensor
            训练标签
        noise_indices : Optional[np.ndarray]
            噪声样本索引

        Returns:
        --------
        Dict: 分析结果
        """
        print("📊 分析影响力分数...")

        # 基础统计
        stats = {
            "mean_influence": float(np.mean(influence_scores)),
            "std_influence": float(np.std(influence_scores)),
            "min_influence": float(np.min(influence_scores)),
            "max_influence": float(np.max(influence_scores)),
            "total_samples": len(influence_scores),
        }

        # 按类别分析影响力
        if isinstance(y_train, torch.Tensor):
            y_train_np = y_train.cpu().numpy()
        else:
            y_train_np = y_train

        positive_indices = np.where(y_train_np == 1)[0]
        negative_indices = np.where(y_train_np == 0)[0]

        stats["class_analysis"] = {
            "positive_samples": {
                "count": len(positive_indices),
                "mean_influence": float(np.mean(influence_scores[positive_indices])),
                "std_influence": float(np.std(influence_scores[positive_indices])),
            },
            "negative_samples": {
                "count": len(negative_indices),
                "mean_influence": float(np.mean(influence_scores[negative_indices])),
                "std_influence": float(np.std(influence_scores[negative_indices])),
            },
        }

        # 找出最有影响力和最无影响力的样本
        top_k = min(20, len(influence_scores))
        most_influential_indices = np.argsort(influence_scores)[-top_k:][::-1]
        least_influential_indices = np.argsort(influence_scores)[:top_k]

        stats["ranking"] = {
            "most_influential": {
                "indices": most_influential_indices.tolist(),
                "values": influence_scores[most_influential_indices].tolist(),
            },
            "least_influential": {
                "indices": least_influential_indices.tolist(),
                "values": influence_scores[least_influential_indices].tolist(),
            },
        }

        # 如果有噪声信息, 分析噪声vs干净样本的影响力
        if noise_indices is not None:
            clean_indices = np.setdiff1d(
                np.arange(len(influence_scores)), noise_indices
            )

            stats["noise_analysis"] = {
                "noise_samples": {
                    "count": len(noise_indices),
                    "mean_influence": float(np.mean(influence_scores[noise_indices])),
                    "std_influence": float(np.std(influence_scores[noise_indices])),
                },
                "clean_samples": {
                    "count": len(clean_indices),
                    "mean_influence": float(np.mean(influence_scores[clean_indices])),
                    "std_influence": float(np.std(influence_scores[clean_indices])),
                },
            }

            # 噪声样本在影响力排名中的位置
            sorted_indices = np.argsort(influence_scores)
            noise_ranks = []
            for noise_idx in noise_indices:
                rank = np.where(sorted_indices == noise_idx)[0][0]
                rank_percentile = rank / len(influence_scores)
                noise_ranks.append(rank_percentile)

            stats["noise_analysis"]["noise_rank_percentiles"] = noise_ranks
            stats["noise_analysis"]["mean_noise_rank_percentile"] = float(
                np.mean(noise_ranks)
            )

        print("✅ 影响力分析完成")
        return stats

    def select_bottom_k_samples(
        self, influence_scores: np.ndarray, k_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择影响力最低的k%样本进行剪枝

        Parameters:
        -----------
        influence_scores : np.ndarray
            影响力分数
        k_ratio : float
            要剪枝的样本比例 (0.0-1.0)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]: (剪枝索引, 保留索引)
        """
        n_samples = len(influence_scores)
        k = int(n_samples * k_ratio)

        # 按影响力排序, 选择最低的k个
        sorted_indices = np.argsort(influence_scores)
        prune_indices = sorted_indices[:k]
        keep_indices = sorted_indices[k:]

        print("✂️  选择剪枝样本")
        print(f"   总样本数: {n_samples}")
        print(f"   剪枝比例: {k_ratio*100:.1f}%")
        print(f"   剪枝样本数: {k}")
        print(f"   保留样本数: {len(keep_indices)}")
        print(
            f"   剪枝样本影响力范围: [{influence_scores[prune_indices].min():.6f}, "
            f"{influence_scores[prune_indices].max():.6f}]"
        )

        return prune_indices, keep_indices

    def save_influence_results(
        self, influence_scores: np.ndarray, analysis: Dict, save_path: str
    ):
        """保存影响力计算结果"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存影响力分数
        np.save(save_path / "influence_scores.npy", influence_scores)

        # 保存分析结果
        import json

        with open(save_path / "influence_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"💾 影响力结果已保存到: {save_path}")


def create_tim_calculator(
    t1: int = 0,
    t2: Optional[int] = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    regularization: float = 0.01,
    finite_diff_eps: float = 1e-5,
    random_state: int = 42,
) -> BertTimInfluenceCalculator:
    """
    工厂函数: 创建TIM影响力计算器

    Parameters:
    -----------
    t1 : int
        时间窗口开始步骤, 默认0
    t2 : Optional[int]
        时间窗口结束步骤, 默认None (到结束)
    num_epochs : int
        训练轮数, 默认3
    batch_size : int
        批次大小, 默认8
    regularization : float
        L2正则化参数, 默认0.01
    finite_diff_eps : float
        有限差分参数, 默认1e-5
    random_state : int
        随机种子, 默认42

    Returns:
    --------
    BertTimInfluenceCalculator
        配置好的影响力计算器
    """
    return BertTimInfluenceCalculator(
        t1=t1,
        t2=t2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        regularization=regularization,
        finite_diff_eps=finite_diff_eps,
        random_state=random_state,
    )


if __name__ == "__main__":
    # 测试TIM计算器
    print("🧪 测试TIM影响力计算器")

    calculator = create_tim_calculator()
    print("✅ TIM计算器创建成功")
