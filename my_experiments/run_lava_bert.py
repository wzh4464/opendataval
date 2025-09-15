#!/usr/bin/env python3
"""
LAVA + BERT 情感分析 CLI

功能
----
- 运行 LAVA 数据价值评估, 使用 BERT 作为嵌入模型 (默认输出CLS pooled embedding) 。
- 支持命令行参数调整数据规模、epoch、学习率、batch size、设备等。
- 支持按阶段 [0, t] (t <= T) 计算影响力:
  对每个 t, 使用 t 作为 fine-tune 的总 epoch 数来训练/微调 BERT, 然后计算一次 LAVA。

注意
----
- LAVA 自身不依赖时间窗口; 这里的阶段是通过“将 fine-tune 轮数设为 t”来得到不同阶段的嵌入, 再计算 LAVA。
- 若 t=0, 则直接使用预训练 BERT (不微调) 。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from opendataval.dataloader import DataFetcher
from opendataval.dataval.lava import LavaEvaluator
from opendataval.model import BertClassifier


def select_device(pref: str = "auto", gpu: Optional[int] = None) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        if gpu is not None and 0 <= gpu < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        if gpu is not None and 0 <= gpu < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


class BertEmbeddingWrapper(torch.nn.Module):
    """包装 BertClassifier, 使其 predict 返回可用作 LAVA 特征的向量。

    模式:
    - pooled: 返回 CLS pooled embedding (首 token 对应的隐藏状态) 。
    - logits: 返回分类器线性层输出 (未 softmax 的 logits) 。
    - probs: 返回分类概率 (BertClassifier.predict 的默认输出) 。
    """

    def __init__(self, bert_model: BertClassifier, mode: str = "pooled"):
        super().__init__()
        assert mode in {"pooled", "logits", "probs"}
        self.bert_model = bert_model
        self.mode = mode

    @property
    def device(self) -> torch.device:
        return self.bert_model.bert.device

    def predict(self, x_dataset) -> torch.Tensor:
        # x_dataset 是 ListDataset[str] 或兼容的 Dataset
        # 统一走 tokenization, 然后按需要返回特征
        if len(x_dataset) == 0:
            return torch.zeros(0, 1, device=self.device)

        # 将文本转为 input_ids & attention_mask
        bert_inputs = self.bert_model.tokenize(x_dataset)
        input_ids, attention_mask = [t.to(self.device) for t in bert_inputs.tensors]

        with torch.no_grad():
            if self.mode == "pooled":
                hidden_states = self.bert_model.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
                pooled = hidden_states[:, 0]
                return pooled
            elif self.mode == "logits":
                # 复用分类器, 但截断 softmax: 只取线性层 logits
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


def to_class_indices(y: torch.Tensor) -> torch.Tensor:
    """one-hot/soft 标签 -> 类别索引 (LongTensor)。"""
    if y.ndim > 1 and y.shape[1] > 1:
        return torch.argmax(y, dim=1).long()
    # 已是一维, 确保 long
    return y.long().view(-1)


def run_stage_lava(
    dataset: str,
    train_count: int,
    valid_count: int,
    test_count: int,
    seed: int,
    pretrained_model_name: str,
    device: torch.device,
    stage_epochs: int,
    finetune_batch_size: int,
    finetune_lr: float,
    embedding_mode: str,
) -> np.ndarray:
    """在指定 stage (t=stage_epochs) 下运行一次 LAVA, 返回 data values。"""
    # 加载数据 (不加噪声)
    fetcher = DataFetcher.setup(
        dataset_name=dataset,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        random_state=seed,
        add_noise=None,
    )
    x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints

    # 构建 BERT 模型
    num_classes = fetcher.label_dim[0]
    bert = BertClassifier(
        pretrained_model_name=pretrained_model_name, num_classes=num_classes
    ).to(device)

    # 按 t 进行微调 (t=0则跳过)
    if stage_epochs > 0:
        y_train_idx = to_class_indices(y_train)
        bert.fit(
            x_train,
            y_train_idx,
            batch_size=finetune_batch_size,
            epochs=stage_epochs,
            lr=finetune_lr,
        )

    # 包装用于 LAVA 的嵌入模型
    emb_model = BertEmbeddingWrapper(bert, mode=embedding_mode)

    lava = LavaEvaluator(device=device, embedding_model=emb_model, random_state=seed)
    lava.input_data(x_train, y_train, x_valid, y_valid)
    lava.train_data_values()
    dv = lava.evaluate_data_values()
    return dv


def parse_stages(arg: str, T: int) -> List[int]:
    if arg == "all":
        return list(range(T + 1))
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    stages: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except Exception as e:
            raise ValueError(f"Invalid stage '{p}', must be int or 'all'") from e
        if v < 0 or v > T:
            raise ValueError(f"Stage {v} out of range [0, {T}]")
        stages.append(v)
    return sorted(set(stages))


def main():
    parser = argparse.ArgumentParser(description="LAVA + BERT Sentiment CLI")
    # 数据集与规模
    parser.add_argument(
        "--dataset", default="imdb", help="dataset name (default: imdb)"
    )
    parser.add_argument("--train-count", type=int, default=16384)
    parser.add_argument("--valid-count", type=int, default=2048)
    parser.add_argument("--test-count", type=int, default=2048)

    # 训练/阶段
    parser.add_argument(
        "--epochs", type=int, default=10, help="T: max fine-tune epochs"
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="stage list like '0,2,5' or 'all' for [0..T]",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="fine-tune batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="fine-tune learning rate"
    )

    # 模型与设备
    parser.add_argument(
        "--pretrained-model",
        default="distilbert-base-uncased",
        help="HF pretrained model id for DistilBERT",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["pooled", "logits", "probs"],
        default="pooled",
        help="feature type for LAVA (default: pooled)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="device preference",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use when CUDA is available (e.g., --gpu 2)",
    )
    parser.add_argument("--seed", type=int, default=42)

    # 输出
    parser.add_argument(
        "--output-dir",
        default="results/lava_bert_cli",
        help="directory to save outputs",
    )
    parser.add_argument(
        "--save-prefix",
        default="lava",
        help="filename prefix for saved arrays",
    )

    args = parser.parse_args()

    device = select_device(args.device, args.gpu)
    # If CUDA with explicit GPU id, set it
    if device.type == "cuda" and args.gpu is not None:
        try:
            torch.cuda.set_device(device)
        except Exception:
            # Fallback: ignore if cannot set
            pass
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = parse_stages(args.stages, args.epochs)

    print("🔧 Config:")
    print(f"  dataset         : {args.dataset}")
    print(
        f"  splits          : train={args.train_count}, valid={args.valid_count}, test={args.test_count}"
    )
    print(f"  epochs (T)      : {args.epochs}")
    print(f"  stages          : {stages}")
    print(f"  lr, batch_size  : {args.lr}, {args.batch_size}")
    print(f"  pretrained      : {args.pretrained_model}")
    print(f"  embedding_mode  : {args.embedding_mode}")
    gpu_descr = f", gpu={args.gpu}" if (device.type == "cuda" and args.gpu is not None) else ""
    print(f"  device, seed    : {device}{gpu_descr}, {args.seed}")
    print(f"  output_dir      : {out_dir}")

    all_stats = {}

    for t in stages:
        print("=" * 60)
        print(f"🚀 Stage [0, {t}] → fine-tune epochs = {t}")
        dv = run_stage_lava(
            dataset=args.dataset,
            train_count=args.train_count,
            valid_count=args.valid_count,
            test_count=args.test_count,
            seed=args.seed,
            pretrained_model_name=args.pretrained_model,
            device=device,
            stage_epochs=t,
            finetune_batch_size=args.batch_size,
            finetune_lr=args.lr,
            embedding_mode=args.embedding_mode,
        )

        arr = np.asarray(dv, dtype=np.float64)
        stats = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": int(arr.size),
        }
        all_stats[f"stage_{t}"] = stats

        npy_path = (
            out_dir / f"{args.save_prefix}_{args.dataset}_t{t}_seed{args.seed}.npy"
        )
        np.save(npy_path, arr)
        print(f"✅ Saved stage {t} data values → {npy_path}")
        print(f"   stats: {stats}")

    summary = {
        "dataset": args.dataset,
        "train_count": args.train_count,
        "valid_count": args.valid_count,
        "test_count": args.test_count,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "pretrained_model": args.pretrained_model,
        "embedding_mode": args.embedding_mode,
        "device": str(device),
        "seed": args.seed,
        "stages": stages,
        "stats": all_stats,
    }

    summary_path = (
        out_dir / f"{args.save_prefix}_{args.dataset}_summary_seed{args.seed}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"💾 Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
