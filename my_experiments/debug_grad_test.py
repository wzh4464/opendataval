#!/usr/bin/env python3
"""
调试梯度问题
"""
import torch
from bert_sentiment_analysis import BertTimExperiment


def debug_gradient_issue():
    """调试梯度问题"""
    print("🔍 调试梯度问题")

    experiment = BertTimExperiment(
        dataset_name="imdb", train_count=5, valid_count=3, test_count=2, random_state=42
    )

    # 准备数据
    data = experiment.prepare_data()
    x_train, y_train, x_valid, y_valid, x_test, y_test = data

    # 创建模型 - 强制使用CPU避免MPS问题
    from bert_sentiment_analysis import get_bert_model_configs

    from opendataval.model import BertClassifier

    configs = get_bert_model_configs()
    model_config = configs["distilbert-base-uncased"]

    model = BertClassifier(
        pretrained_model_name=model_config["pretrained_model_name"],
        num_classes=2,
        dropout_rate=0.2,
        num_train_layers=2,
    )

    # 强制使用CPU
    device = torch.device("cpu")
    model = model.to(device)
    print(f"🤖 模型设备: {device}")

    print("🧪 测试BERT模型梯度...")

    # 测试基本BERT模型梯度
    model.train()

    # Tokenize数据
    train_dataset = model.tokenize(x_train)
    train_input_ids = train_dataset.tensors[0]
    train_attention_mask = train_dataset.tensors[1]

    print(f"Input IDs shape: {train_input_ids.shape}")
    print(f"Input IDs requires_grad: {train_input_ids.requires_grad}")

    # 测试模型前向传播
    print("\n🔄 测试模型前向传播...")
    with torch.enable_grad():
        outputs = model(train_input_ids[:2], train_attention_mask[:2])
        print(f"Outputs shape: {outputs.shape}")
        print(f"Outputs requires_grad: {outputs.requires_grad}")

        # 计算损失
        targets = y_train[:2]
        print(f"Targets shape: {targets.shape}")
        print(f"Targets: {targets}")

        # 如果是one-hot编码, 转换为索引
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            # one-hot to indices
            target_indices = torch.argmax(targets, dim=1)
            print(f"Target indices: {target_indices}")
            loss = torch.nn.functional.cross_entropy(outputs, target_indices)
        else:
            loss = torch.nn.functional.mse_loss(outputs, targets.float())

        print(f"Loss: {loss}")
        print(f"Loss requires_grad: {loss.requires_grad}")

        # 测试反向传播
        print("\n🔄 测试反向传播...")
        loss.backward()

        # 检查梯度
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                print(f"Parameter {name}: grad_norm = {param.grad.norm().item():.6f}")
            else:
                print(f"Parameter {name}: NO GRADIENT")

        print(f"\n✅ 总共 {grad_count} 个参数有梯度")

        if grad_count > 0:
            print("🎉 BERT模型梯度正常! ")
            return True
        else:
            print("❌ BERT模型没有梯度")
            return False


if __name__ == "__main__":
    debug_gradient_issue()
