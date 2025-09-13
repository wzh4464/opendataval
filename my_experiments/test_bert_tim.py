#!/usr/bin/env python3
"""
简化的BERT + TIM测试脚本
测试基本功能是否正常
"""

from bert_sentiment_analysis import BertTimExperiment, get_bert_model_configs


def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试BERT + TIM基本功能")
    print("=" * 50)

    # 显示可用模型
    configs = get_bert_model_configs()
    print("📋 可用模型:")
    for name, config in configs.items():
        print(f"  • {name}: {config['description']}")
    print()

    # 创建小型实验实例 (更少样本)
    experiment = BertTimExperiment(
        dataset_name="imdb",
        train_count=50,  # 非常小的数据集用于测试
        valid_count=20,
        test_count=20,
        random_state=42,
        output_dir="./test_results",
    )

    print("🔄 测试数据加载...")
    try:
        data = experiment.prepare_data()
        print("✅ 数据加载成功")
        x_train, y_train, x_valid, y_valid, x_test, y_test = data
        print(f"   训练数据类型: {type(x_train)}, 长度: {len(x_train)}")
        print(f"   标签类型: {type(y_train)}, 形状: {y_train.shape}")

        # 测试单个模型创建 (不运行完整实验)
        print("\n🤖 测试模型创建...")
        model_config = configs["distilbert-base-uncased"]
        model = experiment.create_bert_model(model_config)
        print("✅ 模型创建成功")
        print(f"   模型类型: {type(model)}")

        print("\n⚙️ 测试TIM评估器设置...")
        tim_evaluator = experiment.setup_tim_evaluator(
            t1=0, t2=None, num_epochs=1, batch_size=4
        )
        print("✅ TIM评估器创建成功")
        print(f"   评估器类型: {type(tim_evaluator)}")

        print("\n📊 测试TIM数据输入...")
        # 测试tokenization和数据转换
        print("   🔄 对文本数据进行tokenization...")

        train_dataset = model.tokenize(x_train)
        valid_dataset = model.tokenize(x_valid)

        # 获取tokenized的tensor数据
        train_input_ids = train_dataset.tensors[0]
        train_attention_mask = train_dataset.tensors[1]
        valid_input_ids = valid_dataset.tensors[0]
        _ = valid_dataset.tensors[1]

        print(f"   Tokenized数据形状: {train_input_ids.shape}")

        # 为TIM创建tensor输入
        tim_evaluator.input_data(
            x_train=train_input_ids,
            y_train=y_train,
            x_valid=valid_input_ids,
            y_valid=y_valid,
        )
        print("✅ TIM数据输入成功")
        print(f"   TIM训练样本数: {tim_evaluator.num_points}")

        # 创建BERT包装器
        import torch

        class BertTimWrapper(torch.nn.Module):
            def __init__(self, bert_model, attention_mask):
                super().__init__()
                self.bert_model = bert_model
                self.attention_mask = attention_mask

            def forward(self, input_ids):
                batch_size = input_ids.shape[0]
                mask = self.attention_mask[:batch_size]
                return self.bert_model(input_ids, attention_mask=mask)

            def predict(self, input_ids):
                return self.forward(input_ids)

            def parameters(self):
                return self.bert_model.parameters()

            def named_parameters(self):
                return self.bert_model.named_parameters()

            def zero_grad(self):
                return self.bert_model.zero_grad()

        bert_wrapper = BertTimWrapper(model, train_attention_mask)
        tim_evaluator.pred_model = bert_wrapper
        print("✅ TIM BERT包装器设置成功")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 基本功能测试通过! ")
        print("现在可以运行完整实验: python my_experiments/bert_sentiment_analysis.py")
    else:
        print("\n💥 基本功能测试失败")
        print("需要进一步调试")
