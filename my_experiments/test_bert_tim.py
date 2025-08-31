#!/usr/bin/env python3
"""
简化的BERT + TIM测试脚本
测试基本功能是否正常
"""

from bert_sentiment_analysis import BertTimExperiment, get_bert_model_configs

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试BERT + TIM基本功能")
    print("="*50)
    
    # 显示可用模型
    configs = get_bert_model_configs()
    print("📋 可用模型:")
    for name, config in configs.items():
        print(f"  • {name}: {config['description']}")
    print()
    
    # 创建小型实验实例（更少样本）
    experiment = BertTimExperiment(
        dataset_name="imdb",
        train_count=50,    # 非常小的数据集用于测试
        valid_count=20,
        test_count=20,
        random_state=42,
        output_dir="./test_results"
    )
    
    print("🔄 测试数据加载...")
    try:
        data = experiment.prepare_data()
        print("✅ 数据加载成功")
        x_train, y_train, x_valid, y_valid, x_test, y_test = data
        print(f"   训练数据类型: {type(x_train)}, 长度: {len(x_train)}")
        print(f"   标签类型: {type(y_train)}, 形状: {y_train.shape}")
        
        # 测试单个模型创建（不运行完整实验）
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
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 基本功能测试通过！")
        print("现在可以运行完整实验: python my_experiments/bert_sentiment_analysis.py")
    else:
        print("\n💥 基本功能测试失败")
        print("需要进一步调试")