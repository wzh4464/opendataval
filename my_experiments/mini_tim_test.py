#!/usr/bin/env python3
"""
最小的端到端TIM测试 - 包括实际训练
"""

from bert_sentiment_analysis import BertTimExperiment

def mini_tim_test():
    """运行最小的端到端TIM测试"""
    print("🧪 最小端到端TIM测试")
    print("="*40)
    
    # 极小的数据集
    experiment = BertTimExperiment(
        dataset_name="imdb",
        train_count=10,   # 极小
        valid_count=5,
        test_count=5,
        random_state=42,
        output_dir="./mini_test_results"
    )
    
    # 只测试最小的模型
    selected_models = ["distilbert-base-uncased"]
    
    # 极简配置
    tim_config = {
        't1': 0,
        't2': None,
        'num_epochs': 1,  # 只训练1轮
        'batch_size': 2   # 极小batch
    }
    
    try:
        print("🔄 准备数据...")
        data = experiment.prepare_data()
        
        print("🚀 运行mini TIM实验...")
        result = experiment.run_single_experiment(
            model_name="distilbert-base-uncased",
            model_config={
                "pretrained_model_name": "distilbert-base-uncased",
                "description": "Mini test"
            },
            data=data,
            tim_config=tim_config
        )
        
        if result['status'] == 'success':
            print("🎉 Mini TIM测试成功！")
            print(f"   影响力分数数量: {len(result['data_values'])}")
            print(f"   平均影响力: {result['statistics']['mean_influence']:.6f}")
            return True
        else:
            print(f"❌ 测试失败: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"💥 异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = mini_tim_test()
    if success:
        print("\n✅ 准备好运行完整实验了！")
    else:
        print("\n❌ 仍有问题需要修复")