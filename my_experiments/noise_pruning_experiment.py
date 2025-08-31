#!/usr/bin/env python3
"""
噪声数据剪枝实验主脚本

完整的BERT + TIM噪声数据影响力分析和剪枝实验流程：
1. 加载干净数据
2. 注入30%标签噪声 
3. 训练BERT模型，记录损失曲线和[0,T]影响力分数
4. 移除影响力最低的30%样本
5. 用相同初始化重新训练
6. 对比分析性能和收敛性
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch
import numpy as np
import time

# 导入我们的模块
from my_experiments.noise_data_module import create_noise_processor
from my_experiments.bert_training_module import create_bert_trainer
from my_experiments.tim_influence_module import create_tim_calculator
from my_experiments.visualization_module import create_visualizer


class NoisePruningExperiment:
    """噪声数据剪枝实验类"""
    
    def __init__(
        self,
        # 数据配置
        dataset_name: str = "imdb",
        train_count: int = 1000,
        valid_count: int = 200,
        test_count: int = 200,
        noise_rate: float = 0.3,
        
        # 模型配置
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        num_train_layers: int = 2,
        
        # 训练配置
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        
        # TIM配置
        tim_epochs: int = 3,
        tim_batch_size: int = 8,
        t1: int = 0,
        t2: Optional[int] = None,
        regularization: float = 0.01,
        finite_diff_eps: float = 1e-5,
        
        # 剪枝配置
        prune_ratio: float = 0.3,
        
        # 实验配置
        random_state: int = 42,
        device: str = "auto",
        output_dir: str = "./noise_pruning_results",
        save_plots: bool = True
    ):
        """
        初始化噪声剪枝实验
        
        Parameters:
        -----------
        dataset_name : str
            数据集名称，默认"imdb"
        train_count : int
            训练样本数，默认1000
        valid_count : int
            验证样本数，默认200
        test_count : int
            测试样本数，默认200
        noise_rate : float
            标签噪声比例，默认0.3 (30%)
        model_name : str
            BERT模型名称，默认"distilbert-base-uncased"
        num_classes : int
            分类类别数，默认2
        dropout_rate : float
            Dropout率，默认0.2
        num_train_layers : int
            微调层数，默认2
        epochs : int
            训练轮数，默认5
        batch_size : int
            批次大小，默认16
        learning_rate : float
            学习率，默认2e-5
        tim_epochs : int
            TIM训练轮数，默认3
        tim_batch_size : int
            TIM批次大小，默认8
        t1 : int
            TIM时间窗口开始，默认0
        t2 : Optional[int]
            TIM时间窗口结束，默认None (到结束)
        regularization : float
            L2正则化，默认0.01
        finite_diff_eps : float
            有限差分参数，默认1e-5
        prune_ratio : float
            剪枝比例，默认0.3 (30%)
        random_state : int
            随机种子，默认42
        device : str
            计算设备，默认"auto"
        output_dir : str
            结果保存目录，默认"./noise_pruning_results"
        save_plots : bool
            是否保存图表，默认True
        """
        
        # 保存所有配置
        self.config = {
            'dataset_name': dataset_name,
            'train_count': train_count,
            'valid_count': valid_count,
            'test_count': test_count,
            'noise_rate': noise_rate,
            'model_name': model_name,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
            'num_train_layers': num_train_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'tim_epochs': tim_epochs,
            'tim_batch_size': tim_batch_size,
            't1': t1,
            't2': t2,
            'regularization': regularization,
            'finite_diff_eps': finite_diff_eps,
            'prune_ratio': prune_ratio,
            'random_state': random_state,
            'device': device,
            'output_dir': output_dir,
            'save_plots': save_plots
        }
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.data_processor = None
        self.bert_trainer = None
        self.tim_calculator = None
        self.visualizer = None
        
        # 实验结果存储
        self.results = {
            'config': self.config,
            'start_time': None,
            'end_time': None,
            'status': 'initialized',
            'data_stats': {},
            'original_training': {},
            'influence_analysis': {},
            'pruning_analysis': {},
            'pruned_training': {},
            'comparative_analysis': {},
            'error_log': []
        }
        
    def setup_components(self):
        """设置实验组件"""
        print("⚙️  设置实验组件...")
        
        try:
            # 1. 数据处理器
            self.data_processor = create_noise_processor(
                dataset_name=self.config['dataset_name'],
                train_count=self.config['train_count'],
                valid_count=self.config['valid_count'],
                test_count=self.config['test_count'],
                noise_rate=self.config['noise_rate'],
                random_state=self.config['random_state']
            )
            
            # 2. BERT训练器
            self.bert_trainer = create_bert_trainer(
                model_name=self.config['model_name'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout_rate'],
                num_train_layers=self.config['num_train_layers'],
                device=self.config['device'],
                random_state=self.config['random_state']
            )
            
            # 3. TIM计算器
            self.tim_calculator = create_tim_calculator(
                t1=self.config['t1'],
                t2=self.config['t2'],
                num_epochs=self.config['tim_epochs'],
                batch_size=self.config['tim_batch_size'],
                regularization=self.config['regularization'],
                finite_diff_eps=self.config['finite_diff_eps'],
                random_state=self.config['random_state']
            )
            
            # 4. 可视化器
            if self.config['save_plots']:
                self.visualizer = create_visualizer(
                    save_dir=str(self.output_dir / "plots")
                )
            
            print("✅ 实验组件设置完成")
            
        except Exception as e:
            error_msg = f"组件设置失败: {e}"
            print(f"❌ {error_msg}")
            self.results['error_log'].append(error_msg)
            raise
            
    def prepare_data(self) -> Dict:
        """准备实验数据"""
        print("🔄 准备实验数据...")
        
        try:
            # 1. 加载干净数据
            clean_data = self.data_processor.load_clean_data()
            
            # 2. 注入标签噪声
            noisy_data, noise_indices = self.data_processor.inject_label_noise()
            
            # 3. 获取噪声统计
            noise_stats = self.data_processor.get_noise_statistics()
            
            # 保存数据统计
            self.results['data_stats'] = {
                'clean_data_info': {
                    'train_samples': len(clean_data['y_train']),
                    'valid_samples': len(clean_data['y_valid']),
                    'test_samples': len(clean_data['y_test'])
                },
                'noise_info': noise_stats
            }
            
            print("✅ 数据准备完成")
            print(f"   总训练样本: {len(noisy_data['y_train'])}")
            print(f"   噪声样本: {len(noise_indices)} ({len(noise_indices)/len(noisy_data['y_train'])*100:.1f}%)") 
            
            return noisy_data, noise_indices
            
        except Exception as e:
            error_msg = f"数据准备失败: {e}"
            print(f"❌ {error_msg}")
            self.results['error_log'].append(error_msg)
            raise
            
    def train_original_model(self, data: Dict) -> Dict:
        """训练原始（含噪声）模型"""
        print("🚀 训练原始（含噪声）模型...")
        
        try:
            # 创建模型
            original_model = self.bert_trainer.create_model()
            
            # 保存初始状态用于后续相同初始化
            initial_state = self.bert_trainer.save_model_state(original_model)
            
            # 训练模型
            training_history = self.bert_trainer.train_model(
                model=original_model,
                data=data,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                learning_rate=self.config['learning_rate']
            )
            
            # 保存训练历史
            self.bert_trainer.save_training_history(
                training_history, 
                str(self.output_dir / "original_training")
            )
            
            # 保存结果
            self.results['original_training'] = {
                'history': training_history,
                'model_path': str(self.output_dir / "original_model.pt"),
                'initial_state_path': str(self.output_dir / "initial_state.pt")
            }
            
            # 保存模型和初始状态
            torch.save(original_model.state_dict(), self.output_dir / "original_model.pt")
            torch.save(initial_state, self.output_dir / "initial_state.pt")
            
            print("✅ 原始模型训练完成")
            
            return original_model, initial_state, training_history
            
        except Exception as e:
            error_msg = f"原始模型训练失败: {e}"
            print(f"❌ {error_msg}")
            self.results['error_log'].append(error_msg)
            raise
            
    def compute_influence_scores(self, model, data: Dict) -> np.ndarray:
        """计算TIM影响力分数"""
        print("📊 计算TIM影响力分数...")
        
        try:
            # 计算影响力
            influence_scores = self.tim_calculator.compute_influence(model, data)
            
            # 分析影响力分数
            noise_indices = self.data_processor.noise_indices
            analysis = self.tim_calculator.analyze_influence_scores(
                influence_scores, 
                data['y_train'], 
                noise_indices
            )
            
            # 保存影响力结果
            self.tim_calculator.save_influence_results(
                influence_scores, 
                analysis, 
                str(self.output_dir / "influence_analysis")
            )
            
            # 保存结果
            self.results['influence_analysis'] = {
                'scores': influence_scores.tolist(),
                'statistics': analysis,
                'save_path': str(self.output_dir / "influence_analysis")
            }
            
            print("✅ 影响力计算完成")
            
            return influence_scores, analysis
            
        except Exception as e:
            error_msg = f"影响力计算失败: {e}"
            print(f"❌ {error_msg}")
            self.results['error_log'].append(error_msg)
            raise
            
    def prune_data(self, influence_scores: np.ndarray, original_data: Dict) -> Dict:
        """根据影响力分数剪枝数据"""
        print("✂️  根据影响力剪枝数据...")
        
        try:
            # 选择要剪枝的样本
            prune_indices, keep_indices = self.tim_calculator.select_bottom_k_samples(
                influence_scores, 
                k_ratio=self.config['prune_ratio']
            )
            
            # 执行数据剪枝
            pruned_data, remaining_indices = self.data_processor.prune_data_by_indices(prune_indices)
            
            # 保存剪枝数据
            self.data_processor.save_data(pruned_data, str(self.output_dir / "pruned_data"))
            
            # 分析剪枝效果
            noise_indices = self.data_processor.noise_indices
            pruned_noise = np.intersect1d(prune_indices, noise_indices)
            noise_recall = len(pruned_noise) / len(noise_indices) if len(noise_indices) > 0 else 0
            noise_precision = len(pruned_noise) / len(prune_indices) if len(prune_indices) > 0 else 0
            
            pruning_analysis = {
                'prune_indices': prune_indices.tolist(),
                'keep_indices': keep_indices.tolist(),
                'remaining_indices': remaining_indices.tolist(),
                'original_samples': len(original_data['y_train']),
                'pruned_samples': len(prune_indices),
                'remaining_samples': len(remaining_indices),
                'prune_ratio': len(prune_indices) / len(original_data['y_train']),
                'noise_detection': {
                    'total_noise': len(noise_indices),
                    'captured_noise': len(pruned_noise),
                    'noise_recall': noise_recall,
                    'noise_precision': noise_precision
                }
            }
            
            # 保存结果
            self.results['pruning_analysis'] = pruning_analysis
            
            print("✅ 数据剪枝完成")
            print(f"   剪枝样本: {len(prune_indices)} ({len(prune_indices)/len(original_data['y_train'])*100:.1f}%)")
            print(f"   噪声捕获: {len(pruned_noise)}/{len(noise_indices)} (召回率: {noise_recall:.2%})")
            
            return pruned_data, prune_indices, keep_indices
            
        except Exception as e:
            error_msg = f"数据剪枝失败: {e}"
            print(f"❌ {error_msg}")
            self.results['error_log'].append(error_msg)
            raise
            
    def train_pruned_model(self, pruned_data: Dict, initial_state: Dict) -> Dict:
        """用相同初始化训练剪枝后的模型"""
        print("🚀 训练剪枝后模型（相同初始化）...")
        
        try:
            # 创建新模型并加载相同的初始状态
            pruned_model = self.bert_trainer.create_model()
            pruned_model = self.bert_trainer.load_model_state(pruned_model, initial_state)
            
            # 训练模型
            pruned_history = self.bert_trainer.train_model(
                model=pruned_model,
                data=pruned_data,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                learning_rate=self.config['learning_rate']
            )
            
            # 保存训练历史
            self.bert_trainer.save_training_history(
                pruned_history,
                str(self.output_dir / "pruned_training")
            )
            
            # 保存结果
            self.results['pruned_training'] = {
                'history': pruned_history,
                'model_path': str(self.output_dir / "pruned_model.pt")
            }
            
            # 保存模型
            torch.save(pruned_model.state_dict(), self.output_dir / "pruned_model.pt")
            
            print("✅ 剪枝后模型训练完成")
            
            return pruned_model, pruned_history
            
        except Exception as e:
            error_msg = f"剪枝后模型训练失败: {e}"
            print(f"❌ {error_msg}")
            self.results['error_log'].append(error_msg)
            raise
            
    def create_visualizations(
        self,
        original_history: Dict,
        pruned_history: Dict,
        influence_scores: np.ndarray,
        influence_analysis: Dict,
        prune_indices: np.ndarray,
        keep_indices: np.ndarray
    ):
        """创建可视化图表"""
        if not self.config['save_plots'] or self.visualizer is None:
            print("⏭️  跳过可视化（save_plots=False）")
            return
            
        print("🎨 创建可视化图表...")
        
        try:
            noise_indices = self.data_processor.noise_indices
            
            # 1. 原始训练曲线
            self.visualizer.plot_training_curves(
                original_history,
                title="原始（含噪声）训练曲线",
                save_name="original_training_curves.png"
            )
            
            # 2. 剪枝后训练曲线
            self.visualizer.plot_training_curves(
                pruned_history,
                title="剪枝后训练曲线", 
                save_name="pruned_training_curves.png"
            )
            
            # 3. 影响力分布
            self.visualizer.plot_influence_distribution(
                influence_scores,
                noise_indices,
                title="TIM影响力分数分布",
                save_name="influence_distribution.png"
            )
            
            # 4. 剪枝分析
            self.visualizer.plot_pruning_analysis(
                influence_scores,
                prune_indices,
                keep_indices,
                noise_indices,
                title="数据剪枝分析",
                save_name="pruning_analysis.png"
            )
            
            # 5. 对比分析
            self.visualizer.plot_comparative_analysis(
                original_history,
                pruned_history,
                influence_analysis,
                title="剪枝前后对比分析",
                save_name="comparative_analysis.png"
            )
            
            print("✅ 可视化图表创建完成")
            
        except Exception as e:
            error_msg = f"可视化创建失败: {e}"
            print(f"⚠️  {error_msg}")
            self.results['error_log'].append(error_msg)
            
    def run_complete_experiment(self) -> Dict:
        """运行完整的噪声剪枝实验"""
        print("🧪 开始完整的噪声剪枝实验")
        print("=" * 60)
        
        self.results['start_time'] = time.time()
        self.results['status'] = 'running'
        
        try:
            # 1. 设置组件
            self.setup_components()
            
            # 2. 准备数据
            noisy_data, noise_indices = self.prepare_data()
            
            # 3. 训练原始模型
            original_model, initial_state, original_history = self.train_original_model(noisy_data)
            
            # 4. 计算影响力分数
            influence_scores, influence_analysis = self.compute_influence_scores(original_model, noisy_data)
            
            # 5. 剪枝数据
            pruned_data, prune_indices, keep_indices = self.prune_data(influence_scores, noisy_data)
            
            # 6. 训练剪枝后模型
            pruned_model, pruned_history = self.train_pruned_model(pruned_data, initial_state)
            
            # 7. 创建可视化
            self.create_visualizations(
                original_history, pruned_history, influence_scores, 
                influence_analysis, prune_indices, keep_indices
            )
            
            # 8. 生成对比分析
            self.generate_comparative_analysis(original_history, pruned_history)
            
            # 9. 保存实验结果
            self.save_experiment_results()
            
            self.results['status'] = 'success'
            self.results['end_time'] = time.time()
            
            print("🎉 实验完成！")
            self.print_experiment_summary()
            
            return self.results
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['end_time'] = time.time()
            error_msg = f"实验失败: {e}"
            print(f"💥 {error_msg}")
            self.results['error_log'].append(error_msg)
            traceback.print_exc()
            return self.results
            
    def generate_comparative_analysis(self, original_history: Dict, pruned_history: Dict):
        """生成对比分析结果"""
        print("📈 生成对比分析...")
        
        try:
            orig_final = original_history.get('final_performance', {})
            pruned_final = pruned_history.get('final_performance', {})
            
            comparative_analysis = {
                'performance_improvement': {
                    'train_accuracy': {
                        'original': orig_final.get('train_accuracy', 0),
                        'pruned': pruned_final.get('train_accuracy', 0),
                        'improvement': pruned_final.get('train_accuracy', 0) - orig_final.get('train_accuracy', 0)
                    },
                    'valid_accuracy': {
                        'original': orig_final.get('valid_accuracy', 0),
                        'pruned': pruned_final.get('valid_accuracy', 0),
                        'improvement': pruned_final.get('valid_accuracy', 0) - orig_final.get('valid_accuracy', 0)
                    },
                    'train_loss': {
                        'original': orig_final.get('train_loss', 0),
                        'pruned': pruned_final.get('train_loss', 0),
                        'improvement': orig_final.get('train_loss', 0) - pruned_final.get('train_loss', 0)  # 损失越低越好
                    }
                },
                'training_efficiency': {
                    'original_time': orig_final.get('total_time', 0),
                    'pruned_time': pruned_final.get('total_time', 0),
                    'time_saving': orig_final.get('total_time', 0) - pruned_final.get('total_time', 0)
                },
                'convergence_analysis': {
                    'original_final_epoch_loss': original_history.get('train_loss', [])[-1] if original_history.get('train_loss') else 0,
                    'pruned_final_epoch_loss': pruned_history.get('train_loss', [])[-1] if pruned_history.get('train_loss') else 0
                }
            }
            
            self.results['comparative_analysis'] = comparative_analysis
            
            print("✅ 对比分析生成完成")
            
        except Exception as e:
            error_msg = f"对比分析生成失败: {e}"
            print(f"⚠️  {error_msg}")
            self.results['error_log'].append(error_msg)
            
    def save_experiment_results(self):
        """保存完整的实验结果"""
        print("💾 保存实验结果...")
        
        try:
            # 保存主结果文件
            with open(self.output_dir / "experiment_results.json", 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
                
            # 保存配置文件
            with open(self.output_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
                
            print(f"✅ 实验结果已保存到: {self.output_dir}")
            
        except Exception as e:
            error_msg = f"结果保存失败: {e}"
            print(f"⚠️  {error_msg}")
            self.results['error_log'].append(error_msg)
            
    def print_experiment_summary(self):
        """打印实验摘要"""
        print("\\n" + "=" * 60)
        print("📊 实验摘要")
        print("=" * 60)
        
        # 基本信息
        print(f"📁 结果目录: {self.output_dir}")
        print(f"⏱️  总耗时: {self.results.get('end_time', 0) - self.results.get('start_time', 0):.1f}秒")
        print(f"🎯 实验状态: {self.results.get('status', 'unknown')}")
        
        # 数据信息
        data_stats = self.results.get('data_stats', {})
        if data_stats:
            noise_info = data_stats.get('noise_info', {})
            print(f"\\n📊 数据统计:")
            print(f"   训练样本: {noise_info.get('total_samples', 'N/A')}")
            print(f"   噪声样本: {noise_info.get('noise_count', 'N/A')} ({noise_info.get('noise_rate', 0)*100:.1f}%)")
        
        # 性能对比
        comparative = self.results.get('comparative_analysis', {})
        if comparative:
            perf = comparative.get('performance_improvement', {})
            print(f"\\n🚀 性能改进:")
            
            train_acc = perf.get('train_accuracy', {})
            print(f"   训练准确率: {train_acc.get('original', 0):.3f} → {train_acc.get('pruned', 0):.3f} ({train_acc.get('improvement', 0):+.3f})")
            
            valid_acc = perf.get('valid_accuracy', {})
            print(f"   验证准确率: {valid_acc.get('original', 0):.3f} → {valid_acc.get('pruned', 0):.3f} ({valid_acc.get('improvement', 0):+.3f})")
            
            train_loss = perf.get('train_loss', {})
            print(f"   训练损失: {train_loss.get('original', 0):.3f} → {train_loss.get('pruned', 0):.3f} ({train_loss.get('improvement', 0):+.3f})")
        
        # 噪声检测效果
        pruning = self.results.get('pruning_analysis', {})
        if pruning:
            noise_det = pruning.get('noise_detection', {})
            print(f"\\n🔍 噪声检测效果:")
            print(f"   噪声召回率: {noise_det.get('noise_recall', 0)*100:.1f}%")
            print(f"   剪枝精确率: {noise_det.get('noise_precision', 0)*100:.1f}%")
        
        # 错误日志
        if self.results.get('error_log'):
            print(f"\\n⚠️  错误日志:")
            for error in self.results['error_log']:
                print(f"   • {error}")
        
        print("=" * 60)


def create_experiment(
    # Data configuration
    dataset_name: str = "imdb",
    train_count: int = 5000,  # Large-scale data for server
    valid_count: int = 1000,
    test_count: int = 1000,
    noise_rate: float = 0.3,
    
    # Model configuration
    model_name: str = "distilbert-base-uncased", 
    
    # Training configuration
    epochs: int = 10,  # More epochs for better convergence
    batch_size: int = 32,  # Larger batch size for server
    
    # TIM configuration
    tim_epochs: int = 5,  # More TIM epochs for better influence computation
    
    # Experiment configuration
    output_dir: str = "./large_scale_noise_pruning_results",
    random_state: int = 42
) -> NoisePruningExperiment:
    """
    Factory function: Create noise pruning experiment
    
    Parameters:
    -----------
    dataset_name : str
        Dataset name, default "imdb"
    train_count : int
        Training sample count, default 5000
    valid_count : int
        Validation sample count, default 1000
    test_count : int
        Test sample count, default 1000
    noise_rate : float
        Noise ratio, default 0.3
    model_name : str
        Model name, default "distilbert-base-uncased"
    epochs : int
        Training epochs, default 10
    batch_size : int
        Batch size, default 32
    tim_epochs : int
        TIM epochs, default 5
    output_dir : str
        Output directory, default "./large_scale_noise_pruning_results"
    random_state : int
        Random seed, default 42
        
    Returns:
    --------
    NoisePruningExperiment
        Configured experiment object
    """
    return NoisePruningExperiment(
        dataset_name=dataset_name,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        noise_rate=noise_rate,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        tim_epochs=tim_epochs,
        output_dir=output_dir,
        random_state=random_state
    )


def main():
    """Main function - Run complete large-scale experiment"""
    print("🧪 Large-scale Noise Data Pruning Experiment")
    print("=" * 60)
    
    # Create large-scale experiment with default parameters optimized for server
    experiment = create_experiment()  # Uses large-scale defaults
    
    # Run experiment
    results = experiment.run_complete_experiment()
    
    # Return results
    return results


if __name__ == "__main__":
    success = main()
    if success and success.get('status') == 'success':
        print("\\n🎉 实验成功完成！")
        sys.exit(0)
    else:
        print("\\n❌ 实验失败")
        sys.exit(1)