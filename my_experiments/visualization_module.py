"""
可视化模块

提供BERT训练过程和TIM影响力分析的可视化功能，支持损失曲线、影响力分布和对比分析。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class ExperimentVisualizer:
    """实验可视化器"""
    
    def __init__(
        self, 
        save_dir: str = "./experiment_plots",
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 150
    ):
        """
        初始化可视化器
        
        Parameters:
        -----------
        save_dir : str
            图片保存目录
        figure_size : Tuple[int, int]
            图片尺寸
        dpi : int
            图片分辨率
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = figure_size
        self.dpi = dpi
        
        # 配色方案
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#28B463', 
            'accent': '#F39C12',
            'warning': '#E74C3C',
            'neutral': '#85929E',
            'noise': '#E74C3C',
            'clean': '#28B463',
            'original': '#2E86C1',
            'pruned': '#8E44AD'
        }
        
    def plot_training_curves(
        self,
        history: Dict,
        title: str = "训练曲线",
        save_name: str = "training_curves.png"
    ):
        """
        绘制训练损失和准确率曲线
        
        Parameters:
        -----------
        history : Dict
            训练历史数据
        title : str
            图表标题
        save_name : str
            保存文件名
        """
        print(f"📊 绘制训练曲线: {title}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Epoch级别的损失和准确率
        if 'train_loss' in history and 'train_accuracy' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            
            # 训练损失
            ax1.plot(epochs, history['train_loss'], 
                    color=self.colors['primary'], linewidth=2, label='训练损失')
            if 'valid_loss' in history and len(history['valid_loss']) > 0:
                # 验证损失可能没有每个epoch都有
                valid_steps = np.linspace(1, len(epochs), len(history['valid_loss']))
                ax1.plot(valid_steps, history['valid_loss'], 
                        color=self.colors['accent'], linewidth=2, label='验证损失')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('损失曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 训练准确率
            ax2.plot(epochs, history['train_accuracy'], 
                    color=self.colors['secondary'], linewidth=2, label='训练准确率')
            if 'valid_accuracy' in history and len(history['valid_accuracy']) > 0:
                valid_steps = np.linspace(1, len(epochs), len(history['valid_accuracy']))
                ax2.plot(valid_steps, history['valid_accuracy'], 
                        color=self.colors['accent'], linewidth=2, label='验证准确率')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('准确率曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 2. Step级别的详细曲线（如果有）
        if 'step_losses' in history:
            steps = range(1, len(history['step_losses']) + 1)
            ax3.plot(steps, history['step_losses'], 
                    color=self.colors['primary'], linewidth=1, alpha=0.7)
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Loss')
            ax3.set_title('Step级损失曲线')
            ax3.grid(True, alpha=0.3)
            
        if 'step_accuracies' in history:
            steps = range(1, len(history['step_accuracies']) + 1)
            ax4.plot(steps, history['step_accuracies'], 
                    color=self.colors['secondary'], linewidth=1, alpha=0.7)
            ax4.set_xlabel('Training Step')  
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Step级准确率曲线')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        print(f"💾 训练曲线已保存: {save_path}")
        
    def plot_influence_distribution(
        self,
        influence_scores: np.ndarray,
        noise_indices: Optional[np.ndarray] = None,
        title: str = "影响力分数分布",
        save_name: str = "influence_distribution.png"
    ):
        """
        绘制影响力分数分布图
        
        Parameters:
        -----------
        influence_scores : np.ndarray
            影响力分数
        noise_indices : Optional[np.ndarray]
            噪声样本索引
        title : str
            图表标题
        save_name : str
            保存文件名
        """
        print(f"📊 绘制影响力分布: {title}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 影响力分数直方图
        ax1.hist(influence_scores, bins=50, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(influence_scores), color=self.colors['warning'], 
                   linestyle='--', linewidth=2, label=f'均值: {np.mean(influence_scores):.6f}')
        ax1.set_xlabel('影响力分数')
        ax1.set_ylabel('频数')
        ax1.set_title('影响力分数分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 影响力排名图
        sorted_indices = np.argsort(influence_scores)
        ranked_scores = influence_scores[sorted_indices]
        
        ax2.plot(range(len(ranked_scores)), ranked_scores, 
                color=self.colors['primary'], linewidth=2)
        ax2.set_xlabel('样本排名（按影响力）')
        ax2.set_ylabel('影响力分数')
        ax2.set_title('影响力排名曲线')
        ax2.grid(True, alpha=0.3)
        
        # 3. 如果有噪声信息，绘制噪声vs干净样本对比
        if noise_indices is not None:
            clean_indices = np.setdiff1d(np.arange(len(influence_scores)), noise_indices)
            
            # 分别绘制噪声和干净样本的影响力
            ax3.hist(influence_scores[noise_indices], bins=25, alpha=0.7, 
                    color=self.colors['noise'], label=f'噪声样本 (n={len(noise_indices)})')
            ax3.hist(influence_scores[clean_indices], bins=25, alpha=0.7,
                    color=self.colors['clean'], label=f'干净样本 (n={len(clean_indices)})')
            ax3.set_xlabel('影响力分数')
            ax3.set_ylabel('频数')
            ax3.set_title('噪声vs干净样本影响力对比')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 噪声样本在排名中的位置
            noise_ranks = []
            for noise_idx in noise_indices:
                rank = np.where(sorted_indices == noise_idx)[0][0]
                rank_percentile = rank / len(influence_scores)
                noise_ranks.append(rank_percentile)
                
            ax4.hist(noise_ranks, bins=20, color=self.colors['noise'], 
                    alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(noise_ranks), color=self.colors['warning'], 
                       linestyle='--', linewidth=2, 
                       label=f'平均排名百分位: {np.mean(noise_ranks):.3f}')
            ax4.set_xlabel('排名百分位')
            ax4.set_ylabel('噪声样本数')
            ax4.set_title('噪声样本排名分布')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # 如果没有噪声信息，显示影响力统计
            stats_text = f"""影响力统计:
均值: {np.mean(influence_scores):.6f}
标准差: {np.std(influence_scores):.6f}
最小值: {np.min(influence_scores):.6f}
最大值: {np.max(influence_scores):.6f}
样本数: {len(influence_scores)}"""
            
            ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['neutral'], alpha=0.3))
            ax3.set_title('影响力统计信息')
            ax3.axis('off')
            
            # 显示前20和后20的影响力分数
            top_k = min(20, len(influence_scores))
            most_influential = np.argsort(influence_scores)[-top_k:][::-1]
            least_influential = np.argsort(influence_scores)[:top_k]
            
            ax4.barh(range(top_k), influence_scores[most_influential], 
                    color=self.colors['secondary'], alpha=0.7, label='最有影响力')
            ax4.barh(range(top_k, 2*top_k), influence_scores[least_influential], 
                    color=self.colors['warning'], alpha=0.7, label='最无影响力')
            ax4.set_xlabel('影响力分数')
            ax4.set_ylabel('样本索引')
            ax4.set_title(f'Top/Bottom {top_k} 影响力样本')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        print(f"💾 影响力分布图已保存: {save_path}")
        
    def plot_comparative_analysis(
        self,
        original_history: Dict,
        pruned_history: Dict,
        influence_stats: Dict,
        title: str = "剪枝前后对比分析",
        save_name: str = "comparative_analysis.png"
    ):
        """
        绘制剪枝前后的对比分析图
        
        Parameters:
        -----------
        original_history : Dict
            原始（含噪声）训练历史
        pruned_history : Dict
            剪枝后训练历史
        influence_stats : Dict
            影响力统计信息
        title : str
            图表标题
        save_name : str
            保存文件名
        """
        print(f"📊 绘制对比分析: {title}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 训练损失对比
        if 'train_loss' in original_history and 'train_loss' in pruned_history:
            orig_epochs = range(1, len(original_history['train_loss']) + 1)
            pruned_epochs = range(1, len(pruned_history['train_loss']) + 1)
            
            ax1.plot(orig_epochs, original_history['train_loss'], 
                    color=self.colors['original'], linewidth=2, label='原始（含噪声）')
            ax1.plot(pruned_epochs, pruned_history['train_loss'], 
                    color=self.colors['pruned'], linewidth=2, label='剪枝后')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('训练损失对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 训练准确率对比
        if 'train_accuracy' in original_history and 'train_accuracy' in pruned_history:
            ax2.plot(orig_epochs, original_history['train_accuracy'], 
                    color=self.colors['original'], linewidth=2, label='原始（含噪声）')
            ax2.plot(pruned_epochs, pruned_history['train_accuracy'], 
                    color=self.colors['pruned'], linewidth=2, label='剪枝后')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Training Accuracy')
            ax2.set_title('训练准确率对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 最终性能对比
        orig_final = original_history.get('final_performance', {})
        pruned_final = pruned_history.get('final_performance', {})
        
        metrics = ['train_accuracy', 'valid_accuracy']
        orig_values = [orig_final.get(m, 0) for m in metrics]
        pruned_values = [pruned_final.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, orig_values, width, label='原始（含噪声）', 
               color=self.colors['original'], alpha=0.7)
        ax3.bar(x + width/2, pruned_values, width, label='剪枝后', 
               color=self.colors['pruned'], alpha=0.7)
        
        ax3.set_xlabel('指标')
        ax3.set_ylabel('准确率')
        ax3.set_title('最终性能对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['训练准确率', '验证准确率'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (orig, pruned) in enumerate(zip(orig_values, pruned_values)):
            ax3.text(i - width/2, orig + 0.01, f'{orig:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
            ax3.text(i + width/2, pruned + 0.01, f'{pruned:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 4. 实验设置和改进总结
        summary_text = f"""实验设置:
• 数据集: {influence_stats.get('total_samples', 'N/A')} 个训练样本
• 噪声率: {influence_stats.get('noise_rate', 0)*100:.1f}%
• 剪枝样本: {influence_stats.get('noise_count', 'N/A')} 个

性能改进:
• 训练准确率: {orig_final.get('train_accuracy', 0):.3f} → {pruned_final.get('train_accuracy', 0):.3f} ({pruned_final.get('train_accuracy', 0) - orig_final.get('train_accuracy', 0):+.3f})
• 验证准确率: {orig_final.get('valid_accuracy', 0):.3f} → {pruned_final.get('valid_accuracy', 0):.3f} ({pruned_final.get('valid_accuracy', 0) - orig_final.get('valid_accuracy', 0):+.3f})
• 训练时间: {orig_final.get('total_time', 0):.1f}s → {pruned_final.get('total_time', 0):.1f}s

TIM影响力分析:
• 平均影响力: {influence_stats.get('mean_influence', 0):.6f}
• 影响力标准差: {influence_stats.get('std_influence', 0):.6f}"""

        if 'noise_analysis' in influence_stats:
            noise_analysis = influence_stats['noise_analysis']
            summary_text += f"""
• 噪声样本平均影响力: {noise_analysis.get('noise_samples', {}).get('mean_influence', 0):.6f}
• 干净样本平均影响力: {noise_analysis.get('clean_samples', {}).get('mean_influence', 0):.6f}
• 噪声样本平均排名百分位: {noise_analysis.get('mean_noise_rank_percentile', 0):.3f}"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['neutral'], alpha=0.2))
        ax4.set_title('实验总结')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        print(f"💾 对比分析图已保存: {save_path}")
        
    def plot_pruning_analysis(
        self,
        influence_scores: np.ndarray,
        prune_indices: np.ndarray,
        keep_indices: np.ndarray,
        noise_indices: Optional[np.ndarray] = None,
        title: str = "数据剪枝分析",
        save_name: str = "pruning_analysis.png"
    ):
        """
        绘制数据剪枝分析图
        
        Parameters:
        -----------
        influence_scores : np.ndarray
            影响力分数
        prune_indices : np.ndarray
            被剪枝的样本索引
        keep_indices : np.ndarray
            保留的样本索引
        noise_indices : Optional[np.ndarray]
            噪声样本索引
        title : str
            图表标题
        save_name : str
            保存文件名
        """
        print(f"📊 绘制剪枝分析: {title}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 剪枝vs保留样本的影响力对比
        ax1.hist(influence_scores[prune_indices], bins=30, alpha=0.7, 
                color=self.colors['warning'], label=f'剪枝样本 (n={len(prune_indices)})')
        ax1.hist(influence_scores[keep_indices], bins=30, alpha=0.7,
                color=self.colors['secondary'], label=f'保留样本 (n={len(keep_indices)})')
        ax1.set_xlabel('影响力分数')
        ax1.set_ylabel('频数')
        ax1.set_title('剪枝vs保留样本影响力分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 影响力排名与剪枝决策
        sorted_indices = np.argsort(influence_scores)
        ranks = np.arange(len(influence_scores))
        
        # 标记剪枝和保留的样本
        prune_mask = np.isin(sorted_indices, prune_indices)
        keep_mask = np.isin(sorted_indices, keep_indices)
        
        ax2.scatter(ranks[prune_mask], influence_scores[sorted_indices[prune_mask]], 
                   color=self.colors['warning'], alpha=0.7, s=20, label='剪枝样本')
        ax2.scatter(ranks[keep_mask], influence_scores[sorted_indices[keep_mask]], 
                   color=self.colors['secondary'], alpha=0.7, s=20, label='保留样本')
        
        # 添加剪枝阈值线
        threshold_rank = len(prune_indices)
        ax2.axvline(threshold_rank, color=self.colors['accent'], 
                   linestyle='--', linewidth=2, label=f'剪枝阈值')
        
        ax2.set_xlabel('影响力排名')
        ax2.set_ylabel('影响力分数')
        ax2.set_title('影响力排名与剪枝决策')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 如果有噪声信息，分析剪枝效果
        if noise_indices is not None:
            # 计算剪枝捕获的噪声样本比例
            pruned_noise = np.intersect1d(prune_indices, noise_indices)
            noise_recall = len(pruned_noise) / len(noise_indices) if len(noise_indices) > 0 else 0
            noise_precision = len(pruned_noise) / len(prune_indices) if len(prune_indices) > 0 else 0
            
            # 创建混淆矩阵风格的可视化
            categories = ['剪枝样本', '保留样本']
            noise_counts = [
                len(np.intersect1d(prune_indices, noise_indices)),
                len(np.intersect1d(keep_indices, noise_indices))
            ]
            clean_counts = [
                len(prune_indices) - noise_counts[0],
                len(keep_indices) - noise_counts[1]
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, noise_counts, width, label='噪声样本', 
                           color=self.colors['noise'], alpha=0.7)
            bars2 = ax3.bar(x + width/2, clean_counts, width, label='干净样本', 
                           color=self.colors['clean'], alpha=0.7)
            
            ax3.set_xlabel('剪枝决策')
            ax3.set_ylabel('样本数量')
            ax3.set_title(f'噪声捕获效果 (召回率: {noise_recall:.2%}, 精确率: {noise_precision:.2%})')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontweight='bold')
        
        # 4. 剪枝统计摘要
        stats_text = f"""剪枝统计:
总样本数: {len(influence_scores)}
剪枝样本数: {len(prune_indices)} ({len(prune_indices)/len(influence_scores)*100:.1f}%)
保留样本数: {len(keep_indices)} ({len(keep_indices)/len(influence_scores)*100:.1f}%)

影响力阈值: {influence_scores[sorted_indices[len(prune_indices)-1]]:.6f}

剪枝样本影响力:
• 均值: {np.mean(influence_scores[prune_indices]):.6f}
• 标准差: {np.std(influence_scores[prune_indices]):.6f}

保留样本影响力:
• 均值: {np.mean(influence_scores[keep_indices]):.6f}
• 标准差: {np.std(influence_scores[keep_indices]):.6f}"""
        
        if noise_indices is not None:
            stats_text += f"""

噪声检测效果:
• 噪声样本数: {len(noise_indices)}
• 捕获噪声数: {len(np.intersect1d(prune_indices, noise_indices))}
• 噪声召回率: {noise_recall:.2%}
• 剪枝精确率: {noise_precision:.2%}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['neutral'], alpha=0.2))
        ax4.set_title('剪枝统计摘要')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        print(f"💾 剪枝分析图已保存: {save_path}")
        
    def save_experiment_summary(
        self,
        experiment_results: Dict,
        save_name: str = "experiment_summary.json"
    ):
        """
        保存实验结果摘要
        
        Parameters:
        -----------
        experiment_results : Dict
            完整的实验结果
        save_name : str
            保存文件名
        """
        save_path = self.save_dir / save_name
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"💾 实验摘要已保存: {save_path}")


def create_visualizer(
    save_dir: str = "./experiment_plots",
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> ExperimentVisualizer:
    """
    工厂函数：创建实验可视化器
    
    Parameters:
    -----------
    save_dir : str
        图片保存目录，默认"./experiment_plots"
    figure_size : Tuple[int, int]
        图片尺寸，默认(12, 8)
    dpi : int
        图片分辨率，默认150
        
    Returns:
    --------
    ExperimentVisualizer
        配置好的可视化器
    """
    return ExperimentVisualizer(
        save_dir=save_dir,
        figure_size=figure_size,
        dpi=dpi
    )


if __name__ == "__main__":
    # 测试可视化器
    print("🧪 测试可视化器")
    
    # 创建可视化器
    visualizer = create_visualizer()
    
    # 生成测试数据
    np.random.seed(42)
    
    # 模拟训练历史
    epochs = 5
    test_history = {
        'train_loss': np.random.exponential(0.5, epochs) + 0.1,
        'train_accuracy': np.random.beta(8, 2, epochs),
        'valid_loss': np.random.exponential(0.6, epochs//2) + 0.2,
        'valid_accuracy': np.random.beta(7, 3, epochs//2),
        'step_losses': np.random.exponential(0.5, epochs*20) + 0.1,
        'step_accuracies': np.random.beta(8, 2, epochs*20)
    }
    
    # 模拟影响力数据
    n_samples = 100
    influence_scores = np.random.normal(0.5, 0.2, n_samples)
    noise_indices = np.random.choice(n_samples, 30, replace=False)
    
    print("🎨 测试训练曲线绘制...")
    visualizer.plot_training_curves(test_history, title="测试训练曲线")
    
    print("🎨 测试影响力分布绘制...")
    visualizer.plot_influence_distribution(influence_scores, noise_indices, title="测试影响力分布")
    
    print("✅ 可视化器测试完成")