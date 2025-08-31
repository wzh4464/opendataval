"""
数据噪声注入和处理模块

使用OpenDataVal的噪声注入API来处理标签噪声，支持模块化的数据处理流程。
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path

from opendataval.dataloader import DataFetcher
from opendataval.dataloader.noisify import add_gauss_noise, add_noise


class NoiseDataProcessor:
    """数据噪声注入和处理器"""
    
    def __init__(
        self, 
        dataset_name: str = "imdb",
        train_count: int = 1000,
        valid_count: int = 200,
        test_count: int = 200,
        noise_rate: float = 0.3,
        random_state: int = 42
    ):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        dataset_name : str
            数据集名称
        train_count : int
            训练样本数量
        valid_count : int 
            验证样本数量
        test_count : int
            测试样本数量
        noise_rate : float
            标签噪声比例 (0.0-1.0)
        random_state : int
            随机种子
        """
        self.dataset_name = dataset_name
        self.train_count = train_count
        self.valid_count = valid_count
        self.test_count = test_count
        self.noise_rate = noise_rate
        self.random_state = random_state
        
        # 数据存储
        self.clean_data = None
        self.noisy_data = None
        self.noise_indices = None
        
    def load_clean_data(self) -> Tuple:
        """加载干净的数据"""
        print(f"🔄 加载干净数据: {self.dataset_name}")
        print(f"📊 数据规模: 训练={self.train_count}, 验证={self.valid_count}, 测试={self.test_count}")
        
        # 使用DataFetcher加载数据
        fetcher = DataFetcher.setup(
            dataset_name=self.dataset_name,
            train_count=self.train_count,
            valid_count=self.valid_count,
            test_count=self.test_count,
            random_state=self.random_state
        )
        
        x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
        
        # 数据格式处理
        x_train_data, y_train_clean = self._process_data_format(x_train, y_train)
        x_valid_data, y_valid_clean = self._process_data_format(x_valid, y_valid)
        x_test_data, y_test_clean = self._process_data_format(x_test, y_test)
        
        self.clean_data = {
            'x_train': x_train_data,
            'y_train': y_train_clean,
            'x_valid': x_valid_data,
            'y_valid': y_valid_clean,
            'x_test': x_test_data,
            'y_test': y_test_clean
        }
        
        print("✅ 干净数据加载完成")
        print(f"   训练集样本数: {len(x_train_data)}")
        print(f"   验证集样本数: {len(x_valid_data)}")
        print(f"   测试集样本数: {len(x_test_data)}")
        
        return self.clean_data
    
    def inject_label_noise(self, noise_type: str = "uniform") -> Tuple:
        """
        注入标签噪声
        
        Parameters:
        -----------
        noise_type : str
            噪声类型: 'uniform' 或 'class_dependent'
            
        Returns:
        --------
        Tuple: (noisy_data, noise_indices)
        """
        if self.clean_data is None:
            raise ValueError("请先调用load_clean_data()加载数据")
            
        print(f"🔀 注入标签噪声: {self.noise_rate*100:.1f}% ({noise_type})")
        
        y_train_clean = self.clean_data['y_train']
        n_samples = len(y_train_clean)
        n_noisy = int(n_samples * self.noise_rate)
        
        # 设置随机种子
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # 选择要加噪声的样本
        noise_indices = np.random.choice(n_samples, size=n_noisy, replace=False)
        
        # 创建噪声标签
        y_train_noisy = y_train_clean.clone()
        
        if noise_type == "uniform":
            # 均匀标签翻转
            for idx in noise_indices:
                # 对于二分类，直接翻转标签
                y_train_noisy[idx] = 1 - y_train_noisy[idx]
                
        elif noise_type == "class_dependent":
            # 类别相关噪声（可以扩展）
            for idx in noise_indices:
                # 简化版：也是翻转，但可以根据类别设置不同噪声率
                y_train_noisy[idx] = 1 - y_train_noisy[idx]
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")
        
        # 创建噪声数据副本
        self.noisy_data = self.clean_data.copy()
        self.noisy_data['y_train'] = y_train_noisy
        self.noise_indices = noise_indices
        
        # 统计噪声信息
        actually_flipped = torch.sum(y_train_clean != y_train_noisy).item()
        
        print(f"✅ 噪声注入完成")
        print(f"   目标噪声样本: {n_noisy}")
        print(f"   实际翻转样本: {actually_flipped}")
        print(f"   实际噪声率: {actually_flipped/n_samples*100:.1f}%")
        
        return self.noisy_data, self.noise_indices
    
    def _process_data_format(self, x_data, y_data):
        """处理数据格式，统一为列表和tensor"""
        # 处理特征数据
        if hasattr(x_data, 'dataset'):
            # 如果是Subset对象，提取实际数据
            x_processed = [x_data.dataset[i] for i in x_data.indices]
        else:
            x_processed = list(x_data)
            
        # 处理标签数据
        if not isinstance(y_data, torch.Tensor):
            y_processed = torch.tensor(y_data, dtype=torch.long)
        else:
            y_processed = y_data.clone()
            
        # 如果是one-hot编码，转换为索引
        if len(y_processed.shape) > 1 and y_processed.shape[1] > 1:
            y_processed = torch.argmax(y_processed, dim=1)
            
        return x_processed, y_processed
    
    def get_noise_statistics(self) -> dict:
        """获取噪声统计信息"""
        if self.noisy_data is None or self.noise_indices is None:
            return {"error": "噪声数据未生成"}
            
        clean_labels = self.clean_data['y_train']
        noisy_labels = self.noisy_data['y_train']
        
        # 按类别统计
        stats = {
            'total_samples': len(clean_labels),
            'noise_indices': self.noise_indices.tolist(),
            'noise_count': len(self.noise_indices),
            'noise_rate': len(self.noise_indices) / len(clean_labels),
            'actually_flipped': torch.sum(clean_labels != noisy_labels).item(),
            'class_distribution': {
                'clean': {
                    'positive': torch.sum(clean_labels == 1).item(),
                    'negative': torch.sum(clean_labels == 0).item()
                },
                'noisy': {
                    'positive': torch.sum(noisy_labels == 1).item(),
                    'negative': torch.sum(noisy_labels == 0).item()
                }
            }
        }
        
        return stats
    
    def prune_data_by_indices(self, prune_indices: np.ndarray) -> Tuple:
        """
        根据索引剪枝数据
        
        Parameters:
        -----------
        prune_indices : np.ndarray
            要移除的样本索引
            
        Returns:
        --------
        Tuple: (pruned_data, remaining_indices)
        """
        if self.noisy_data is None:
            raise ValueError("噪声数据未生成，请先调用inject_label_noise()")
            
        print(f"✂️  剪枝数据: 移除 {len(prune_indices)} 个样本")
        
        # 获取保留的索引
        all_indices = np.arange(len(self.noisy_data['y_train']))
        remaining_indices = np.setdiff1d(all_indices, prune_indices)
        
        # 创建剪枝后的数据
        pruned_data = {}
        
        # 处理训练数据
        x_train_pruned = [self.noisy_data['x_train'][i] for i in remaining_indices]
        y_train_pruned = self.noisy_data['y_train'][remaining_indices]
        
        pruned_data = {
            'x_train': x_train_pruned,
            'y_train': y_train_pruned,
            'x_valid': self.noisy_data['x_valid'],  # 验证集不变
            'y_valid': self.noisy_data['y_valid'],
            'x_test': self.noisy_data['x_test'],    # 测试集不变  
            'y_test': self.noisy_data['y_test']
        }
        
        print(f"✅ 剪枝完成")
        print(f"   原始训练样本: {len(self.noisy_data['y_train'])}")
        print(f"   剪枝后样本: {len(y_train_pruned)}")
        print(f"   保留率: {len(remaining_indices)/len(all_indices)*100:.1f}%")
        
        return pruned_data, remaining_indices
    
    def save_data(self, data: dict, save_path: str):
        """保存数据到文件"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存为pytorch格式
        torch.save(data, save_path / "processed_data.pt")
        
        # 保存统计信息
        if hasattr(self, 'noise_indices') and self.noise_indices is not None:
            stats = self.get_noise_statistics()
            import json
            with open(save_path / "noise_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
                
        print(f"💾 数据已保存到: {save_path}")


def create_noise_processor(
    dataset_name: str = "imdb",
    train_count: int = 1000,
    valid_count: int = 200, 
    test_count: int = 200,
    noise_rate: float = 0.3,
    random_state: int = 42
) -> NoiseDataProcessor:
    """
    工厂函数：创建噪声数据处理器
    
    Parameters:
    -----------
    dataset_name : str
        数据集名称，默认"imdb"
    train_count : int
        训练样本数量，默认1000
    valid_count : int
        验证样本数量，默认200
    test_count : int
        测试样本数量，默认200
    noise_rate : float
        标签噪声比例，默认0.3 (30%)
    random_state : int
        随机种子，默认42
        
    Returns:
    --------
    NoiseDataProcessor
        配置好的数据处理器
    """
    return NoiseDataProcessor(
        dataset_name=dataset_name,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        noise_rate=noise_rate,
        random_state=random_state
    )


if __name__ == "__main__":
    # 测试数据处理器
    processor = create_noise_processor(
        train_count=100,
        valid_count=50,
        test_count=50,
        noise_rate=0.3
    )
    
    # 加载数据
    clean_data = processor.load_clean_data()
    
    # 注入噪声
    noisy_data, noise_indices = processor.inject_label_noise()
    
    # 获取统计
    stats = processor.get_noise_statistics()
    print(f"📊 噪声统计: {stats}")