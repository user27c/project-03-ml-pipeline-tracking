"""
图像数据集模块

定义 PyTorch Dataset 类用于加载图像分类数据。

"""

import os
from PIL import Image
from typing import Tuple, Optional, Callable
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms


class ImageClassificationDataset(Dataset):
    """
    图像分类数据集。
    
    该类从DataFrame读取图像路径和标签，加载图像并应用转换。
    
    属性：
        df (DataFrame): 包含图像路径和标签的数据框
        root_dir (str): 图像根目录
        transform (callable): 图像转换
        image_paths (list): 图像路径列表
        labels (list): 标签列表
        label_to_idx (dict): 标签到索引的映射
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        label_column: str = 'label',
        image_path_column: str = 'image_path',
        precomputed_labels: bool = False
    ) -> None:
        """
        初始化图像分类数据集。
        
        参数：
            df: 包含图像路径和标签的DataFrame
            root_dir: 图像根目录（如果CSV中是相对路径）
            transform: 图像转换（训练时使用数据增强）
            label_column: 标签列名
            image_path_column: 图像路径列名
            precomputed_labels: 是否使用预计算的标签索引
        """
        # 存储DataFrame
        self.df = df.copy()
        
        # 存储列名
        self.label_column = label_column
        self.image_path_column = image_path_column
        
        # 如果提供了root_dir，将相对路径转换为绝对路径
        self.root_dir = root_dir
        if self.root_dir:
            self.df[image_path_column] = self.df[image_path_column].apply(
                lambda x: os.path.join(self.root_dir, x)
            )
        
        # 获取图像路径
        self.image_paths = self.df[image_path_column].tolist()
        
        # 获取标签
        if precomputed_labels:
            # 使用预计算的标签索引
            self.labels = self.df[label_column].tolist()
            # 从标签索引推断类别数量
            self.num_classes = len(set(self.labels))
            self.label_to_idx = None
            self.idx_to_label = None
        else:
            # 从原始标签创建映射
            self.labels = self.df[label_column].tolist()
            unique_labels = sorted(set(self.labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            self.num_classes = len(unique_labels)
        
        # 转换
        self.transform = transform
        
        print(f"加载了 {len(self.df)} 张图像，{self.num_classes} 个类别")
    
    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个样本。
        
        参数：
            idx: 样本索引
            
        返回：
            tuple: (image, label)
        """
        # 获取图像路径
        image_path = self.image_paths[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 获取标签
        label_idx = self.labels[idx]
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx


def create_data_loaders(
    csv_path: str,
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    train_split: str = 'train',
    val_split: str = 'val',
    test_split: str = 'test',
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器。
    
    参数：
        csv_path: CSV文件路径
        root_dir: 图像根目录
        batch_size: 批大小
        train_split: 训练集split值
        val_split: 验证集split值
        test_split: 测试集split值
        num_workers: 数据加载器的工作线程数
        pin_memory: 是否使用锁页内存
        
    返回：
        tuple: (train_loader, val_loader, test_loader)
    """
    # 定义图像转换
    # 训练时使用数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 验证和测试时只做基本转换
    eval_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 读取CSV文件
    full_df = pd.read_csv(csv_path)
    
    # 获取唯一标签并创建映射
    unique_labels = sorted(full_df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 过滤出训练集
    train_df = full_df[full_df['split'] == train_split].copy()
    train_df['label_idx'] = train_df['label'].map(label_to_idx)
    
    # 过滤出验证集
    val_df = full_df[full_df['split'] == val_split].copy()
    val_df['label_idx'] = val_df['label'].map(label_to_idx)
    
    # 过滤出测试集
    test_df = full_df[full_df['split'] == test_split].copy()
    test_df['label_idx'] = test_df['label'].map(label_to_idx)
    
    # 创建数据集（直接传递DataFrame，避免重复读取CSV）
    train_dataset = ImageClassificationDataset(
        df=train_df,
        root_dir=root_dir,
        transform=train_transform,
        label_column='label_idx',
        precomputed_labels=True
    )
    
    val_dataset = ImageClassificationDataset(
        df=val_df,
        root_dir=root_dir,
        transform=eval_transform,
        label_column='label_idx',
        precomputed_labels=True
    )
    
    test_dataset = ImageClassificationDataset(
        df=test_df,
        root_dir=root_dir,
        transform=eval_transform,
        label_column='label_idx',
        precomputed_labels=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"创建了数据加载器:")
    print(f"  训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次")
    print(f"  验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次")
    print(f"  测试集: {len(test_loader.dataset)} 样本, {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader


# 示例用法
if __name__ == "__main__":
    """示例用法。"""
    csv_path = 'data/raw/cifar-10/dataset.csv'
    root_dir = '/home/22-7/Dev/ai-infra-learn/ai-infra-project/junior-engineer/project-03-ml-pipeline-tracking'
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_path=csv_path,
        root_dir=root_dir,
        batch_size=32
    )
    
    # 测试一个批次
    for images, labels in train_loader:
        print(f"Batch shape: images={images.shape}, labels={labels.shape}")
        break
