"""
准备CIFAR-10数据集脚本

此脚本将CIFAR-10数据集转换为符合项目要求的格式：
- 添加image_path列
- 添加split列（train/val/test，按70/15/15比例）
- 保存为CSV文件

"""

import pandas as pd
from pathlib import Path
import random

# 设置随机种子以保证可复现性
random.seed(42)

# 配置
RAW_DATA_PATH = Path("data/raw/cifar-10")
OUTPUT_PATH = Path("data/raw/cifar-10/dataset.csv")

# CIFAR-10类别
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"
]

def create_dataset_csv():
    """
    从trainLabels.csv创建包含image_path和split列的完整数据集CSV
    
    返回：
        DataFrame: 包含image_path, label, split三列的数据集
    """
    print("正在读取trainLabels.csv...")
    
    # 读取标签文件
    labels_df = pd.read_csv(RAW_DATA_PATH / "trainLabels.csv")
    print(f"总记录数: {len(labels_df)}")
    
    # 创建image_path列
    labels_df['image_path'] = labels_df['id'].apply(
        lambda x: f"data/raw/cifar-10/train/{x}.png"
    )
    
    # 随机分配split（70% train, 15% val, 15% test）
    splits = []
    for _ in range(len(labels_df)):
        r = random.random()
        if r < 0.70:
            splits.append('train')
        elif r < 0.85:
            splits.append('val')
        else:
            splits.append('test')
    
    labels_df['split'] = splits
    
    # 统计每个split的数量
    split_counts = labels_df['split'].value_counts()
    print("\n数据集划分统计:")
    for split, count in split_counts.items():
        percentage = (count / len(labels_df)) * 100
        print(f"  {split}: {count} ({percentage:.1f}%)")
    
    # 确保至少有1000条记录（满足最小要求）
    if len(labels_df) < 1000:
        print("警告: 记录数少于1000，可能不满足要求")
    
    return labels_df[['image_path', 'label', 'split']]


def verify_data(df):
    """
    验证数据集的有效性
    
    参数：
        df: 包含image_path, label, split的DataFrame
    """
    print("\n正在验证数据...")
    
    # 检查必需的列
    required_cols = ['image_path', 'label', 'split']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    # 检查split值
    valid_splits = {'train', 'val', 'test'}
    unique_splits = set(df['split'].unique())
    if not unique_splits.issubset(valid_splits):
        raise ValueError(f"无效的split值: {unique_splits - valid_splits}")
    
    # 检查标签值
    unique_labels = set(df['label'].unique())
    print(f"标签类别: {sorted(unique_labels)}")
    
    # 检查图像文件是否存在
    missing_files = []
    for idx, row in df.iterrows():
        if not Path(row['image_path']).exists():
            missing_files.append(row['image_path'])
    
    if missing_files:
        print(f"警告: {len(missing_files)} 个图像文件不存在")
        print(f"示例: {missing_files[:5]}")
    else:
        print("所有图像文件都存在 ✓")
    
    print("数据验证完成 ✓")


def main():
    """主函数"""
    print("=" * 60)
    print("CIFAR-10数据集准备脚本")
    print("=" * 60)
    
    # 创建数据集CSV
    df = create_dataset_csv()
    
    # 验证数据
    verify_data(df)
    
    # 保存到CSV
    print(f"\n正在保存到 {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ 已保存 {len(df)} 条记录")
    
    # 显示前几行
    print("\n前5行数据:")
    print(df.head())
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
