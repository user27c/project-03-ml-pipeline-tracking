"""
数据预处理模块

该模块处理用于模型训练的原始数据的清洗和转换。

学习目标：
- 实现数据清洗（重复值、缺失值）
- 编码分类变量
- 创建训练/验证/测试集划分
- 保存预处理工件以实现可复现性

"""

from datetime import datetime

import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    处理ML管道的数据预处理。

    该类提供清洗数据、编码标签、创建划分和保存预处理工件以实现可复现性的方法。

    属性：
        config (Dict[str, Any]): 配置字典
        label_encoder (LabelEncoder): 用于分类标签的编码器
        scaler (StandardScaler): 用于数值特征的缩放器
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        使用配置初始化DataPreprocessor。

        参数：
            config: 配置字典，包含：
                - processed_data_path: 保存处理后数据的路径
                - artifacts_path: 保存预处理工件的路径
                - required_columns: 必需的列名列表
                - test_size: 测试集比例（默认：0.2）
                - val_size: 验证集比例（默认：0.1）
                - random_state: 用于可复现性的随机种子（默认：42）

        TODO：
        1. 提取配置参数
        2. 如果不存在则创建输出目录
        3. 初始化标签编码器和缩放器
        4. 记录初始化
        """
        self.config = config

        # TODO: Extract paths from config
        self.processed_data_path = Path(config.get('processed_data_path'))
        self.artifacts_path = Path(config.get('artifacts_path'))

        # TODO: Create directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        # TODO: Initialize encoders and scalers
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # TODO: Extract configuration parameters with defaults
        self.required_columns = config.get('required_columns', [])
        self.test_size = config.get('test_size', 0.2)  # TODO: Get from config
        self.val_size = config.get('val_size', 0.1)
        self.random_state = config.get('random_state', 42)

        logger.info("DataPreprocessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        通过删除重复值和处理缺失值来清洗数据。

        参数：
            df: 要清洗的原始DataFrame

        返回：
            清洗后的DataFrame

        说明：
        1. 记录初始数据形状
        2. 删除完全重复的行
        3. 记录删除的重复行数
        4. 处理必需列中的缺失值（删除行）
        5. 记录删除的缺失值行数
        6. 重置索引
        7. 记录最终数据形状
        8. 返回清洗后的DataFrame

        清洗步骤：
        - 删除完全重复的行（保留第一个出现的）
        - 删除必需列中缺失值的行
        - 保留必需列中缺失值的行
        - 删除行后重置索引

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> df_clean = preprocessor.clean_data(df_raw)
            >>> print(f"清洗后：{len(df_raw)} -> {len(df_clean)} 行")
            清洗后：52000 -> 50000 行
        """
        logger.info(f"开始数据清洗。初始形状：{df.shape}")

        # 存储初始行数
        initial_rows = len(df)

        # 删除重复值
        df = df.drop_duplicates()
        # 计算并记录删除的重复值
        duplicates_removed = initial_rows - len(df)

        logger.info(f"删除了{duplicates_removed}行重复值")

        # 处理必需列中的缺失值
        df = df.dropna(subset=self.required_columns)
        # 计算并记录删除的缺失值
        missing_removed = initial_rows - len(df)

        logger.info(f"删除了{missing_removed}行包含必需缺失值的行")

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(f"数据清洗完成。最终形状：{df.shape}")

        return df

    def encode_labels(
        self,
        df: pd.DataFrame,
        label_column: str = 'label'
    ) -> pd.DataFrame:
        """
        将分类标签编码为整数。

        参数：
            df: 包含分类标签的DataFrame
            label_column: 标签列的名称

        返回：
            包含编码标签的新列的DataFrame

        TODO：
        1. 检查label_column是否存在
        2. 在唯一标签上拟合标签编码器
        3. 将标签转换为整数
        4. 添加编码列（例如：'label_encoded'）
        5. 将标签编码器保存到工件目录
        6. 将标签映射（原始->编码）保存为JSON
        7. 记录编码信息
        8. 返回带有新列的DataFrame

        标签编码器应保存以供推理期间使用。

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> df = preprocessor.encode_labels(df, 'label')
            >>> print(df[['label', 'label_encoded']].head())
               label  label_encoded
            0    cat              0
            1    dog              1
            2   bird              2
        """
        logger.info(f"正在从列编码标签：{label_column}")

        # 检查列是否存在
        if label_column not in df.columns:
            raise ValueError(f"列'{label_column}'未在DataFrame中找到")

        # 获取编码前的唯一标签
        unique_labels = None  # 用df[label_column].unique()替换

        # 拟合并转换标签
        encoded = self.label_encoder.fit_transform(df[label_column])

        # 将编码列添加到DataFrame
        encoded_column_name = f"{label_column}_encoded"
        df[encoded_column_name] = encoded

        # 保存标签编码器
        encoder_path = self.artifacts_path / 'label_encoder.pkl'
        # 提示：joblib.dump(self.label_encoder, encoder_path)

        # 创建并保存标签映射
        label_mapping = {label: encoded for label, encoded in zip(unique_labels, range(len(unique_labels)))}

        mapping_path = self.artifacts_path / 'label_mapping.json'
        # 将映射保存为JSON
        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=4)

        logger.info(f"编码了{len(unique_labels)}个唯一标签")
        logger.info(f"将标签编码器保存到{encoder_path}")
        logger.info(f"标签映射：{label_mapping}")

        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        使用StandardScaler归一化数值特征。

        参数：
            df: 包含数值特征的DataFrame
            feature_columns: 要归一化的列名列表

        返回：
            包含归一化特征的DataFrame

        方法：
        1. 验证特征列存在
        2. 在特征列上拟合缩放器
        3. 转换特征
        4. 用归一化值替换原始列
        5. 将缩放器保存到工件目录
        6. 记录归一化统计信息（均值、标准差）
        7. 返回DataFrame

        注意：仅在模型需要归一化特征时使用。
        对于使用预训练模型的图像分类，这可能不需要。

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> df = preprocessor.normalize_features(df, ['height', 'width'])
            >>> print(df[['height', 'width']].describe())
        """
        logger.info(f"归一化{len(feature_columns)}个特征")

        # 验证列存在
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"未找到列：{missing_cols}")

        # 拟合并转换特征
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])

        # 保存缩放器
        scaler_path = self.artifacts_path / 'scaler.pkl'
        # 提示：joblib.dump(self.scaler, scaler_path)

        # 记录统计信息
        logger.info(f"将缩放器保存到{scaler_path}")

        return df

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        stratify_column: Optional[str] = 'label_encoded'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        创建训练、验证和测试集划分。

        参数：
            df: 要划分的DataFrame
            stratify_column: 用于分层划分的列

        返回：
            包含（train_df, val_df, test_df）的元组

        说明：
        1. 验证stratify_column存在
        2. 第一次划分：从train+val中分离test集
        3. 第二次划分：分离train和验证集
        4. 验证分层是否成功（类似的类别分布）
        5. 记录划分大小和类别分布
        6. 返回三个DataFrame

        划分策略：
        - 使用分层划分以保持类别平衡
        - 默认划分：70%训练，15%验证，15%测试
        - 设置random_state以实现可复现性

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> train, val, test = preprocessor.create_train_test_split(df)
            >>> print(f"训练：{len(train)}，验证：{len(val)}，测试：{len(test)}")
            训练：35000，验证：7500，测试：7500
        """
        logger.info("创建训练/验证/测试集划分")

        # TODO: 验证分层列
        if stratify_column and stratify_column not in df.columns:
            logger.warning(f"分层列'{stratify_column}'未找到。使用随机划分。")
            stratify_column = None

        # 如果列存在则获取分层数组
        stratify_array = None
        if stratify_column:
            stratify_array = df[stratify_column]

        # 第一次划分：train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_array
        )

        # 计算验证集相对于train+val的大小
        # 如果总共是100%，测试是20%，那么train+val是80%
        # 我们希望验证是总大小的15%，所以15/80 = 0.1875的train+val
        val_ratio = self.val_size / (1 - self.test_size)

        # 第二次划分：train vs val
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val[stratify_column] if stratify_column else None
        )

        #记录划分大小
        logger.info(f"划分大小 - 训练：{0}，验证：{0}，测试：{0}")

        #如果分层则记录类别分布
        if stratify_column:
            #计算并记录每个划分的类别分布
            logger.info(f"类别分布 - 训练：{train[stratify_column].value_counts()}，验证：{val[stratify_column].value_counts()}，测试：{test[stratify_column].value_counts()}")

        #返回划分（实现时取消注释）
        return train, val, test

    def save_processed_data(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Dict[str, Path]:
        """
        保存处理后的训练、验证和测试集。

        参数：
            train: 训练DataFrame
            val: 验证DataFrame
            test: 测试DataFrame

        返回：
            包含划分名称到文件路径映射的字典

        说明：
        1. 将每个划分保存到processed_data_path中的CSV
        2. 为每个划分创建元数据（记录数、列等）
        3. 将元数据保存为JSON
        4. 记录保存位置
        5. 返回路径字典

        要创建的文件：
        - train.csv, val.csv, test.csv
        - train.meta.json, val.meta.json, test.meta.json

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> paths = preprocessor.save_processed_data(train, val, test)
            >>> print(paths['train'])
            PosixPath('data/processed/train.csv')
        """
        logger.info("保存处理后的数据划分")

        paths = {}

        # 保存每个划分
        splits = {
            'train': train,
            'val': val,
            'test': test
        }

        for split_name, split_df in splits.items():
            # 创建文件路径
            csv_path = self.processed_data_path / f"{split_name}.csv"

            # 保存为CSV
            split_df.to_csv(csv_path, index=False)

            # 创建元数据
            metadata = {
                # 填写元数据
                "split": split_name,
                "record_count": len(split_df),
                "columns": split_df.columns.tolist(),
                "shape": (len(split_df.columns), len(split_df)),
                "created_at": datetime.now()
            }

            # 保存元数据
            meta_path = csv_path.with_suffix('.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            # 添加到paths字典
            paths[split_name] = csv_path

            logger.info(f"将{split_name}划分保存到{csv_path}（{len(split_df)}条记录）")

        return paths

    def run_pipeline(
        self,
        df: pd.DataFrame,
        label_column: str = 'label'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        运行完整的预处理流程。

        参数：
            df: 原始DataFrame
            label_column: 标签列的名称

        返回：
            包含（train_df, val_df, test_df）的元组

        说明：
        1. 运行数据清洗
        2. 运行标签编码
        3. （可选）如果需要则运行特征归一化
        4. 创建训练/验证/测试划分
        5. 保存处理后的数据
        6. 保存预处理配置
        7. 返回划分

        这是运行所有预处理步骤的主入口点。

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> train, val, test = preprocessor.run_pipeline(raw_df)
            >>> print("预处理完成！")
        """
        logger.info("运行完整的预处理流程")

        # 步骤1 - 清洗数据
        logger.info("步骤1/4：清洗数据...")
        df_clean = self.clean_data(df)

        # 步骤2 - 编码标签
        logger.info("步骤2/4：编码标签...")
        df_encoded = self.encode_labels(df_clean, label_column)

        # 步骤3 - 创建划分
        logger.info("步骤3/4：创建训练/验证/测试划分...")
        train, val, test = self.create_train_test_split(df_encoded)

        # 步骤4 - 保存处理后的数据
        logger.info("步骤4/4：保存处理后的数据...")
        paths = self.save_processed_data(train, val, test)
        logger.info(f"已保存处理后的数据到{paths}")

        # 步骤5 - 保存预处理配置
        self._save_preprocessing_config()

        logger.info("预处理流程完成！")

        # 返回划分
        return train, val, test 

    def _save_preprocessing_config(self) -> None:
        """
        保存预处理配置以实现可复现性。

        # 说明：
        1. 创建包含所有预处理设置的配置字典
        2. 保存到工件目录的JSON文件
        3. 记录保存位置

        配置应包括：
        - required_columns
        - test_size, val_size
        - random_state
        - label_column
        - feature_columns（如果使用了归一化）
        """
        config_to_save = {
            # 填写配置
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "required_columns": self.required_columns
        }

        config_path = self.artifacts_path / 'preprocessing_config.json'

        # 步骤6 - 保存配置
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)

        logger.info(f"将预处理配置保存到{config_path}")

    def load_preprocessing_artifacts(self) -> None:
        """
        加载先前保存的预处理工件。

        # 说明：
        1. 从pickle加载标签编码器
        2. 从pickle加载缩放器（如果存在）
        3. 从JSON加载预处理配置
        4. 记录加载的工件
        5. 设置实例属性

        当需要使用与训练数据相同的转换预处理新数据时使用。

        示例：
            >>> preprocessor = DataPreprocessor(config)
            >>> preprocessor.load_preprocessing_artifacts()
            >>> # 现在可以对新数据使用加载的编码器/缩放器
        """
        logger.info("加载预处理工件")

        # 加载标签编码器
        encoder_path = self.artifacts_path / 'label_encoder.pkl'
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"从{encoder_path}加载了标签编码器")

        # 如果存在则加载缩放器
        scaler_path = self.artifacts_path / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"从{scaler_path}加载了缩放器")

        # 加载配置
        config_path = self.artifacts_path / 'preprocessing_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.test_size = config.get("test_size", 0.2)
                self.val_size = config.get("val_size", 0.1)
                self.random_state = config.get("random_state", 42)
                self.required_columns = config.get("required_columns", [])

            logger.info(f"从{config_path}加载了预处理配置")


# 示例用法和测试
if __name__ == "__main__":
    """
    DataPreprocessor类的示例用法。

    说明：
    1. 创建示例配置
    2. 创建示例DataFrame
    3. 初始化DataPreprocessor
    4. 运行预处理流程
    5. 验证输出

    创建一个包含以下内容的示例数据集：
    - image_path列
    - label列（分类）
    - 一些重复行
    - 一些缺失值
    """

    # 示例配置
    config = {
        'processed_data_path': 'data/processed',
        'artifacts_path': 'artifacts',
        'required_columns': ['image_path', 'label'],
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42
    }

    # 为测试创建示例DataFrame
    sample_data = {
        'image_path': ['img1.jpg', 'img2.jpg', ...],
        'label': ['cat', 'dog', 'cat', ...],
        'extra_col': [1, 2, 3, ...]
    }
    df = pd.DataFrame(sample_data)
    # 初始化预处理器
    preprocessor = DataPreprocessor(config)

    # 运行流程
    train, val, test = preprocessor.run_pipeline(df)

    # 验证输出
    print(f"训练：{len(train)}，验证：{len(val)}，测试：{len(test)}")

    print("DataPreprocessor模块已加载。")