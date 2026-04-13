"""
模型评估模块

该模块处理全面的模型评估，包括指标计算和可视化。

学习目标：
- 计算多个评估指标
- 生成混淆矩阵
- 创建分类报告
- 可视化模型性能
- 将结果记录到MLflow

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    在测试集上使用全面指标评估模型。

    该类计算多个指标，生成可视化，并将所有内容记录到MLflow进行跟踪。

    属性：
        config (Dict[str, Any]): 配置字典
        device (torch.device): 推理设备（CPU/GPU）
        class_names (List[str]): 类别名称列表
    """

    def __init__(
        self,
        config: Dict[str, Any],
        class_names: List[str]
    ) -> None:
        """
        初始化ModelEvaluator。

        参数：
            config: 配置字典
            class_names: 用于可视化的类别名称列表

        TODO：
        1. 存储配置
        2. 确定设备（如果有GPU则使用GPU）
        3. 存储类别名称
        4. 创建绘图输出目录
        5. 记录初始化
        """
        self.config = config
        self.class_names = class_names

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = None

        # Create plots directory
        self.plots_dir = Path(config.get('plots_dir', 'plots'))
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelEvaluator initialized with {len(class_names)} classes")

    def predict(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在测试集上生成预测。

        参数：
            model: 训练好的PyTorch模型
            test_loader: 测试数据加载器

        返回：
            包含（真实标签，预测标签）的numpy数组元组

        TODO：
        1. 将模型设置为评估模式
        2. 将模型移动到设备
        3. 遍历测试数据
        4. 生成预测（不需要梯度）
        5. 收集真实标签和预测标签
        6. 转换为numpy数组
        7. 返回两个数组
        """
        logger.info("Generating predictions on test set...")

        # Set model to eval mode and move to device
        model.eval()
        model = model.to(self.device)

        y_true = []
        y_pred = []

        # Disable gradient computation
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = model(inputs)

                # Get predictions
                _, predicted = outputs.max(1)

                # Collect labels
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                pass

        # Convert to numpy arrays
        y_true = np.array(y_true) if y_true else np.array([])
        y_pred = np.array(y_pred) if y_pred else np.array([])

        logger.info(f"Generated predictions for {len(y_true)} samples")

        return y_true, y_pred

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        计算全面的评估指标。

        参数：
            y_true: 真实标签
            y_pred: 预测标签

        返回：
            包含指标名称->值的字典

        说明：
        1. 计算总体准确率
        2. 计算宏平均精确率、召回率、F1
        3. 计算每类的精确率、召回率、F1
        4. 组织成字典
        5. 记录指标摘要
        6. 返回指标字典

        要计算的指标：
        - test_accuracy
        - test_precision (macro)
        - test_recall (macro)
        - test_f1 (macro)
        - per_class_precision (list)
        - per_class_recall (list)
        - per_class_f1 (list)
        """
        logger.info("Computing evaluation metrics...")

        metrics = {}

        # Compute overall metrics
        metrics['test_accuracy'] = accuracy_score(y_true, y_pred)
        metrics['test_precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['test_recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['test_f1'] = f1_score(y_true, y_pred, average='macro')

        # Compute per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None)
        per_class_recall = recall_score(y_true, y_pred, average=None)
        per_class_f1 = f1_score(y_true, y_pred, average=None)

        # Store per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = per_class_precision[i]
            metrics[f'{class_name}_recall'] = per_class_recall[i]
            metrics[f'{class_name}_f1'] = per_class_f1[i]

        # Log summary
        logger.info(f"Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
        logger.info(f"Test F1 Score: {metrics.get('test_f1', 0):.4f}")

        return metrics

    def generate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        生成混淆矩阵。

        参数：
            y_true: 真实标签
            y_pred: 预测标签

        返回：
            作为numpy数组的混淆矩阵

        TODO：
        1. 计算混淆矩阵
        2. 记录矩阵维度
        3. 返回矩阵
        """
        logger.info("Generating confusion matrix...")

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Log shape
        logger.info(f"Confusion matrix shape: {cm.shape}")

        return cm

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        将混淆矩阵绘制为热图。

        参数：
            cm: 混淆矩阵
            save_path: 可选的保存绘图路径

        返回：
            保存的绘图路径

        TODO：
        1. 创建图形
        2. 使用seaborn绘制热图
        3. 添加标签、标题、颜色条
        4. 保存图形
        5. 关闭图形
        6. 返回路径
        """
        logger.info("Plotting confusion matrix...")

        # Create figure
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        # Add labels
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save figure
        if save_path is None:
            save_path = self.plots_dir / 'confusion_matrix.png'

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrix plot to {save_path}")

        return save_path

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ) -> str:
        """
        生成详细的分类报告。

        参数：
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 可选的保存报告路径

        返回：
            作为字符串的分类报告

        TODO：
        1. 使用sklearn生成分类报告
        2. 如果提供了路径则保存到文件
        3. 记录报告
        4. 返回报告字符串
        """
        logger.info("Generating classification report...")

        # Generate report
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )

        # Save to file
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)

        # Log report
        logger.info(f"\nClassification Report:\n{report}")

        return report

    def save_metrics(
        self,
        metrics: Dict[str, float],
        save_path: Path
    ) -> None:
        """
        将指标保存到JSON文件。

        参数：
            metrics: 指标字典
            save_path: 保存JSON文件的路径

        说明：
        1. 创建父目录
        2. 保存为JSON
        3. 记录保存位置
        """
        logger.info("Saving metrics to file...")

        # Create directories
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {save_path}")

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        mlflow_tracker: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        运行完整的评估流程。

        参数：
            model: 要评估的训练模型
            test_loader: 测试数据加载器
            mlflow_tracker: 可选的用于记录的MLflow跟踪器

        返回：
            包含所有计算指标的字典

        说明：
        1. 生成预测
        2. 计算指标
        3. 生成混淆矩阵
        4. 绘制混淆矩阵
        5. 生成分类报告
        6. 保存所有工件
        7. 如果提供了跟踪器则记录到MLflow
        8. 返回指标

        这是运行所有评估步骤的主入口点。
        """
        logger.info("Starting comprehensive model evaluation...")

        # Step 1 - Generate predictions
        y_true, y_pred = self.predict(model, test_loader)

        # Step 2 - Compute metrics
        metrics = self.compute_metrics(y_true, y_pred)

        # Step 3 - Generate confusion matrix
        cm = self.generate_confusion_matrix(y_true, y_pred)

        # Step 4 - Plot confusion matrix
        cm_plot_path = self.plot_confusion_matrix(cm)

        # Step 5 - Generate classification report
        report_path = self.plots_dir / 'classification_report.txt'
        report = self.generate_classification_report(y_true, y_pred, report_path)

        logger.info(f"Classification Report:\n{report}")

        # Step 6 - Save metrics
        metrics_path = self.plots_dir / 'test_metrics.json'
        self.save_metrics(metrics, metrics_path)

        # Step 7 - Log to MLflow if available
        if mlflow_tracker:
            mlflow_tracker.log_metrics(metrics)
            mlflow_tracker.log_artifact(str(cm_plot_path))
            mlflow_tracker.log_artifact(str(report_path))
            mlflow_tracker.log_artifact(str(metrics_path))

        logger.info("Evaluation complete!")

        return metrics


# 示例用法
if __name__ == "__main__":
    """
    ModelEvaluator的示例用法。

    TODO：
    1. 加载训练好的模型
    2. 创建测试数据加载器
    3. 初始化评估器
    4. 运行评估
    5. 查看结果
    """

    # 配置
    config = {
        'plots_dir': 'evaluation_plots'
    }

    # 数据集的类别名称
    class_names = ['cat', 'dog', 'bird', 'fish']

    # 初始化评估器
    evaluator = ModelEvaluator(config, class_names)

    # 加载模型（占位符）
    model = torch.load('path/to/model.pth')

    # 创建测试加载器（占位符）
    test_loader = ...

    # 运行评估
    metrics = evaluator.evaluate(model, test_loader)

    # 打印结果
    print("\n评估指标：")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("ModelEvaluator模块已加载。")
