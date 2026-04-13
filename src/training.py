"""
带MLflow跟踪的模型训练模块

该模块处理模型训练，包含全面的MLflow实验跟踪。

学习目标：
- 实现PyTorch训练循环
- 集成MLflow进行实验跟踪
- 记录参数、指标和工件
- 保存和版本化模型
- 实现提前停止

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import time
import mlflow
import mlflow.pytorch
from datetime import datetime
# from data_ingestion import  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow跟踪操作的包装器。

    该类为所有MLflow操作提供了一个简洁的接口，
    使跟踪实验、记录指标和注册模型变得容易。

    属性：
        tracking_uri (str): MLflow跟踪服务器URI
        experiment_name (str): MLflow实验名称
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """
        初始化MLflow跟踪器。

        参数：
            tracking_uri: MLflow跟踪服务器URL（例如：'http://localhost:5000'）
            experiment_name: 实验名称

        TODO：
        1. 设置MLflow跟踪URI
        2. 设置或创建实验
        3. 存储实验名称
        4. 记录初始化
        """
        # 设置跟踪URI
        mlflow.set_tracking_uri(tracking_uri)

        # 设置实验（如果不存在则创建）
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        logger.info(f"为实验初始化MLflow跟踪器: {experiment_name}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        启动MLflow运行。

        参数：
            run_name: 运行的可选名称
            tags: 可选的附加到运行的标签

        返回：
            活动的MLflow运行对象

        TODO：
        1. 使用可选名称和标签启动MLflow运行
        2. 记录运行开始
        3. 返回运行对象
        """
        # 启动运行
        run = mlflow.start_run(run_name=run_name, tags=tags)

        # 记录运行ID
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # 返回运行
        return run

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        将参数记录到MLflow。

        参数：
            params: 参数名称->值对的字典

        TODO：
        1. 使用mlflow.log_params()记录所有参数
        2. 记录记录的参数数量
        """
        # 记录参数
        mlflow.log_params(params)

        logger.info(f"已将{len(params)}个参数记录到MLflow")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        将指标记录到MLflow。

        参数：
            metrics: 指标名称->值对的字典
            step: 可选的步骤编号（例如：轮次编号）

        TODO：
        1. 使用mlflow.log_metrics()记录所有指标
        2. 如果提供则包含step
        """
        # 记录指标
        mlflow.log_metrics(metrics, step=step)

        # 可选：记录单个指标日志消息
        if step is not None:
            logger.debug(f"在步骤{step}记录指标: {metrics}")

    def log_artifact(self, artifact_path: str) -> None:
        """
        将工件文件记录到MLflow。

        参数：
            artifact_path: 工件文件的路径

        TODO：
        1. 使用mlflow.log_artifact()记录工件
        2. 记录工件路径
        """
        # 记录工件
        mlflow.log_artifact(artifact_path)

        logger.info(f"记录工件: {artifact_path}")

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        **kwargs
    ) -> None:
        """
        将PyTorch模型记录到MLflow。

        参数：
            model: 要记录的PyTorch模型
            artifact_path: 保存模型的运行内路径
            **kwargs: mlflow.pytorch.log_model()的额外参数

        TODO：
        1. 使用mlflow.pytorch.log_model()记录模型
        2. 记录成功消息
        """
        # 记录模型
        mlflow.pytorch.log_model(model, artifact_path, **kwargs)

        logger.info(f"将模型记录到{artifact_path}")

    def end_run(self) -> None:
        """
        结束当前MLflow运行。

        TODO：
        1. 结束MLflow运行
        2. 记录结束消息
        """
        # 结束运行
        mlflow.end_run()

        logger.info("结束MLflow运行")


class ModelTrainer:
    """
    带MLflow跟踪的图像分类模型训练。

    该类处理完整的训练循环，包括验证、
    提前停止和全面的MLflow记录。

    属性：
        config (Dict[str, Any]): 配置字典
        tracker (MLflowTracker): MLflow跟踪器实例
        device (torch.device): 训练设备（CPU/GPU）
    """

    def __init__(
        self,
        config: Dict[str, Any],
        mlflow_tracker: MLflowTracker
    ) -> None:
        """
        初始化ModelTrainer。

        参数：
            config: 配置字典
            mlflow_tracker: 已初始化的MLflowTracker实例

        TODO：
        1. 存储配置和跟踪器
        2. 确定设备（如果可用则为GPU，否则为CPU）
        3. 记录设备信息
        """
        self.config = config
        self.tracker = mlflow_tracker

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"ModelTrainer已初始化。使用设备: {self.device}")

    def create_model(
        self,
        num_classes: int,
        model_name: str = "resnet18"
    ) -> nn.Module:
        """
        创建模型架构。

        参数：
            num_classes: 输出类数
            model_name: 模型架构名称（'resnet18'，'mobilenet_v2'）

        返回：
            PyTorch模型
            
        1. 根据model_name创建模型
        2. 修改最终层以适应num_classes
        3. 将模型移动到设备
        4. 记录模型架构
        5. 返回模型

        支持的模型：
        - resnet18: 良好的基线，11M参数
        - mobilenet_v2: 轻量级，3.5M参数
        """
        logger.info(f"创建{model_name}模型，类数为{num_classes}")

        # 根据名称创建模型
        if model_name == "resnet18":
            # 加载预训练的ResNet18
            model = models.resnet18(pretrained=True)
            # 获取最终层的输入特征数
            num_features = model.fc.in_features
            # 替换最终层
            model.fc = nn.Linear(num_features, num_classes)
            pass

        elif model_name == "mobilenet_v2":
            # 加载预训练的MobileNetV2
            model = models.mobilenet_v2(pretrained=True)
            # 获取输入特征数
            num_features = model.classifier[1].in_features
            # 替换最终层
            model.classifier[1] = nn.Linear(num_features, num_classes)
            pass

        else:
            raise ValueError(f"未知模型: {model_name}")

        # 将模型移动到设备
        model = model.to(self.device)

        # 记录模型信息
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"创建{model_name}，参数数为{total_params:,}")

        return model

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """
        训练一个轮次。

        参数：
            model: PyTorch模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器

        返回：
            (平均损失，准确率)元组

        1. 将模型设置为训练模式
        2. 初始化指标（损失，正确预测）
        3. 遍历批次
        4. 对于每个批次：
           - 将数据移动到设备
           - 前向传播
           - 计算损失
           - 反向传播
           - 更新权重
           - 跟踪指标
        5. 计算轮次统计
        6. 返回平均损失和准确率
        """
        # 将模型设置为训练模式
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # 遍历批次
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将数据移动到设备
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 清零梯度
            optimizer.zero_grad()   

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            # 跟踪指标
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 可选：每N个批次记录进度
            if batch_idx % 10 == 0:
                logger.debug(f"批次 {batch_idx}/{len(train_loader)}，损失: {0:.4f}")

        # 计算轮次统计
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        验证模型。

        参数：
            model: PyTorch模型
            val_loader: 验证数据加载器
            criterion: 损失函数

        返回：
            (平均损失，准确率)元组

        TODO：
        1. 将模型设置为评估模式
        2. 禁用梯度计算
        3. 遍历批次
        4. 对于每个批次：
           - 将数据移动到设备
           - 前向传播（无反向传播！）
           - 计算损失
           - 跟踪指标
        5. 计算验证统计
        6. 返回平均损失和准确率
        """
        # 将模型设置为评估模式
        model.eval()


        val_loss = 0.0
        correct = 0
        total = 0

        # 禁用梯度计算
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 将数据移动到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = model(inputs)

                # 计算损失    
                loss = criterion(outputs, targets)

                # 跟踪指标
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pass

        # 计算验证统计
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        return avg_val_loss, val_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        params: Dict[str, Any]
    ) -> Tuple[nn.Module, float]:
        """
        带MLflow跟踪的完整训练管道。

        参数：
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_classes: 输出类数
            params: 训练超参数

        返回：
            (训练好的模型，最佳验证准确率)元组

        1. 启动MLflow运行
        2. 记录参数
        3. 创建模型、损失函数、优化器
        4. 训练循环：
           - 训练轮次
           - 验证
           - 记录指标到MLflow
           - 保存最佳模型
           - 检查提前停止
        5. 记录最终模型和工件
        6. 结束MLflow运行
        7. 返回模型和最佳准确率

        预期的params：
        - model_name: str
        - num_epochs: int
        - batch_size: int
        - learning_rate: float
        - optimizer: str ('adam', 'sgd')
        - lr_step_size: int (用于调度器)
        - lr_gamma: float (用于调度器)
        - early_stopping_patience: int
        """
        logger.info("开始训练管道")

        # 启动MLflow运行
        run_name = f"train_{params['model_name']}_{int(time.time())}"
        tags = {
            "model_architecture": params['model_name'],
            "framework": "pytorch",
            "task": "image_classification"
        }
        self.tracker.start_run(run_name=run_name, tags=tags)

        # 记录所有参数
        self.tracker.log_params(params)

        # 创建模型
        model = self.create_model(num_classes, params['model_name'])

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 创建优化器
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)

        # 创建学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
                  optimizer,
                  step_size=params['lr_step_size'],
                  gamma=params['lr_gamma']
              )

        #  初始化跟踪变量
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_model_path = Path(self.config['model_save_path']) / 'best_model.pth'

        #  训练循环
        for epoch in range(params['num_epochs']):
            epoch_start = time.time()

            logger.info(f"轮次 {epoch + 1}/{params['num_epochs']}")

            #  训练轮次
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)

            #  验证
            val_loss, val_acc = self.validate(model, val_loader, criterion)

            #  更新学习率
            scheduler.step()

            #  记录指标到MLflow
            self.tracker.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

            #  保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
                logger.info(f"新的最佳模型！验证准确率: {val_acc:.2f}%")
            else:
                epochs_without_improvement += 1

            #  提前停止检查
            if epochs_without_improvement >= params.get('early_stopping_patience', 5):
                logger.info(f"在{epoch + 1}轮后触发提前停止")
                break

            #  记录轮次时间
            epoch_time = time.time() - epoch_start
            logger.info(f"轮次在{epoch_time:.2f}s内完成")

        #  记录最佳模型到MLflow
        self.tracker.log_model(model, "model")
        self.tracker.log_artifact(str(best_model_path))

        #  记录最终指标
        self.tracker.log_metrics({
            'best_val_accuracy': best_val_acc,
            'total_epochs': epoch + 1
        })

        #  结束MLflow运行
        self.tracker.end_run()

        logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")

        #  返回模型和最佳准确率
        return model, best_val_acc


# 示例用法和测试
if __name__ == "__main__":
    """
    ModelTrainer类的示例用法。

    说明：
    1. 设置MLflow跟踪器
    2. 创建示例数据加载器
    3. 初始化ModelTrainer
    4. 运行训练
    5. 在MLflow UI中查看结果

    查看结果：
    - 启动MLflow UI: mlflow ui --port 5000
    - 打开浏览器: http://localhost:5000
    """

    #  MLflow配置
    mlflow_config = {
        'tracking_uri': 'http://localhost:5000',
        'experiment_name': 'image_classification_test'
    }

    #  训练配置
    training_config = {
        'model_save_path': 'models',
        'num_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_name': 'resnet18',
        'optimizer': 'adam',
        'lr_step_size': 3,
        'lr_gamma': 0.1,
        'early_stopping_patience': 3
    }

    #  初始化MLflow跟踪器
    tracker = MLflowTracker(
        tracking_uri=mlflow_config['tracking_uri'],
        experiment_name=mlflow_config['experiment_name']
    )

    #  初始化训练器
    trainer = ModelTrainer(training_config, tracker)

    #  创建示例数据加载器
    # 注意：您需要使用实际数据创建实际的DataLoaders
    train_loader = None
    val_loader = None
    # 目前，这只是占位符

    #  运行训练
    model, best_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=4,
        params=training_config
    )

    print("ModelTrainer模块已加载。实现TODO以完成功能。")
    print("别忘了启动MLflow服务器: mlflow server --port 5000")