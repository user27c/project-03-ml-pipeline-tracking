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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _denorm_cifar_style(tensor_chw: torch.Tensor) -> np.ndarray:
    """与 dataset.py 中 eval 变换一致: Normalize(0.5,0.5) -> 显示 RGB [0,1]。"""
    x = tensor_chw.detach().float().cpu().clone()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    x = (x * std + mean).clamp(0.0, 1.0)
    return x.numpy().transpose(1, 2, 0)


def _forward_activation_shape(
    model: nn.Module, layer: nn.Module, x: torch.Tensor
) -> Optional[Tuple[int, int, int, int]]:
    """对单层注册 hook，跑一次完整前向，返回该层输出 shape (N,C,H,W)；非 4D 则 None。"""
    shape_holder: list = []

    def _hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        if torch.is_tensor(out) and out.dim() == 4:
            shape_holder.append(tuple(out.shape))

    h = layer.register_forward_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    return shape_holder[0] if shape_holder else None


def _pick_gradcam_layer(
    model: nn.Module, x_sample: torch.Tensor, arch_hint: Optional[str] = None
) -> Optional[nn.Module]:
    """
    从靠近输出的卷积块向前搜索，选用空间尺寸 >=2 的最深一层（小输入下避免 1x1 特征图）。
    """
    candidates: List[nn.Module] = []
    if hasattr(model, "layer1"):
        candidates.extend(
            [model.layer1, model.layer2, model.layer3, model.layer4]
        )
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        candidates.extend(list(model.features.children()))
    if not candidates:
        return None

    device = next(model.parameters()).device
    x = x_sample[:1].to(device)
    chosen: Optional[nn.Module] = None
    chosen_shape: Optional[Tuple[int, int, int, int]] = None
    for lyr in reversed(candidates):
        if not any(True for _ in lyr.parameters()):
            continue
        try:
            sh = _forward_activation_shape(model, lyr, x)
        except Exception:
            continue
        if not sh or len(sh) != 4:
            continue
        _, _, hh, ww = sh
        if hh < 2 or ww < 2:
            continue
        chosen = lyr
        chosen_shape = sh
        break
    if chosen is not None:
        logger.info(
            "Grad-CAM 使用目标层: %s (arch_hint=%r, act shape=%s)",
            type(chosen).__name__,
            arch_hint,
            chosen_shape,
        )
    else:
        logger.warning(
            "Grad-CAM: 未找到合适卷积层 (arch_hint=%r)",
            arch_hint,
        )
    return chosen


class _GradCAM:
    """标准 Grad-CAM：对目标层激活与梯度做通道加权。"""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fh = target_layer.register_forward_hook(self._save_act)
        self._bh = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(
        self, _m: nn.Module, _inp: Any, out: torch.Tensor
    ) -> None:
        self.activations = out

    def _save_grad(
        self, _m: nn.Module, _grad_in: Any, grad_out: Any
    ) -> None:
        if grad_out[0] is not None:
            self.gradients = grad_out[0]

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()

    def compute(
        self, x: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        x: (1, C, H, W)。返回 (H_img, W_img) 的归一化 CAM，与输入图同空间尺寸。
        """
        self.model.eval()
        self.activations = None
        self.gradients = None
        x = x.detach().requires_grad_(True)
        out = self.model(x)
        if target_class is None:
            target_class = int(out.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        score = out[0, target_class]
        score.backward(retain_graph=False)
        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM: 未捕获到激活或梯度")
        # acts, grads: (1, C, h, w)
        a0 = acts[0]
        g0 = grads[0]
        weights = g0.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * a0).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_up = F.interpolate(
            cam.view(1, 1, *cam.shape),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return cam_up.detach().float().cpu().numpy()


def _overlay_heatmap_on_image(
    rgb_01: np.ndarray, cam_hw: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """rgb_01: H,W,3 in [0,1]; cam_hw: H,W in [0,1]"""
    heat = plt.cm.jet(cam_hw)[:, :, :3]
    out = (1.0 - alpha) * rgb_01 + alpha * heat
    return np.clip(out, 0.0, 1.0)


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create plots directory
        self.plots_dir = Path(config.get("plots_dir", "plots"))
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

    def plot_prediction_samples(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[Path] = None,
    ) -> Path:
        """网格展示若干测试样本：真值 vs 预测（与 dataset 评估变换一致的反归一化显示）。"""
        model.eval()
        cap = int(self.config.get("eval_prediction_samples", 12))
        n_tot = images.size(0)
        n = max(1, min(cap, n_tot))
        images = images[:n].to(self.device)
        labels = labels[:n].to(self.device)
        with torch.no_grad():
            pred = model(images).argmax(dim=1)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
        axes_flat = np.atleast_1d(axes).ravel()
        for i in range(n):
            ax = axes_flat[i]
            img = _denorm_cifar_style(images[i])
            ax.imshow(img)
            ti, pi = int(labels[i]), int(pred[i])
            tn, pn = self.class_names[ti], self.class_names[pi]
            ok = ti == pi
            ax.set_title(
                f"T: {tn}\nP: {pn}",
                fontsize=8,
                color=("green" if ok else "red"),
            )
            ax.axis("off")
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")
        fig.suptitle("Prediction samples (T=true, P=predicted)", fontsize=11, y=1.02)
        plt.tight_layout()
        path = Path(save_path) if save_path else self.plots_dir / "prediction_samples.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved prediction samples to %s", path)
        return path

    def plot_gradcam_samples(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        model_arch: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """对若干样本做 Grad-CAM（解释模型预测类别）并叠加热力图。"""
        cap = int(self.config.get("eval_gradcam_samples", 6))
        n = min(cap, images.size(0))
        if n < 1:
            return None
        model = model.to(self.device)
        images = images[:n].to(self.device)
        labels_cpu = labels[:n].detach().cpu()
        layer = _pick_gradcam_layer(model, images, model_arch)
        if layer is None:
            return None
        gc = _GradCAM(model, layer)
        path: Optional[Path] = None
        try:
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
            axes_flat = np.atleast_1d(axes).ravel()
            for i in range(n):
                xi = images[i : i + 1].detach()
                with torch.no_grad():
                    pred_cls = int(model(xi).argmax(dim=1).item())
                cam = gc.compute(xi, target_class=pred_cls)
                rgb = _denorm_cifar_style(images[i])
                overlay = _overlay_heatmap_on_image(rgb, cam, alpha=0.48)
                ax = axes_flat[i]
                ax.imshow(overlay)
                ti = int(labels_cpu[i].item())
                ax.set_title(
                    f"P: {self.class_names[pred_cls]}\n"
                    f"T: {self.class_names[ti]}",
                    fontsize=8,
                )
                ax.axis("off")
            for j in range(n, len(axes_flat)):
                axes_flat[j].axis("off")
            fig.suptitle(
                "Grad-CAM (explains predicted class)", fontsize=11, y=1.02
            )
            plt.tight_layout()
            path = Path(save_path) if save_path else self.plots_dir / "gradcam_samples.png"
            plt.savefig(path, dpi=200, bbox_inches="tight")
            plt.close()
            logger.info("Saved Grad-CAM samples to %s", path)
        except Exception as e:
            logger.warning("Grad-CAM 生成失败，跳过: %s", e)
            path = None
        finally:
            gc.remove()
        return path

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        mlflow_tracker: Optional[Any] = None,
        model_arch: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        运行完整的评估流程。

        参数：
            model: 要评估的训练模型
            test_loader: 测试数据加载器
            mlflow_tracker: 可选的用于记录的MLflow跟踪器
            model_arch: 可选，如 ``mobilenet_v2`` / ``resnet18``，用于 Grad-CAM 层选择日志与调试

        返回：
            包含所有计算指标的字典

        说明：
        1. 生成预测
        2. 计算指标
        3. 生成混淆矩阵
        4. 绘制混淆矩阵
        5. 生成分类报告
        6. 保存所有工件
        7. 使用 test_loader 的首个 batch 生成 ``prediction_samples.png`` 与 ``gradcam_samples.png``
           （数量可由 config 中 ``eval_prediction_samples`` / ``eval_gradcam_samples`` 覆盖）
        8. 如果提供了跟踪器则记录到MLflow
        9. 返回指标

        这是运行所有评估步骤的主入口点。
        """
        logger.info("Starting comprehensive model evaluation...")

        try:
            viz_images, viz_labels = next(iter(test_loader))
        except StopIteration:
            viz_images, viz_labels = None, None  # type: ignore

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

        samples_path: Optional[Path] = None
        gradcam_path: Optional[Path] = None
        if viz_images is not None and viz_labels is not None:
            samples_path = self.plot_prediction_samples(
                model, viz_images, viz_labels
            )
            gradcam_path = self.plot_gradcam_samples(
                model, viz_images, viz_labels, model_arch=model_arch
            )

        # Step 7 - Log to MLflow if available
        if mlflow_tracker:
            mlflow_tracker.log_metrics(metrics)
            mlflow_tracker.log_artifact(str(cm_plot_path))
            mlflow_tracker.log_artifact(str(report_path))
            mlflow_tracker.log_artifact(str(metrics_path))
            if samples_path and samples_path.is_file():
                mlflow_tracker.log_artifact(str(samples_path))
            if gradcam_path and gradcam_path.is_file():
                mlflow_tracker.log_artifact(str(gradcam_path))

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
