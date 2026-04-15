话说，mobilenet_v2模型不是训练好的吗？如果没训练，意思是只有参数吗？训练又是怎么训练，什么过程呢？我在airflow训练了，为什么还要mlflow?

好问题，分三层讲最清楚：

## 1) `mobilenet_v2` “训练好了吗”是什么意思？

在你项目里，`mobilenet_v2` 是：

- 先加载 **预训练权重**（通常是 ImageNet 上训练好的通用特征）
- 再把最后分类层替换成你任务的类别数
- 然后在你自己的数据上继续训练（fine-tune）

所以它不是“完全没训练”，也不是“直接能用你的类别”。  
更准确说：**有通用参数，但还需要在你的数据上再训练，学会你的标签分布。**

---

## 2) 训练过程到底在干嘛？

每个 epoch 大致是：

1. 从 `train_loader` 按 batch 读图像和标签  
2. 前向计算得到预测  
3. 用损失函数（交叉熵）算预测和真实标签差距  
4. 反向传播算梯度  
5. 优化器（Adam/SGD）更新模型参数  
6. 在验证集上算 `val_loss/val_accuracy`  
7. 如果更好就保存 `best_model.pth`  
8. 记录指标到 MLflow（你项目是每个 epoch 记录一次）

所以“训练”本质上是：**反复更新参数，让模型在你的任务上误差变小。**

---

## 3) 既然 Airflow 在训练，为什么还要 MLflow？

一句话：**Airflow 管流程，MLflow 管实验。**

- **Airflow**：负责“按顺序执行哪些步骤”
  - 例如 ingest → preprocess → train → evaluate → register
  - 负责调度、重试、依赖
- **MLflow**：负责“每次训练到底用了什么，结果怎样”
  - 参数（lr、batch_size、model）
  - 指标（train/val loss/acc）
  - 工件（模型、曲线、ONNX、metrics.json）
  - 模型注册（版本、阶段）

没有 MLflow，你只能知道“任务跑没跑完”；  
有 MLflow，你能回答“哪次训练最好、为什么最好、能否复现、该发布哪个版本”。

---

如果你愿意，我下一步可以给你画一张你这个项目的“Airflow vs MLflow 职责图”（放进 `docs/technology`），一眼就能讲清楚给面试官听。