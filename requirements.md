# 项目要求：带实验跟踪的ML管道

**版本：** 1.0
**最后更新：** 2025年10月18日

---

## 目录

1. [功能需求](#功能需求)
2. [非功能需求](#非功能需求)
3. [技术规格](#技术规格)
4. [验收标准](#验收标准)
5. [约束和假设](#约束和假设)

---

## 功能需求

### FR-1：数据管道

#### FR-1.1：数据摄取

**描述：** 系统必须从多个源摄取数据

**需求：**

- 支持CSV文件摄取
- 支持REST API数据获取
- 支持数据库查询（PostgreSQL）
- 处理不同模式的数据
- 使用时间戳保存原始数据

**输入：**

- 来自本地文件系统或云存储的CSV文件
- 返回JSON的REST API端点
- 针对PostgreSQL数据库的SQL查询

**输出：**

- 原始数据保存到`data/raw/`目录
- 包含记录计数的摄取日志
- 关于数据源和时间戳的元数据

**验收标准：**

- 成功从CSV摄取100K+记录
- 从公共API获取数据（例如OpenML、Kaggle API）
- 从PostgreSQL数据库查询数据
- 使用重试优雅地处理网络错误
- 记录所有摄取活动

---

#### FR-1.2：数据验证

**描述：** 使用Great Expectations验证数据质量

**需求：**

- 为数据集模式定义期望套件
- 验证行数（最小/最大边界）
- 验证列的存在和类型
- 验证值范围和类别
- 检测必需列中的空值
- 生成验证报告

**验证规则：**

- 最少1,000条记录，最多1,000,000条记录
- 必需列：`image_path`、`label`、`split`
- 标签值必须在定义的集合中（例如"cat", "dog", "bird", "fish"]）
- 关键列中无空值
- 图像文件路径必须存在

**输出：**

- 验证报告（HTML）
- 布尔验证结果（通过/失败）
- 失败的详细错误消息

**验收标准：**

- 创建了5+期望的期望套件
- 验证捕获模式违规
- 验证捕获缺失值
- 验证捕获无效标签
- 如果验证失败则停止管道

---

#### FR-1.3：数据版本控制

**描述：** 使用DVC（数据版本控制）版本化数据集

**需求：**

- 在项目中初始化DVC
- 配置远程存储（MinIO/S3）
- 跟踪原始和处理后的数据集
- 使用有意义的名称标记版本
- 启用从任何版本检索数据

**DVC工作流：**

```bash
# 将数据集添加到DVC
dvc add data/raw/dataset.csv

# 将DVC文件提交到Git
git add data/raw/dataset.csv.dvc
git commit -m "Dataset v1.0 - Initial import"
git tag -a "data-v1.0" -m "Dataset version 1.0"

# 推送到远程
dvc push
```

**验收标准：**

- DVC已初始化并配置
- 原始数据使用DVC跟踪
- 处理后的数据使用DVC跟踪
- 远程存储已配置（MinIO）
- 可以使用`dvc pull`检索数据
- 至少3个数据版本已标记

---

#### FR-1.4：数据预处理

**描述：** 清洗和预处理数据以用于模型训练

**需求：**

- 删除重复记录
- 处理缺失值（删除或填充）
- 编码分类标签
- 归一化/标准化特征
- 创建训练/验证/测试分割（70/15/15）
- 保存预处理工件（编码器、标准化器）

**预处理步骤：**

1. **清洗：**
  - 删除完全重复的记录
  - 删除缺失关键值的行
  - 修复数据类型不一致
2. **编码：**
  - 标签编码分类目标
  - 保存编码器以供推理使用
3. **分割：**
  - 分层分割以保持类别平衡
  - 固定随机种子以实现可复现性

**输出：**

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `artifacts/label_encoder.pkl`
- `artifacts/scaler.pkl`（如果适用）

**验收标准：**

- 预处理删除重复项
- 缺失值适当处理
- 标签编码为整数
- 数据分割为train/val/test
- 分割大小遵循70/15/15比例
- 编码器保存以供重用
- 预处理使用固定种子可复现

---

#### FR-1.5：特征存储（可选）

**描述：** 在PostgreSQL中存储处理后的特征

**需求：**

- 在PostgreSQL中创建特征表
- *使用ID存储特征向量*
- *启用按ID检索特征*
- 跟踪特征版本

**模式：**

```sql
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE,
    feature_vector FLOAT[],
    label INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**验收标准：**

- 特征表已创建
- 特征成功存储
- 特征可按ID检索
- 特征版本已跟踪

---

### FR-2：模型训练管道

#### FR-2.1：模型训练

**描述：** 使用PyTorch训练图像分类模型

**需求：**

- 支持多种模型架构（ResNet18、MobileNetV2）
- 实现带验证的训练循环
- 支持迁移学习（预训练模型）
- 跟踪训练进度（损失、准确率）
- 根据验证准确率保存最佳模型
- 支持早停

**模型架构：**

1. **ResNet18**（基线）
2. **MobileNetV2**（轻量级）

**训练配置：**

```python
{
    "model_name": "resnet18",
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "lr_scheduler": "step_lr",
    "lr_step_size": 7,
    "lr_gamma": 0.1,
    "early_stopping_patience": 5
}
```

**验收标准：**

- 模型成功训练
- 训练和验证损失下降
- 模型达到>75%验证准确率
- 根据验证准确率保存最佳模型
- 每个实验的训练在<30分钟内完成

---

#### FR-2.2：超参数调优

**描述：** 系统超参数优化

**需求：**

- 支持网格搜索或Optuna优化
- 调优至少3个超参数
- 运行最少10种配置
- 在MLflow中跟踪所有实验

**要调优的超参数：**

- 学习率：0.0001, 0.001, 0.01]
- 批大小：16, 32, 64]
- 模型架构：resnet18, mobilenetv2]
- 优化器：adam, sgd]

**搜索空间：**

```python
{
    "learning_rate": [0.0001, 0.001, 0.01],
    "batch_size": [16, 32, 64],
    "model_name": ["resnet18", "mobilenet_v2"],
    "optimizer": ["adam", "sgd"]
}
```

**验收标准：**

- 最少10个实验使用不同配置
- 所有实验在MLflow中跟踪
- 确定最佳超参数
- 系统比较结果

---

#### FR-2.3：实验跟踪

**描述：** 使用MLflow跟踪所有训练运行

**需求：**

- 记录所有超参数
- 每个epoch记录指标（训练/验证损失、准确率）
- 记录最终评估指标
- 保存模型工件
- 使用有意义的标识符标记运行
- 捕获代码版本（git提交哈希）

**记录内容：**

**参数：**

```python
{
    "model_name": "resnet18",
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "dataset_version": "v1.2",
    "git_commit": "a3b4c5d"
}
```

**指标（每个epoch）：**

```python
{
    "train_loss": 0.345,
    "train_accuracy": 87.5,
    "val_loss": 0.412,
    "val_accuracy": 85.2,
    "learning_rate": 0.001
}
```

**工件：**

- `model.pth` - 保存的模型权重
- `training_plot.png` - 损失/准确率曲线
- `confusion_matrix.png` - 分类结果
- `metrics.json` - 最终评估指标

**验收标准：**

- 所有参数已记录
- 每个epoch的指标已记录
- 模型工件已保存
- 运行已适当标记
- Git提交哈希已捕获
- MLflow UI显示所有实验

---

#### FR-2.4：模型工件

**描述：** 保存和组织模型工件

**需求：**

- 以PyTorch格式保存模型权重
- 保存ONNX导出用于推理
- 保存训练图（损失/准确率曲线）
- 保存评估指标（JSON）
- 按运行ID组织工件

**工件结构：**

```
mlruns/
└── 0/                          # 实验ID
    └── abc123def456/           # 运行ID
        └── artifacts/
            ├── model/
            │   ├── model.pth           # PyTorch权重
            │   ├── model.onnx          # ONNX导出
            │   └── config.json         # 模型配置
            ├── plots/
            │   ├── training_curves.png
            │   └── confusion_matrix.png
            └── metrics/
                └── evaluation.json
```

**验收标准：**

- 模型权重已保存
- ONNX导出已创建
- 训练图已生成
- 指标已保存为JSON
- 工件已正确组织

---

#### FR-2.5：模型评估

**描述：** 在保留的测试集上评估模型

**需求：**

- 在测试集上评估（训练期间从未见过）
- 计算多个指标（准确率、精确率、召回率、F1）
- 生成混淆矩阵
- 生成分类报告
- 与基线比较

**要计算的指标：**

- 准确率
- 精确率（每个类别和宏平均）
- 召回率（每个类别和宏平均）
- F1分数（每个类别和宏平均）
- 混淆矩阵

**评估输出：**

```python
{
    "test_accuracy": 86.4,
    "test_precision": 85.7,
    "test_recall": 86.1,
    "test_f1": 85.9,
    "per_class_metrics": {
        "cat": {"precision": 88.2, "recall": 89.1, "f1": 88.6},
        "dog": {"precision": 87.5, "recall": 86.8, "f1": 87.1},
        "bird": {"precision": 82.1, "recall": 83.0, "f1": 82.5},
        "fish": {"precision": 85.0, "recall": 85.5, "f1": 85.2}
    }
}
```

**验收标准：**

- 已执行测试集评估
- 已计算多个指标
- 已生成混淆矩阵
- 已计算每个类别的指标
- 结果已记录到MLflow

---

### FR-3：模型注册

#### FR-3.1：模型注册

**描述：** 在MLflow模型注册表中注册模型

**需求：**

- 成功训练后注册模型
- 分配语义版本
- 添加模型元数据（描述、标签）
- 链接到训练运行
- 跟踪注册时间戳

**注册流程：**

```python
# 注册模型
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(
    model_uri=model_uri,
    name="image_classifier",
    tags={
        "architecture": "resnet18",
        "dataset_version": "v1.2",
        "framework": "pytorch"
    }
)

# 添加描述
client.update_model_version(
    name="image_classifier",
    version=result.version,
    description="ResNet18 trained on v1.2 dataset, 86.4% test accuracy"
)
```

**验收标准：**

- 模型成功注册
- 版本按顺序编号
- 元数据附加到模型
- 运行链接保持
- 注册在MLflow UI中可见

---

#### FR-3.2：模型生命周期阶段

**描述：** 通过阶段管理模型生命周期

**需求：**

- 支持阶段转换（None → Staging → Production → Archived）
- 跟踪阶段历史
- 同时只允许一个Production模型
- Production升级需要批准（手动步骤）

**生命周期阶段：**

1. **None** - 新注册的模型
2. **Staging** - 正在测试的模型
3. **Production** - 当前部署的模型
4. **Archived** - 已弃用的模型

**阶段转换：**

```python
client.transition_model_version_stage(
    name="image_classifier",
    version=3,
    stage="Production",
    archive_existing_versions=True  # 归档之前的Production模型
)
```

**验收标准：**

- 模型可以在阶段之间转换
- 同时只有一个Production模型
- 阶段历史已跟踪
- 之前的Production模型自动归档

---

#### FR-3.3：模型元数据

**描述：** 将全面的元数据附加到模型

**需求：**

- 记录训练数据集版本
- 记录评估指标
- 记录超参数
- 记录代码版本（git提交）
- 记录训练持续时间
- 记录使用的硬件

**元数据结构：**

```python
{
    "dataset_version": "v1.2",
    "data_size": 50000,
    "test_accuracy": 86.4,
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 20
    },
    "code_version": "a3b4c5d",
    "training_duration": 1847,  # 秒
    "hardware": "NVIDIA Tesla T4",
    "trained_by": "airflow-pipeline",
    "training_date": "2025-10-18T14:32:00Z"
}
```

**验收标准：**

- 所有元数据字段已填充
- 元数据在MLflow中可搜索
- 血统可追溯

---

#### FR-3.4：模型检索API

**描述：** 检索生产模型的API

**需求：**

- 获取最新的Production模型
- 获取特定模型版本
- 下载模型工件
- 检索模型元数据

**API示例：**

```python
# 获取最新的Production模型
model = mlflow.pyfunc.load_model(
    model_uri="models:/image_classifier/Production"
)

# 获取特定版本
model = mlflow.pyfunc.load_model(
    model_uri="models:/image_classifier/3"
)

# 获取模型元数据
client = mlflow.tracking.MlflowClient()
model_version = client.get_model_version(
    name="image_classifier",
    version="3"
)
print(model_version.description)
```

**验收标准：**

- 可以加载Production模型
- 可以加载特定版本
- 可以检索元数据
- 模型已准备好用于推理

---

### FR-4：工作流编排

#### FR-4.1：Airflow DAG设计

**描述：** 将ML管道实现为Airflow DAG

**需求：**

- 定义任务依赖关系
- 实现任务重试
- 配置任务超时
- 在任务之间传递数据（XCom）
- 在Airflow UI中可视化DAG

**DAG结构：**

```python
ingest_data
    ↓
validate_data
    ↓
preprocess_data
    ↓
version_data_dvc
    ↓
train_model
    ↓
evaluate_model
    ↓
register_model
    ↓
notify_success
```

**任务配置：**

```python
default_args = {
    'owner': 'ml-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'email_on_failure': True,
    'email': ['mlops@example.com']
}
```

**验收标准：**

- DAG在Airflow UI中可见
- 任务依赖关系正确
- 任务按顺序执行
- 失败的任务自动重试
- DAG可以手动触发

---

#### FR-4.2：管道调度

**描述：** 调度自动管道执行

**需求：**

- 按计划运行管道（每周、每天等）
- 支持手动触发
- 支持参数化运行
- 处理并发运行

**调度配置：**

```python
dag = DAG(
    'ml_training_pipeline',
    schedule_interval='@weekly',  # 每周日凌晨午夜
    catchup=False,  # 不为过去日期运行
    max_active_runs=1,  # 无并发运行
    start_date=datetime(2025, 10, 1)
)
```

**验收标准：**

- 管道按计划运行
- 手动触发有效
- catchup已禁用
- 防止并发运行

---

#### FR-4.3：错误处理

**描述：** 强大的错误处理和恢复

**需求：**

- 重试暂时性故障（最多3次）
- 发送故障通知（电子邮件/Slack）
- 使用堆栈跟踪记录错误
- 尽可能继续下游任务
- 实现断路器

**错误处理：**

```python
@task(retries=3, retry_delay=timedelta(minutes=5))
def train_model(**context):
    try:
        # 训练代码
        pass
    except MemoryError as e:
        logger.error(f"OOM error: {e}")
        raise AirflowFailException("Insufficient memory")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
```

**验收标准：**

- 失败的任务重试3次
- 错误已详细记录
- 发送故障电子邮件
- 管道在关键故障时停止
- 尽可能优雅降级

---

#### FR-4.4：管道监控

**描述：** 监控管道执行和健康状况

**需求：**

- 跟踪任务持续时间
- 监控成功/失败率
- SLA违规时发出警报
- 可视化管道指标
- 生成执行报告

**监控指标：**

- 任务持续时间（平均值、p95、p99）
- 成功率（最近7天、30天）
- 按任务划分的失败率
- 管道端到端持续时间
- 资源使用情况（CPU、内存）

**验收标准：**

- 任务持续时间已记录
- 成功率已跟踪
- SLA违规已检测
- 指标在Airflow UI中可视化
- 生成每周执行报告

---

#### FR-4.5：通知

**描述：** 通知团队管道状态

**需求：**

- 故障时发送电子邮件通知
- 成功/失败时发送Slack通知
- 在通知中包含运行详细信息
- 附加故障日志
- 成功时生成摘要报告

**通知内容：**

**故障电子邮件：**

```
Subject: [FAILURE] ML Training Pipeline - 2025-10-18

Pipeline: ml_training_pipeline
Status: FAILED
Failed Task: train_model
Execution Date: 2025-10-18 14:30:00
Duration: 45 minutes

Error: OutOfMemoryError during model training

Logs: [Attached]
Airflow Link: http://airflow:8080/dags/ml_training_pipeline/grid
```

**成功电子邮件：**

```
Subject: [SUCCESS] ML Training Pipeline - 2025-10-18

Pipeline: ml_training_pipeline
Status: SUCCESS
Execution Date: 2025-10-18 14:30:00
Duration: 2 hours 15 minutes

Results:
- Test Accuracy: 86.4%
- Model Version: 12
- Model Stage: Staging

MLflow Link: http://mlflow:5000/experiments/0
```

**验收标准：**

- 发送故障通知
- 发送成功通知
- 通知包含相关详细信息
- 包含Airflow和MLflow链接

---

## 非功能需求

### NFR-1：可复现性

**描述：** 所有实验必须完全可复现

**需求：**

- 所有操作的固定随机种子
- 代码版本控制（Git）
- 数据版本控制（DVC）
- 固定依赖版本
- 文档化环境设置

**可复现性检查清单：**

- 固定随机种子（`torch.manual_seed(42)`）
- 依赖项在`requirements.txt`中固定
- 数据使用DVC版本化
- 代码使用Git版本化
- 环境文档化（Docker/Conda）
- 实验可以从元数据复现

---

### NFR-2：性能

**描述：** 管道必须达到性能目标

**性能目标：**

- 数据摄取：在<10分钟内处理100K记录
- 数据验证：在<5分钟内完成
- 预处理：在<10分钟内完成
- 训练：每个实验在<30分钟内完成
- MLflow UI：在<2秒内加载实验
- Airflow：处理10+并发任务

**验收标准：**

- 数据管道在<25分钟内完成
- 训练在<30分钟内完成
- MLflow UI响应迅速（<2秒加载）
- Airflow处理并发任务
- 长时间运行时无内存泄漏

---

### NFR-3：可靠性

**描述：** 管道必须高度可靠

**可靠性目标：**

- 管道成功率：>95%
- 自动重试暂时性故障
- 数据验证防止不良数据
- 监控异常时发出警报

**验收标准：**

- 20次运行中成功率>95%
- 暂时性故障自动重试
- 不良数据被验证捕获
- 监控警报正常工作

---

### NFR-4：可扩展性

**描述：** 系统必须能够扩展以处理增长

**可扩展性目标：**

- MLflow支持100+实验
- 处理高达1GB的数据集
- 模型注册支持50+版本
- Airflow调度10+管道

**验收标准：**

- 跟踪100+实验
- 成功处理1GB数据集
- 注册50+模型版本
- 调度多个管道

---

### NFR-5：可用性

**描述：** 系统必须易于使用

**需求：**

- 清晰的错误消息
- 全面的文档
- 示例笔记本
- 设置自动化
- 直观的UI导航

**验收标准：**

- 错误消息可操作
- 文档完整
- 设置脚本可用
- 提供示例笔记本
- 新用户可以在<1小时内开始使用

---

### NFR-6：可维护性

**描述：** 代码必须可维护

**需求：**

- 模块化代码结构
- 全面的日志记录
- 组件的单元测试
- 管道的集成测试
- 代码文档（文档字符串）

**验收标准：**

- 代码按模块组织
- 所有函数都有文档字符串
- 日志记录级别适当
- 测试覆盖率>70%
- 集成测试通过

---

## 技术规格

### 技术栈版本

```yaml
python: "3.11+"
pytorch: "2.1.0"
mlflow: "2.8.1"
airflow: "2.7.3"
dvc: "3.30.1"
great-expectations: "0.18.3"
postgresql: "15"
redis: "7"
minio: "latest"
```

### 基础设施需求

**最低硬件：**

- CPU：4核
- RAM：16GB
- 存储：50GB
- GPU：可选（加快训练速度）

**推荐硬件：**

- CPU：8核
- RAM：32GB
- 存储：100GB
- GPU：NVIDIA T4或更好

### 网络需求

**要暴露的端口：**

- 5000：MLflow UI
- 8080：Airflow UI
- 9000：MinIO API
- 9001：MinIO控制台
- 5432：PostgreSQL
- 6379：Redis

---

## 验收标准

### 阶段1：基础设施 ✅

- Docker Compose文件已创建
- 所有服务成功启动
- MLflow UI可在[http://localhost:5000访问](http://localhost:5000访问)
- Airflow UI可在[http://localhost:8080访问](http://localhost:8080访问)
- MinIO控制台可在[http://localhost:9001访问](http://localhost:9001访问)
- PostgreSQL可在端口5432访问
- 服务在重启之间持久化数据

### 阶段2：数据管道 ✅

- 数据摄取模块已完成
- 从CSV、API、数据库摄取
- 使用Great Expectations进行数据验证
- 验证套件有5+期望
- 预处理管道功能正常
- 创建了train/val/test分割
- DVC已初始化并配置
- 数据使用DVC版本化
- 3+数据版本已标记

### 阶段3：训练和跟踪 ✅

- 训练模块已完成
- 支持ResNet18和MobileNetV2
- MLflow跟踪已集成
- 参数已记录
- 每个epoch的指标已记录
- 模型工件已保存
- 5+实验已跟踪
- 模型达到>75%验证准确率
- 训练在<30分钟内完成

### 阶段4：模型注册 ✅

- 模型在MLflow中注册
- 3+模型版本存在
- 生命周期阶段正常工作
- Production模型已指定
- 模型元数据完整
- 回滚到先前版本已测试

### 阶段5：编排 ✅

- Airflow DAG已实现
- DAG中有7+任务
- 任务依赖关系正确
- DAG端到端成功运行
- 管道每周调度
- 错误处理和重试正常工作
- 通知已配置
- 3+成功调度运行

### 阶段6：文档 ✅

- README.md完整
- Architecture.md包含图表
- 设置指南已创建
- MLflow使用指南
- DVC工作流已文档化
- 故障排除指南
- 代码注释和文档字符串
- 示例笔记本

---

## 约束和假设

### 约束

1. **仅本地开发** - 无需云部署
2. **小数据集** - 最大1GB用于快速迭代
3. **无需GPU** - CPU训练可接受
4. **单机** - 无需分布式训练
5. **英文文档** - 所有文档为英文

### 假设

1. **Docker可用** - 用户已安装Docker
2. **互联网访问** - 用于下载模型和数据
3. **基本ML知识** - 用户理解ML概念
4. **Git知识** - 用户熟悉Git
5. **Linux/MacOS** - 主要开发平台（Windows需要调整）

---

## 范围外

以下内容**不是**此项目所需：

- ❌ 云部署（AWS、GCP、Azure）
- ❌ 分布式训练（多GPU、多节点）
- ❌ 生产级基础设施
- ❌ 模型服务/部署（在项目1-2中涵盖）
- ❌ A/B测试框架
- ❌ 模型漂移检测
- ❌ 特征存储（Feast）- 可选，非必需
- ❌ AutoML集成
- ❌ 实时推理
- ❌ 模型可解释性（SHAP、LIME）

这些主题在后续项目（高级工程师级别）中涵盖。

---

**需求版本：** 1.0
**批准：** AI基础设施课程团队
**审查日期：** 2025年10月18日