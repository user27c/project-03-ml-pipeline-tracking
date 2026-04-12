# 架构：带实验跟踪的机器学习管道

**版本：** 1.0
**最后更新：** 2026年4月12日

---

## 目录

1. [系统架构](#系统架构)
2. [组件设计](#组件设计)
3. [数据流](#数据流)
4. [基础设施架构](#基础设施架构)
5. [MLflow架构](#mlflow架构)
6. [Airflow架构](#airflow架构)
7. [技术决策](#技术决策)
8. [设计模式](#设计模式)
9. [安全考虑](#安全考虑)
10. [可扩展性考虑](#可扩展性考虑)

---

## 系统架构

### 高层概述

```
┌────────────────────────────────────────────────────────────────┐
│                   ML Pipeline Ecosystem                        │
└────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   Data Sources   │
                    │  - CSV Files     │
                    │  - REST APIs     │
                    │  - Databases     │
                    └────────┬─────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │      Data Ingestion & Validation      │
         │   - DataIngestion Class               │
         │   - Great Expectations Validator      │
         └─────────┬─────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐    ┌──────────────┐
         │   Data Versioning   │◄───│     DVC      │
         │   - Raw Data        │    │  - MinIO     │
         │   - Processed Data  │    │  - Git       │
         └─────────┬───────────┘    └──────────────┘
                   │
                   ▼
         ┌───────────────────────────────────────┐
         │      Data Preprocessing               │
         │   - Cleaning                          │
         │   - Feature Engineering               │
         │   - Train/Val/Test Split              │
         └─────────┬─────────────────────────────┘
                   │
                   ▼
         ┌───────────────────────────────────────┐
         │      Feature Store (PostgreSQL)       │
         │   - Structured Features               │
         │   - Feature Metadata                  │
         └─────────┬─────────────────────────────┘
                   │
                   ▼
         ┌───────────────────────────────────────┐
         │      Model Training                   │
         │   - PyTorch Models                    │
         │   - Hyperparameter Tuning             │
         │   - Cross Validation                  │
         └─────────┬─────────────────────────────┘
                   │
                   ├──────────────────────────┐
                   │                          │
                   ▼                          ▼
         ┌──────────────────┐      ┌──────────────────┐
         │  Experiment      │      │  Model           │
         │  Tracking        │      │  Artifacts       │
         │  - MLflow        │      │  - PyTorch       │
         │  - Parameters    │      │  - ONNX          │
         │  - Metrics       │      │  - Plots         │
         └──────────────────┘      └──────────────────┘
                   │
                   ▼
         ┌───────────────────────────────────────┐
         │      Model Evaluation                 │
         │   - Test Set Metrics                  │
         │   - Confusion Matrix                  │
         │   - Per-Class Performance             │
         └─────────┬─────────────────────────────┘
                   │
                   ▼
         ┌───────────────────────────────────────┐
         │      Model Registry                   │
         │   - Versioned Models                  │
         │   - Lifecycle Stages                  │
         │   - Production Deployment             │
         └───────────────────────────────────────┘

                   ┌──────────────────────┐
                   │  Workflow            │
                   │  Orchestration       │
                   │  - Apache Airflow    │
                   │  - Task Scheduling   │
                   │  - Monitoring        │
                   └──────────────────────┘
```

---

## 组件设计

### 1. 数据摄取组件

**目的：** 将数据从多个源加载到管道中

**类图：**
```python
class DataIngestion:
    """
    处理来自多个源的数据摄取。

    职责：
    - 从CSV、API、数据库加载数据
    - 验证源连接性
    - 保存带元数据的原始数据
    """

    def __init__(self, config: Dict[str, Any])
    def ingest_from_csv(self, file_path: str) -> pd.DataFrame
    def ingest_from_api(self, api_url: str, params: Dict) -> pd.DataFrame
    def ingest_from_database(self, conn_string: str, query: str) -> pd.DataFrame
    def save_raw_data(self, df: pd.DataFrame, filename: str) -> Path
    def get_metadata(self) -> Dict[str, Any]
```

**输入/输出：**
- **输入：** 文件路径、API URL、数据库连接字符串
- **输出：** 保存到 `data/raw/` 的原始数据、DataFrame对象
- **副作用：** 创建目录、写入文件

**错误处理：**
- 网络错误：指数退避重试（最多3次重试）
- 文件未找到：记录错误并抛出FileNotFoundError
- 数据库错误：记录连接详情（已清理）并抛出异常

---

### 2. 数据验证组件

**目的：** 在处理前验证数据质量

**类图：**
```python
class DataValidator:
    """
    使用Great Expectations验证数据。

    职责：
    - 定义期望套件
    - 运行验证检查
    - 生成验证报告
    """

    def __init__(self, context_root_dir: str)
    def create_expectation_suite(self, suite_name: str) -> ExpectationSuite
    def validate_data(self, df: pd.DataFrame, suite_name: str) -> bool
    def get_validation_report(self) -> str
    def add_expectation(self, expectation: Dict) -> None
```

**验证规则：**
```python
expectations = [
    # 模式验证
    {
        "type": "expect_table_column_count_to_equal",
        "value": 10
    },
    # 数据质量
    {
        "type": "expect_column_values_to_not_be_null",
        "column": "image_path"
    },
    # 领域验证
    {
        "type": "expect_column_values_to_be_in_set",
        "column": "label",
        "value_set": ["cat", "dog", "bird", "fish"]
    },
    # 范围验证
    {
        "type": "expect_table_row_count_to_be_between",
        "min_value": 1000,
        "max_value": 1000000
    }
]
```

**管道集成：**
- 验证在摄取后运行
- 如果验证失败，管道停止
- 验证报告保存到 `reports/` 目录

---

### 3. 数据预处理组件

**目的：** 清理和转换数据以供训练

**类图：**
```python
class DataPreprocessor:
    """
    为模型训练预处理数据。

    职责：
    - 清理数据（重复、缺失值）
    - 编码标签
    - 分割数据
    - 保存工件
    """

    def __init__(self, config: Dict[str, Any])
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]
    def save_processed_data(self, train, val, test) -> None
    def run_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]
```

**处理管道：**
```
原始数据
    ↓
移除重复项 (df.drop_duplicates())
    ↓
处理缺失值 (dropna/fillna)
    ↓
编码分类标签 (LabelEncoder)
    ↓
分层分割 (train_test_split, stratify=y)
    ↓
保存分割+工件
```

**生成的工件：**
- `artifacts/label_encoder.pkl` - 用于推理
- `artifacts/scaler.pkl` - 如果应用了归一化
- `artifacts/preprocessing_config.json` - 可复现性

---

### 4. 模型训练组件

**目的：** 带实验跟踪的模型训练

**类图：**
```python
class ModelTrainer:
    """
    使用MLflow跟踪训练模型。

    职责：
    - 创建模型架构
    - 训练并验证
    - 记录到MLflow
    - 保存最佳模型
    """

    def __init__(self, config: Dict, mlflow_tracker: MLflowTracker)
    def create_model(self, num_classes: int, model_name: str) -> nn.Module
    def train_epoch(self, model, train_loader, criterion, optimizer) -> Tuple[float, float]
    def validate(self, model, val_loader, criterion) -> Tuple[float, float]
    def train(self, train_loader, val_loader, num_classes: int, params: Dict) -> Tuple[nn.Module, float]
```

**训练循环：**
```python
for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

    # 验证
    val_loss, val_acc = validate(model, val_loader, criterion)

    # 记录到MLflow
    mlflow.log_metrics({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }, step=epoch)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    # 提前停止
    if no_improvement_for(patience):
        break
```

**支持的架构：**
1. **ResNet18** - 良好的基线，11M参数
2. **MobileNetV2** - 轻量级，3.5M参数

---

### 5. MLflow跟踪组件

**目的：** 集中管理MLflow操作

**类图：**
```python
class MLflowTracker:
    """
    MLflow跟踪操作的包装器。

    职责：
    - 开始/结束运行
    - 记录参数、指标、工件
    - 注册模型
    - 管理模型阶段
    """

    def __init__(self, tracking_uri: str, experiment_name: str)
    def start_run(self, run_name: str, tags: Dict) -> Run
    def log_params(self, params: Dict) -> None
    def log_metrics(self, metrics: Dict, step: int) -> None
    def log_artifact(self, artifact_path: str) -> None
    def log_model(self, model, artifact_path: str) -> None
    def register_model(self, model_uri: str, model_name: str) -> ModelVersion
    def transition_model_stage(self, model_name: str, version: int, stage: str) -> None
    def end_run(self) -> None
```

**使用模式：**
```python
# 初始化
tracker = MLflowTracker(
    tracking_uri="http://mlflow:5000",
    experiment_name="image_classification"
)

# 开始运行
tracker.start_run(run_name="resnet18_exp1", tags={"model": "resnet18"})

# 训练期间记录
tracker.log_params({"lr": 0.001, "batch_size": 32})
tracker.log_metrics({"val_acc": 85.2}, step=10)

# 记录模型
tracker.log_model(model, "model")

# 注册模型
model_version = tracker.register_model(
    model_uri=f"runs:/{run_id}/model",
    model_name="image_classifier"
)

# 提升到生产环境
tracker.transition_model_stage(
    model_name="image_classifier",
    version=model_version.version,
    stage="Production"
)

# 结束运行
tracker.end_run()
```

---

### 6. 模型评估组件

**目的：** 综合模型评估

**类图：**
```python
class ModelEvaluator:
    """
    在测试集上评估模型。

    职责：
    - 计算指标（准确率、精确率、召回率、F1）
    - 生成混淆矩阵
    - 创建可视化
    - 记录结果到MLflow
    """

    def __init__(self, config: Dict, mlflow_tracker: MLflowTracker)
    def evaluate(self, model, test_loader) -> Dict[str, float]
    def compute_metrics(self, y_true, y_pred) -> Dict
    def generate_confusion_matrix(self, y_true, y_pred) -> np.ndarray
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> plt.Figure
    def generate_classification_report(self, y_true, y_pred) -> str
```

**计算的指标：**
```python
metrics = {
    "test_accuracy": accuracy_score(y_true, y_pred),
    "test_precision": precision_score(y_true, y_pred, average='macro'),
    "test_recall": recall_score(y_true, y_pred, average='macro'),
    "test_f1": f1_score(y_true, y_pred, average='macro'),
    "per_class_precision": precision_score(y_true, y_pred, average=None),
    "per_class_recall": recall_score(y_true, y_pred, average=None),
    "per_class_f1": f1_score(y_true, y_pred, average=None)
}
```

---

## 数据流

### 端到端管道流程

```
┌─────────────────────┐
│  1. 数据摄取        │
│  - 从源加载         │
│  - 保存到raw/       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. 数据验证        │
│  - GE期望           │
│  - 通过/失败检查    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. 预处理          │
│  - 清理数据         │
│  - 编码标签         │
│  - 分割数据         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. DVC版本控制     │
│  - dvc add          │
│  - git commit       │
│  - dvc push         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. 模型训练        │
│  - 加载数据         │
│  - 训练模型         │
│  - 在MLflow中跟踪   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. 模型评估        │
│  - 测试指标         │
│  - 混淆矩阵         │
│  - 记录到MLflow     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  7. 模型注册        │
│  - 注册模型         │
│  - 阶段转换         │
│  - 元数据更新       │
└─────────────────────┘
```

### 数据持久化流程

```
┌───────────────┐
│   原始数据     │
│   (CSV/API)    │
└───────┬───────┘
        │
        ▼
┌───────────────────┐      ┌─────────────┐
│  data/raw/        │─────▶│  DVC跟踪    │
│  dataset.csv      │      │  .dvc文件   │
└───────┬───────────┘      └─────────────┘
        │                          │
        │                          ▼
        │                  ┌───────────────┐
        │                  │  MinIO/S3     │
        │                  │  (远程)       │
        │                  └───────────────┘
        ▼
┌───────────────────┐
│  data/processed/  │
│  - train.csv      │
│  - val.csv        │
│  - test.csv       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  PostgreSQL       │
│  特征存储         │
└───────────────────┘
```

### MLflow工件流程

```
┌──────────────┐
│  训练        │
│  执行        │
└──────┬───────┘
       │
       ├────────────────────────────┐
       │                            │
       ▼                            ▼
┌──────────────┐            ┌──────────────┐
│  参数        │            │   指标       │
│  - lr: 0.001 │            │  - accuracy  │
│  - batch: 32 │            │  - loss      │
└──────┬───────┘            └──────┬───────┘
       │                            │
       ▼                            ▼
┌──────────────────────────────────────┐
│      PostgreSQL (MLflow后端)         │
│      - 运行元数据                    │
│      - 参数                          │
│      - 指标                          │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  工件        │
│  - model.pth │
│  - plots/    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│      MinIO/S3 (工件存储)              │
│      - 模型文件                      │
│      - 图表                          │
│      - 日志                          │
└──────────────────────────────────────┘
```

---

## 基础设施架构

### Docker Compose架构

```yaml
┌─────────────────────────────────────────────────────┐
│              Docker Compose Network                 │
└─────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  PostgreSQL  │  │    Redis     │  │    MinIO     │
│   Port 5432  │  │   Port 6379  │  │  Ports 9000  │
│              │  │              │  │       9001   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │                 │                 │
       ▼                 │                 │
┌──────────────┐         │                 │
│   MLflow     │         │                 │
│  Tracking    │◄────────┘                 │
│   Port 5000  │◄──────────────────────────┘
└──────────────┘
       ▲
       │
┌──────┴───────────────────────────────────┐
│         Airflow Components               │
├──────────────────────────────────────────┤
│  ┌────────────┐  ┌─────────┐  ┌────────┐│
│  │ Webserver  │  │Scheduler│  │ Worker ││
│  │ Port 8080  │  │         │  │        ││
│  └────────────┘  └─────────┘  └────────┘│
└──────────────────────────────────────────┘
```

### 容器规格

**PostgreSQL容器：**
```yaml
postgres:
  image: postgres:15
  environment:
    POSTGRES_USER: mlflow
    POSTGRES_PASSWORD: mlflow
    POSTGRES_DB: mlflow
  volumes:
    - postgres_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U mlflow"]
    interval: 10s
    timeout: 5s
    retries: 5
```

**MLflow容器：**
```yaml
mlflow:
  build: ./mlflow
  ports:
    - "5000:5000"
  environment:
    MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    AWS_ACCESS_KEY_ID: minioadmin
    AWS_SECRET_ACCESS_KEY: minioadmin
  depends_on:
    postgres:
      condition: service_healthy
    minio:
      condition: service_started
  command: >
    mlflow server
    --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
    --default-artifact-root s3://mlflow/
    --host 0.0.0.0
    --port 5000
```

**MinIO容器：**
```yaml
minio:
  image: minio/minio:latest
  command: server /data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin
  ports:
    - "9000:9000"   # API
    - "9001:9001"   # 控制台
  volumes:
    - minio_data:/data
```

**Airflow Webserver：**
```yaml
airflow-webserver:
  build: ./airflow
  ports:
    - "8080:8080"
  environment:
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://mlflow:mlflow@postgres:5432/airflow
    AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://mlflow:mlflow@postgres:5432/airflow
  depends_on:
    - postgres
    - redis
  command: webserver
```

---

## MLflow架构

### MLflow组件栈

```
┌──────────────────────────────────────────────┐
│          MLflow Tracking Server              │
│              (Port 5000)                     │
└────────────┬─────────────────────────────────┘
             │
             ├─────────────────────┬───────────────────┐
             │                     │                   │
             ▼                     ▼                   ▼
    ┌─────────────────┐   ┌─────────────────┐  ┌──────────────┐
    │  Experiments    │   │  Model Registry │  │  Artifacts   │
    │  - Runs         │   │  - Versions     │  │  - Models    │
    │  - Parameters   │   │  - Stages       │  │  - Plots     │
    │  - Metrics      │   │  - Metadata     │  │  - Logs      │
    └────────┬────────┘   └────────┬────────┘  └──────┬───────┘
             │                     │                   │
             ▼                     ▼                   ▼
    ┌──────────────────────────────────────────────────────┐
    │             PostgreSQL Backend Store                 │
    │  - Experiment metadata                               │
    │  - Run parameters and metrics                        │
    │  - Model registry data                               │
    └──────────────────────────────────────────────────────┘
                               │
                               │
    ┌──────────────────────────────────────────────────────┐
    │             MinIO/S3 Artifact Store                  │
    │  - Model artifacts (model.pth)                       │
    │  - Plots and visualizations                          │
    │  - Logs and other files                              │
    └──────────────────────────────────────────────────────┘
```

### MLflow数据模型

**Experiments表：**
```sql
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    name VARCHAR(256) UNIQUE NOT NULL,
    artifact_location VARCHAR(256),
    lifecycle_stage VARCHAR(32),
    creation_time BIGINT,
    last_update_time BIGINT
);
```

**Runs表：**
```sql
CREATE TABLE runs (
    run_uuid VARCHAR(32) PRIMARY KEY,
    name VARCHAR(256),
    source_type VARCHAR(20),
    source_name VARCHAR(500),
    entry_point_name VARCHAR(50),
    user_id VARCHAR(256),
    status VARCHAR(20),
    start_time BIGINT,
    end_time BIGINT,
    source_version VARCHAR(50),
    lifecycle_stage VARCHAR(20),
    artifact_uri VARCHAR(200),
    experiment_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
```

**模型注册表：**
```sql
CREATE TABLE registered_models (
    name VARCHAR(256) PRIMARY KEY,
    creation_time BIGINT,
    last_updated_time BIGINT,
    description VARCHAR(5000)
);

CREATE TABLE model_versions (
    name VARCHAR(256),
    version INTEGER,
    creation_time BIGINT,
    last_updated_time BIGINT,
    description VARCHAR(5000),
    user_id VARCHAR(256),
    current_stage VARCHAR(20),
    source VARCHAR(500),
    run_id VARCHAR(32),
    status VARCHAR(20),
    status_message VARCHAR(500),
    PRIMARY KEY (name, version),
    FOREIGN KEY (name) REFERENCES registered_models(name)
);
```

---

## Airflow架构

### Airflow组件

```
┌────────────────────────────────────────────────┐
│            Airflow Architecture                │
└────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Web Server  │      │  Scheduler   │      │    Worker    │
│  Port 8080   │      │  (Core)      │      │  (Executor)  │
│              │      │              │      │              │
│ - UI         │      │ - Parse DAGs │      │ - Run tasks  │
│ - REST API   │      │ - Schedule   │      │ - Report     │
│ - Auth       │      │ - Trigger    │      │   status     │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   PostgreSQL   │
                    │   Metadata DB  │
                    │                │
                    │ - DAG metadata │
                    │ - Task state   │
                    │ - Run history  │
                    └────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │     Redis      │
                    │   (Broker)     │
                    │                │
                    │ - Task queue   │
                    │ - Messages     │
                    └────────────────┘
```

### DAG执行流程

```
用户触发DAG
       │
       ▼
调度器拾取DAG
       │
       ▼
解析DAG Python文件
       │
       ▼
创建DAG Run实例
       │
       ▼
按依赖顺序处理每个任务：
    │
    ├─▶ 创建Task Instance
    │       │
    │       ▼
    │   将任务加入Redis队列
    │       │
    │       ▼
    │   Worker拾取任务
    │       │
    │       ▼
    │   执行任务Python代码
    │       │
    │       ▼
    │   向PostgreSQL报告状态
    │       │
    │       ▼
    │   通过XCom传递数据（如需要）
    │
    └─▶ 重复下一任务
```

### DAG结构

```python
# ML Training Pipeline DAG
DAG(
    dag_id='ml_training_pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        'owner': 'ml-team',
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(hours=2)
    }
)

# 任务依赖图
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

---

## 技术决策

### 为什么选择Apache Airflow？

**优点：**
✅ 行业标准（被Airbnb、Netflix、Adobe使用）
✅ 丰富的UI用于监控和调试
✅ 广泛的插件生态系统
✅ Python原生（ML团队易于使用）
✅ 支持复杂依赖
✅ 内置重试和错误处理
✅ XCom用于任务间通信

**缺点：**
❌ 初学者设置复杂
❌ 资源密集
❌ 对于简单管道来说过于复杂

**考虑的替代方案：**
- **Prefect 2.0**：更现代，设置更简单，但成熟度较低
- **Kubeflow Pipelines**：Kubernetes原生，但复杂
- **Luigi**：更简单，但功能较少

**决策：** 选择Airflow是因为其行业相关性和学习价值。

---

### 为什么选择MLflow？

**优点：**
✅ 开源，厂商中立
✅ 最受欢迎的MLOps平台（15K+星标）
✅ 完整的ML生命周期管理
✅ 易于与PyTorch、TensorFlow、sklearn集成
✅ 内置模型注册表
✅ REST API用于自动化
✅ 活跃的社区和开发

**缺点：**
❌ UI可以更现代化
❌ 内置模型服务有限
✅ 在非常大规模时可扩展性挑战

**考虑的替代方案：**
- **Weights & Biases**：更好的UI，但商业/SaaS
- **Neptune.ai**：更多功能，但商业
- **TensorBoard**：仅限TensorFlow生态系统
- **Sacred**：维护不活跃

**决策：** 选择MLflow是因为其开源性、完整性和行业采用。

---

### 为什么选择DVC？

**优点：**
✅ Git风格的数据工作流
✅ 存储无关（S3、GCS、Azure、SSH）
✅ 与Git无缝集成
✅ 轻量级且快速
✅ 支持管道和指标跟踪
✅ 开源且免费

**缺点：**
❌ 需要Git知识
❌ 对于非常大的文件可能较慢
❌ 团队采用的学习曲线

**考虑的替代方案：**
- **Git LFS**：功能有限，无版本控制逻辑
- **Pachyderm**：更复杂，依赖Kubernetes
- **LakeFS**：数据湖的Git，对此项目来说过于复杂

**决策：** 选择DVC是因为其简单性和Git集成。

---

### 为什么选择PostgreSQL？

**优点：**
✅ 经证实的可靠性
✅ ACID合规
✅ 丰富的索引和查询优化
✅ JSON支持元数据
✅ 免费和开源
✅ 卓越的Python支持（psycopg2）

**缺点：**
❌ 对于超大数据集，可扩展性不如NoSQL
❌ 垂直扩展限制

**考虑的替代方案：**
- **MySQL**：类似，但PostgreSQL的JSON支持更好
- **MongoDB**：NoSQL，但元数据需要ACID合规
- **SQLite**：对于多服务设置太有限

**决策：** 选择PostgreSQL是因为其可靠性和功能丰富性。

---

## 设计模式

### 1. 仓库模式
在单独的类中封装数据访问逻辑。

```python
class ExperimentRepository:
    """抽象MLflow实验操作"""

    def get_all_runs(self, experiment_id: str) -> List[Run]:
        return mlflow.search_runs(experiment_ids=[experiment_id])

    def get_best_run(self, experiment_id: str, metric: str) -> Run:
        runs = self.get_all_runs(experiment_id)
        return runs.sort_values(f"metrics.{metric}", ascending=False).iloc[0]
```

### 2. 策略模式
允许不同的数据摄取策略。

```python
class DataIngestionStrategy(ABC):
    @abstractmethod
    def ingest(self) -> pd.DataFrame:
        pass

class CSVIngestionStrategy(DataIngestionStrategy):
    def ingest(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

class APIIngestionStrategy(DataIngestionStrategy):
    def ingest(self) -> pd.DataFrame:
        response = requests.get(self.url)
        return pd.DataFrame(response.json())
```

### 3. 工厂模式
基于配置创建模型。

```python
class ModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int) -> nn.Module:
        if model_name == "resnet18":
            return create_resnet18(num_classes)
        elif model_name == "mobilenet_v2":
            return create_mobilenet_v2(num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
```

### 4. 观察者模式
MLflow记录作为训练事件的观察者。

```python
class TrainingObserver(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        pass

class MLflowObserver(TrainingObserver):
    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        mlflow.log_metrics(metrics, step=epoch)
```

---

## 安全考虑

### 1. 凭证管理
- 将凭证存储在环境变量中
- 永远不要将秘密提交到Git
- 使用 `.env` 文件（gitignored）
- 考虑生产环境使用HashiCorp Vault

### 2. 网络安全
- 服务通信使用内部Docker网络
- 仅向主机暴露必要的端口
- 使用PostgreSQL身份验证
- 需要MinIO访问密钥

### 3. 数据隐私
- 本项目不包含PII（个人身份信息）
- 对于真实项目：加密静态和传输中的数据
- 实施访问控制

### 4. MLflow安全
- MLflow UI默认无身份验证（本地开发OK）
- 生产环境：启用身份验证插件
- 限制工件存储访问

---

## 可扩展性考虑

### 当前限制
- 单机部署
- 无水平扩展
- 仅限本地数据集（<1GB）
- 单个Airflow worker

### 扩展策略（未来）

**水平扩展：**
- 多个Airflow workers（Celery Executor）
- 多个MLflow跟踪服务器（负载均衡）
- PostgreSQL读副本

**数据扩展：**
- 分区大数据集
- 使用数据采样进行开发
- 实现增量加载
- 对于非常大的数据考虑Spark

**计算扩展：**
- GPU加速训练
- 分布式训练（PyTorch DDP）
- 为密集工作负载进行云突发

**存储扩展：**
- 对象存储（S3、GCS）用于工件
- 分布式文件系统（HDFS）
- 数据湖架构

---

**架构版本：** 1.0
**审核人：** AI Infrastructure Curriculum Team
**最后更新：** 2025年10月18日
