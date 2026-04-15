# 带实验跟踪的 ML 管道（MLOps 练习项目）

面向**项目维护者与实现者**：说明本仓库实现什么、代码如何组织、如何在本机或 Docker 中跑通，以及常见配置项。课程级背景与评分细则见 [`requirements.md`](requirements.md)；设计细节见 [`architecture.md`](architecture.md)。

---

## 1. 项目定位

本仓库实现一条 **从数据摄取 → 校验 → 预处理 →（可选 DVC）→ 特征入库 → 训练 → 评估 → 模型注册** 的流水线，并用 **Apache Airflow** 编排、**MLflow** 做实验与模型跟踪、**PostgreSQL** 存特征与元数据、**DVC + MinIO** 支撑数据与工件版本化（按你的环境启用程度而定）。

解决的问题可以概括成：实验可复现、数据与模型可追溯、重复训练可自动化。

---

## 2. 演示截图（预留）

请将演示图片放在仓库根目录下的 **`result_img/`** 中，并在下方替换文件名或取消注释。建议命名便于区分场景。

| 场景 | 建议文件名 | 说明 |
|------|------------|------|
| 端到端架构或数据流 | `result_img/demo_architecture.png` | 可手绘或从 `architecture.md` 导出 |
| Airflow DAG / 运行实例 | `result_img/demo_airflow_dag.png` | Grid / Graph / 一次成功 Run |
| MLflow 实验与指标 | `result_img/demo_mlflow_ui.png` | Runs / Metrics / Artifacts |
| MLflow 模型注册（可选） | `result_img/demo_model_registry.png` | Registered Models |
| MinIO 桶与工件（可选） | `result_img/demo_minio.png` | `mlflow` 桶或 DVC 远程 |
| 评估曲线或混淆矩阵（可选） | `result_img/demo_evaluation.png` | 来自 `evaluation` 产出 |

<!-- **占位图（放入同名文件后即显示）：** -->

<p align="center">

<!-- 将 demo_architecture.png 放入 result_img/ -->
<img src="result_img/image copy 9.png" alt="架构 / 数据流演示（请将图片放到 result_img/demo_architecture.png）" width="720"/>

</p>

<p align="center">

<!-- 将 demo_airflow_dag.png 放入 result_img/ -->
<img src="result_img/image copy.png" alt="Airflow DAG 演示（请将图片放到 result_img/demo_airflow_dag.png）" width="720"/>

</p>

<p align="center">

<!-- 将 demo_mlflow_ui.png 放入 result_img/ -->
<img src="result_img/image copy 7.png" alt="MLflow 演示（请将图片放到 result_img/demo_mlflow_ui.png）" width="720"/>

</p>

<!-- > **说明：** 在图片尚未加入前，部分 Markdown 预览器会显示裂图，属正常现象；补齐 `result_img/` 内文件即可。 -->

---

## 3. 技术栈（实现侧）

| 领域 | 选型 | 在本项目中的角色 |
|------|------|------------------|
| 编排 | Apache Airflow 2.7+ | `dags/ml_pipeline_dag.py` 定义任务依赖与调度 |
| 实验与模型 | MLflow | `src/training.py` 等写入 Run；注册在 DAG `register_model` |
| 训练 / 评估 | PyTorch | `src/training.py`、`src/evaluation.py`、`src/dataset.py` |
| 数据校验 | 自研 `DataValidator`（类 Great Expectations 规则） | `src/data_validation.py` |
| 特征表 | PostgreSQL + `psycopg2` | `src/feature_store.py` |
| 数据版本 | DVC（可选） | 任务 `version_data_dvc`；需本机/容器内 Git + DVC |
| 对象存储 | MinIO（Compose 中） | MLflow artifact、可与 DVC 远程对接 |
| 消息队列 | Redis | Airflow Celery Executor（`docker-compose.yml`） |

---

## 4. 架构概览（实现视角）

```
数据源 (CSV，如 CIFAR-10 列表)
    → 摄取 / 校验 / 预处理 (src/*)
    → 合并划分 all_splits.csv → 训练与评估 (dataset + training + evaluation)
    → MLflow 记录；达标则模型注册
    → 特征写入 PostgreSQL（FeatureStore）
```

Airflow 仅负责**调度与传参**（含 XCom）；业务逻辑集中在 `src/`。DAG 内通过 **`PROJECT_ROOT` / 环境变量** 区分「本机 conda」与「Docker `/opt/airflow`」，避免写死个人路径。

---

## 5. 仓库结构（与代码对应）

```
project-03-ml-pipeline-tracking/
├── README.md                 # 本文件（开发者说明）
├── requirements.md           # 课程 / 需求细则
├── architecture.md           # 架构与设计决策（偏长文）
├── docker-compose.yml        # Postgres、MinIO、MLflow、Redis、Airflow 等
├── dags/
│   └── ml_pipeline_dag.py    # 主 DAG：任务与 XCom、环境变量约定
├── src/
│   ├── data_ingestion.py     # 摄取与落盘
│   ├── data_validation.py    # 数据质量校验
│   ├── preprocessing.py      # 清洗、编码、划分、all_splits.csv
│   ├── feature_store.py      # PostgreSQL 特征表
│   ├── dataset.py            # PyTorch Dataset / DataLoader
│   ├── training.py           # 训练 + MLflow
│   └── evaluation.py         # 评估指标与图表
├── airflow/
│   ├── Dockerfile            # Airflow 镜像：需包含 mlflow、torch 等
│   └── requirements.txt      # 与镜像安装对齐的依赖列表（参考）
├── mlflow/
│   ├── Dockerfile
│   └── MLproject
├── scripts/                  # SQL 初始化、数据准备等
├── tests/
│   └── test_pipeline.py
├── data/                     # raw / processed / validation（运行时产出，勿乱提交大文件）
└── result_img/               # 演示截图（你可自行添加，见上文「演示截图」）
```

---

## 6. 主 DAG 任务链（开发者速查）

任务顺序：**ingest → validate → preprocess → version_data_dvc → store_features → train → evaluate → register → notify**。

维护 DAG 时重点关注：

- **路径**：统一由 `PROJECT_ROOT` 与 `PIPELINE_CONFIG` 推导；Docker 下常设 `ML_PIPELINE_PROJECT_ROOT=/opt/airflow`。
- **XCom**：下游依赖 `preprocess_data` 写入的 `all_splits_csv`（或回退到磁盘上的 `data/processed/all_splits.csv`）。
- **PostgreSQL**：容器内不要用 `localhost` 指代「宿主机上的库」；同 Compose 网络内应使用服务名 **`postgres`**（DAG 中已对 Docker 做了默认检测，仍可通过环境变量覆盖）。
- **MLflow URI**：容器内默认 `http://mlflow:5000`；本机 conda 常用 `http://127.0.0.1:5000`。

常用环境变量（可选，按部署填写）：

| 变量 | 含义 |
|------|------|
| `ML_PIPELINE_PROJECT_ROOT` | 项目根绝对路径（Docker 常为 `/opt/airflow`） |
| `MLFLOW_TRACKING_URI` | MLflow 跟踪地址 |
| `MLFLOW_ARTIFACT_BUCKET` | MLflow 工件桶名（默认 `mlflow`）。DAG 在训练前会尝试自动检查/创建该桶。 |
| `FEATURE_STORE_POSTGRES_HOST` / `FEATURE_STORE_POSTGRES_*` | 特征库连接 |
| `ML_PIPELINE_ALERT_EMAIL` | 若设置，则启用失败邮件相关默认行为（视 Airflow SMTP 配置而定） |
| `MAIL_SMTP_HOST` / `MAIL_SMTP_PORT` / `MAIL_USERNAME` / `MAIL_PASSWORD` | 成功通知邮件 SMTP 配置（DAG `send_success_email` 任务读取；建议通过环境变量注入，不写入仓库） |
| `ML_PIPELINE_REUSE_TRAIN_CACHE` | 设为 `1` / `true` 时，若 `models/best_model.pth` 与 `models/train_cache_meta.json` 存在且与当前 `all_splits.csv` 的 mtime、关键超参数一致，则 **跳过训练**（调下游步骤时省时间）。见 DAG 内说明；**跳过时不会新建 MLflow 训练 Run**，`register_model` 仍关联「最近一次 Run」，需注意一致性。 |
| `ML_PIPELINE_SEED` | 全局随机种子（默认 42），用于训练/采样可复现 |
| `ML_PIPELINE_DETERMINISTIC` | 是否启用尽力 deterministic（默认 true） |
| `ML_PIPELINE_PROMOTE_TO_PROD` | 设为 `1/true` 时注册后自动晋升到 Production，并归档旧 Production（保证唯一） |

---

## 7. 运行方式

### 7.1 Docker Compose（推荐对齐课堂/集成环境）

1. 启动栈：`docker compose up -d`（或 `docker-compose`，视版本而定）。
2. 按 `docker-compose.yml` 文末注释完成 **Airflow DB 初始化、管理员用户、MinIO `mlflow` 桶** 等一次性步骤。
3. 重新构建 Airflow 镜像以纳入代码与依赖变更：`docker compose build --no-cache` 后再 `up`。
4. 浏览器：**Airflow** 常为 `http://localhost:8080`，**MLflow** `http://localhost:5000`。

---

## 7.3 超参数调优（FR-2.2）

本项目提供网格搜索版调参入口（>=10 组配置），所有 trial 都会写入 MLflow。

- **方式 A（推荐）**：在能访问 MLflow 的 Python 环境运行：

```bash
MLFLOW_TRACKING_URI="http://localhost:5000" \
python src/hyperparameter_tuning.py \
  --csv-path "data/processed/all_splits.csv" \
  --root-dir "." \
  --n-trials 10 \
  --num-epochs 2
```

- **方式 B（MLproject）**：`mlflow run . -e tune -P n_trials=10`（依赖 `mlflow/MLproject`）。

调参完成后：
- 在 MLflow 对应实验中会看到一个 parent run（`tune_grid_*`）与多个 nested trial runs
- parent run 的 artifacts 下会有 `best_config.json`

---

## 7.4 模型注册与回滚（FR-3）

- **注册**：DAG 的 `register_model` 会把满足阈值的模型注册为 `image_classifier`，默认进入 `Staging`。
- **自动晋升 Production（可选）**：设置 `ML_PIPELINE_PROMOTE_TO_PROD=true`，会晋升到 `Production` 并自动归档旧 Production（保证唯一）。
- **回滚**：将指定版本切换到 Production：

```bash
MLFLOW_TRACKING_URI="http://localhost:5000" \
python scripts/rollback_model.py --version 3
```

### 7.2 本机 Conda / venv + Airflow

- 在**运行 scheduler / worker 的同一环境**中安装与 `airflow/requirements.txt` 相当的依赖（至少包含 **mlflow、torch、psycopg2-binary** 及项目 `src` 所需包）。
- 将本仓库根目录加入 `PYTHONPATH`，或将 `dags/`、`src/` 放到 Airflow 配置的 `dags_folder` 与可导入路径下。
- 未设置 `ML_PIPELINE_PROJECT_ROOT` 时，DAG 以 **`dags/ml_pipeline_dag.py` 所在仓库根** 推导项目根，请保持目录结构不变。

---

## 8. 文档与扩展阅读

- [`docs/technology/`](docs/technology/)：**技术栈说明、在本项目中的用法、Mermaid 架构图与踩坑要点**（[`项目技术详解.md`](docs/technology/项目技术详解.md)）。
- [`requirements.md`](requirements.md)：功能清单与学习目标（课程向）。
- [`architecture.md`](architecture.md)：组件、数据流、技术决策。
- 上游官方文档：[MLflow](https://mlflow.org/docs/latest/index.html)、[Airflow](https://airflow.apache.org/docs/)、[DVC](https://dvc.org/doc)。

---

## 9. 常见问题（维护者向）

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `ModuleNotFoundError: mlflow` | Airflow 运行环境未安装 mlflow | 在 **Airflow 同一环境** `pip install mlflow`，或重建 **airflow/Dockerfile** 镜像 |
| Postgres `Connection refused` + `localhost` | 任务跑在容器内，却连本机 `localhost` | 使用服务名 `postgres` 或设置 `FEATURE_STORE_POSTGRES_HOST` |
| XCom 缺少 `all_splits_csv` | 仅重试了下游，上游仍是旧 XCom | Clear 上游 `preprocess_data` 或全量重跑；DAG 已支持回退到 `all_splits.csv` 文件 |
| DVC 任务失败 | 容器内无 `.git`/`.dvc` 或未配置远程 | 仅开发环境可跳过该任务或挂载仓库根 |

---

## 10. 版本信息

- **文档面向：** 项目开发者 / 维护者  
- **详细需求与评分：** 见 `requirements.md`  
- **演示素材目录：** `result_img/`（当前可用 `.gitkeep` 占位目录；图片请自行添加）
