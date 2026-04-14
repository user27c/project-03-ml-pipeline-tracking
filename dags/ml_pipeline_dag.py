"""
用于ML训练管道的Airflow DAG

此DAG编排从数据摄取到模型注册的完整ML管道。

路径与 MLflow 地址通过环境变量配置，便于 conda 本地与 Docker 一致运行：
- ML_PIPELINE_PROJECT_ROOT：项目根（可选；默认取本文件上级目录）
- MLFLOW_TRACKING_URI：宿主机默认 http://127.0.0.1:5000；Docker 内未设置时默认 http://mlflow:5000
- FEATURE_STORE_POSTGRES_*：特征库连接。未设置主机时：在 Docker 容器内（存在 /.dockerenv）默认用
  服务名 postgres；在宿主机上默认 127.0.0.1（本机 conda + 本机 Postgres）
- ML_PIPELINE_REUSE_TRAIN_CACHE：设为 1/true 时，若存在有效训练缓存（见 train_model 内说明）则跳过长时间训练。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


def _project_root() -> Path:
    env = os.environ.get("ML_PIPELINE_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _project_root()
_root_str = str(PROJECT_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

logger = logging.getLogger(__name__)


def _default_mlflow_tracking_uri() -> str:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    if Path("/.dockerenv").is_file():
        return "http://mlflow:5000"
    return "http://127.0.0.1:5000"


def _build_pipeline_config() -> dict:
    root = PROJECT_ROOT
    return {
        "raw_data_path": str(root / "data" / "raw"),
        "processed_data_path": str(root / "data" / "processed"),
        "model_save_path": str(root / "models"),
        "artifacts_path": str(root / "artifacts"),
        "mlflow_tracking_uri": _default_mlflow_tracking_uri(),
        "experiment_name": os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "image_classification_pipeline"
        ),
    }


PIPELINE_CONFIG = _build_pipeline_config()


def _feature_store_postgres_host() -> str:
    """
    Postgres 主机：显式环境变量优先；否则在 Docker 内默认连 compose 服务名 postgres，
    避免任务进程里 localhost 指向容器自身导致 Connection refused。
    """
    explicit = os.environ.get("FEATURE_STORE_POSTGRES_HOST") or os.environ.get(
        "POSTGRES_HOST"
    )
    if explicit:
        return explicit
    if Path("/.dockerenv").is_file():
        logger.info(
            "FEATURE_STORE_POSTGRES_HOST 未设置，检测到 Docker 环境，使用主机名 postgres"
        )
        return "postgres"
    return "127.0.0.1"


def _get_all_splits_csv_path(context: dict) -> str:
    """
    从 preprocess_data 的 XCom 读取 all_splits_csv；若缺失则回退到磁盘路径。

    缺失常见原因：DAG 代码升级后只重试了下游任务，上游已成功实例仍只有旧版 XCom
    （例如仅有 preprocessing_complete）。回退路径与 preprocess 写入位置一致。
    """
    ti = context["task_instance"]
    path = ti.xcom_pull(task_ids="preprocess_data", key="all_splits_csv")
    if path:
        return path
    fallback = Path(PIPELINE_CONFIG["processed_data_path"]) / "all_splits.csv"
    if fallback.is_file():
        resolved = str(fallback.resolve())
        logger.warning(
            "XCom 中无 all_splits_csv（常见于仅重试下游或旧运行）；使用文件: %s",
            resolved,
        )
        return resolved
    raise ValueError(
        "无法得到 all_splits.csv：请重新运行 preprocess_data，或确认文件存在: "
        f"{fallback}"
    )


def _xcom_safe_metrics(metrics: dict) -> dict:
    """将 numpy 等类型转为 JSON/XCom 可序列化的 Python 内置类型。"""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

    out: dict = {}
    for k, v in metrics.items():
        if hasattr(v, "item") and callable(getattr(v, "item")):
            try:
                out[k] = float(v.item())
                continue
            except (ValueError, TypeError):
                pass
        if isinstance(v, (float, int)) and not isinstance(v, bool):
            out[k] = float(v)
        elif np is not None and isinstance(v, np.ndarray):
            out[k] = v.astype(float).tolist()
        elif isinstance(v, list):
            conv = []
            for x in v:
                if hasattr(x, "item") and callable(getattr(x, "item")):
                    conv.append(float(x.item()))
                else:
                    conv.append(float(x))
            out[k] = conv
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


_TRAIN_CACHE_PARAM_KEYS = (
    "model_name",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "lr_step_size",
    "lr_gamma",
    "early_stopping_patience",
)


def _train_cache_meta_path() -> Path:
    return Path(PIPELINE_CONFIG["model_save_path"]) / "train_cache_meta.json"


def _reuse_train_cache_enabled() -> bool:
    return os.environ.get("ML_PIPELINE_REUSE_TRAIN_CACHE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _try_reuse_train_cache(all_splits_path: str, params: dict) -> tuple[bool, float]:
    """
    若启用缓存且磁盘上存在与当前数据mtime、关键超参数一致的 checkpoint，则跳过训练。

    注意：跳过时不会新建 MLflow Training Run；evaluate仍使用 best_model.pth。
    register_model 仍取「最近一次」Run注册，若你依赖注册与本次权重严格一致，请勿开缓存或改注册逻辑。
    """
    if not _reuse_train_cache_enabled():
        return False, 0.0
    model_path = Path(PIPELINE_CONFIG["model_save_path"]) / "best_model.pth"
    meta_path = _train_cache_meta_path()
    if not model_path.is_file() or not meta_path.is_file():
        logger.info("训练缓存未命中：缺少 best_model.pth 或 train_cache_meta.json")
        return False, 0.0
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("训练缓存元数据读取失败，将完整训练: %s", e)
        return False, 0.0
    resolved_splits = str(Path(all_splits_path).resolve())
    if meta.get("all_splits_path") != resolved_splits:
        logger.info("训练缓存未命中：all_splits 路径与缓存不一致")
        return False, 0.0
    try:
        current_mtime_ns = Path(all_splits_path).stat().st_mtime_ns
    except OSError:
        return False, 0.0
    if meta.get("all_splits_mtime_ns") != current_mtime_ns:
        logger.info("训练缓存未命中：all_splits.csv 已更新（mtime 变化）")
        return False, 0.0
    cached_params = meta.get("params") or {}
    for key in _TRAIN_CACHE_PARAM_KEYS:
        if cached_params.get(key) != params.get(key):
            logger.info("训练缓存未命中：超参数 %s 与缓存不一致", key)
            return False, 0.0
    acc = float(meta.get("best_val_acc", 0.0))
    logger.warning(
        "使用训练缓存：跳过训练（best_model.pth + train_cache_meta.json）。"
        "MLflow 不会新建本次训练 Run；若需注册与权重严格一致请关闭 ML_PIPELINE_REUSE_TRAIN_CACHE。"
    )
    return True, acc


def _write_train_cache_meta(all_splits_path: str, best_val_acc: float, params: dict) -> None:
    save_dir = Path(PIPELINE_CONFIG["model_save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    p = Path(all_splits_path).resolve()
    meta = {
        "all_splits_path": str(p),
        "all_splits_mtime_ns": p.stat().st_mtime_ns,
        "best_val_acc": float(best_val_acc),
        "params": {k: params.get(k) for k in _TRAIN_CACHE_PARAM_KEYS},
        "written_at": datetime.utcnow().isoformat() + "Z",
    }
    _train_cache_meta_path().write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("已写入训练缓存元数据: %s", _train_cache_meta_path())


def _ensure_mlflow_artifact_bucket() -> None:
    """
    在训练前确保 MLflow artifact bucket 存在，避免首次运行时报 NoSuchBucket。
    """
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket = os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow")

    if not access_key or not secret_key:
        logger.warning(
            "跳过 bucket 自动检查：缺少 AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
        )
        return

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("跳过 bucket 自动检查：未安装 boto3")
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    try:
        s3.head_bucket(Bucket=bucket)
        logger.info("MLflow artifact bucket 已存在: %s", bucket)
        return
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "NotFound"):
            logger.warning("检查 bucket 失败，继续尝试创建: %s", e)

    s3.create_bucket(Bucket=bucket)
    logger.warning("已自动创建 MLflow artifact bucket: %s", bucket)


# ============================================================================
# DAG配置
# ============================================================================

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": [os.environ.get("ML_PIPELINE_ALERT_EMAIL", "")] if os.environ.get("ML_PIPELINE_ALERT_EMAIL") else [],
    "email_on_failure": bool(os.environ.get("ML_PIPELINE_ALERT_EMAIL")),
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ============================================================================
# 任务函数
# ============================================================================


def ingest_data(**context):
    logger.info("Starting data ingestion...")
    from src.data_ingestion import DataIngestion

    ingestion = DataIngestion(PIPELINE_CONFIG)
    source_csv = Path(PIPELINE_CONFIG["raw_data_path"]) / "cifar-10" / "dataset.csv"
    df = ingestion.ingest_from_csv(str(source_csv))
    output_path = ingestion.save_raw_data(df, "raw_dataset.csv")
    context["task_instance"].xcom_push(key="raw_data_path", value=str(output_path))
    logger.info("Data ingestion complete")
    return "Data ingestion successful"


def validate_data(**context):
    logger.info("Starting data validation...")
    raw_data_path = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="raw_data_path"
    )
    import pandas as pd

    df = pd.read_csv(raw_data_path)
    from src.data_validation import DataValidator

    validator = DataValidator()
    validation_results = validator.run_validation(df)
    if not validation_results.get("all_passed", False):
        raise ValueError("Data validation failed! Check validation report.")
    logger.info("Data validation passed")
    return "Data validation successful"


def preprocess_data(**context):
    logger.info("Starting data preprocessing...")
    raw_data_path = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="raw_data_path"
    )
    import pandas as pd

    from src.preprocessing import DataPreprocessor

    df = pd.read_csv(raw_data_path)
    preprocessor = DataPreprocessor(PIPELINE_CONFIG)
    preprocessor.run_pipeline(df, label_column="label")
    all_splits = Path(PIPELINE_CONFIG["processed_data_path"]) / "all_splits.csv"
    if not all_splits.is_file():
        raise FileNotFoundError(f"Expected merged splits file missing: {all_splits}")
    context["task_instance"].xcom_push(key="all_splits_csv", value=str(all_splits))
    context["task_instance"].xcom_push(key="preprocessing_complete", value=True)
    logger.info("Data preprocessing complete")
    return "Preprocessing successful"


def version_data_dvc(**context):
    logger.info("Versioning data with DVC...")
    import subprocess

    root = PROJECT_ROOT
    cwd = str(root)
    dvc_dir = root / ".dvc"
    git_dir = root / ".git"

    # Airflow Docker 服务默认只挂载 dags/src/data 等目录，不一定包含完整的 DVC/Git 仓库元数据。
    # 在这种集成环境下，DVC 版本化作为可选步骤处理，避免整条训练链因为仓库元数据缺失而中断。
    if not dvc_dir.exists():
        logger.warning(
            "Skipping DVC versioning because %s is missing. "
            "If you want to run DVC inside Airflow, mount the repository metadata into the container.",
            dvc_dir,
        )
        return "DVC versioning skipped (.dvc repo metadata missing)"
    if not git_dir.exists():
        logger.warning(
            "Skipping DVC versioning because %s is missing. "
            "Git metadata is required for this workflow's dvc/git commands.",
            git_dir,
        )
        return "DVC versioning skipped (.git metadata missing)"
    try:
        subprocess.run(
            ["dvc", "add", "data/processed"], check=True, cwd=cwd, capture_output=True, text=True
        )
        subprocess.run(["dvc", "push"], check=True, cwd=cwd, capture_output=True, text=True)
        subprocess.run(
            ["git", "add", "data/processed.dvc", ".gitignore"],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"Data version {datetime.now().isoformat()}",
            ],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        logger.warning("DVC/Git 未安装或不在 PATH，跳过版本化: %s", e)
        return "DVC versioning skipped (dvc/git not available)"
    except subprocess.CalledProcessError as e:
        message = e.stderr or e.stdout or str(e)
        if "not inside of a DVC repository" in message:
            logger.warning("Skipping DVC versioning: %s", message.strip())
            return "DVC versioning skipped (not inside a DVC repository)"
        logger.error("DVC versioning failed: %s", message)
        raise
    logger.info("Data versioning complete")
    return "DVC versioning successful"


def store_features(**context):
    logger.info("Starting feature storage...")
    import pandas as pd

    from src.feature_store import FeatureStore

    all_splits_path = _get_all_splits_csv_path(context)
    df = pd.read_csv(all_splits_path)
    train_df = df[df["split"] == "train"]
    label_col = "label_encoded" if "label_encoded" in train_df.columns else "label"

    host = _feature_store_postgres_host()
    port = int(os.environ.get("FEATURE_STORE_POSTGRES_PORT", "5432"))
    database = os.environ.get("FEATURE_STORE_POSTGRES_DB", "mlflow")
    user = os.environ.get("FEATURE_STORE_POSTGRES_USER", "mlflow")
    password = os.environ.get("FEATURE_STORE_POSTGRES_PASSWORD", "mlflow")

    feature_store = FeatureStore(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
    feature_store.create_table()
    inserted_count = feature_store.save_features_from_dataframe(
        train_df,
        image_id_column="image_path",
        label_column=label_col,
        version="1.0",
    )
    feature_store.close()
    context["task_instance"].xcom_push(key="features_stored", value=int(inserted_count))
    logger.info("Feature storage complete: %s features stored", inserted_count)
    return "Features stored successfully"


def train_model(**context):
    logger.info("Starting model training...")
    from src.dataset import create_data_loaders
    from src.training import MLflowTracker, ModelTrainer

    _ensure_mlflow_artifact_bucket()

    all_splits_path = _get_all_splits_csv_path(context)
    params = {
        "model_name": "resnet18",
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "lr_step_size": 5,
        "lr_gamma": 0.1,
        "early_stopping_patience": 3,
    }
    reused, cached_acc = _try_reuse_train_cache(all_splits_path, params)
    if reused:
        context["task_instance"].xcom_push(
            key="best_val_acc", value=float(cached_acc)
        )
        context["task_instance"].xcom_push(key="train_cache_hit", value=True)
        return "Training skipped (reused checkpoint and cache meta)"

    root_dir = str(PROJECT_ROOT)
    train_loader, val_loader, _ = create_data_loaders(
        csv_path=all_splits_path,
        root_dir=root_dir,
        batch_size=params["batch_size"],
        num_workers=0,
        pin_memory=False,
    )
    num_classes = train_loader.dataset.num_classes
    logger.info("检测到 %s 个类别", num_classes)

    tracker = MLflowTracker(
        tracking_uri=PIPELINE_CONFIG["mlflow_tracking_uri"],
        experiment_name=PIPELINE_CONFIG["experiment_name"],
    )
    trainer = ModelTrainer(PIPELINE_CONFIG, tracker)
    model, best_val_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        params=params,
    )
    _write_train_cache_meta(all_splits_path, best_val_acc, params)
    context["task_instance"].xcom_push(
        key="best_val_acc", value=float(best_val_acc)
    )
    context["task_instance"].xcom_push(key="train_cache_hit", value=False)
    logger.info("Model training complete")
    return "Training successful"


def evaluate_model(**context):
    logger.info("Starting model evaluation...")
    from collections import OrderedDict
    import torch
    import torchvision.models as models

    from src.dataset import create_data_loaders
    from src.evaluation import ModelEvaluator

    all_splits_path = _get_all_splits_csv_path(context)
    _, _, test_loader = create_data_loaders(
        csv_path=all_splits_path,
        root_dir=str(PROJECT_ROOT),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )
    model_path = f"{PIPELINE_CONFIG['model_save_path']}/best_model.pth"
    try:
        model_payload = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        model_payload = torch.load(model_path, map_location=torch.device("cpu"))

    # training.py 保存的是 state_dict；这里兼容 state_dict 与完整模型对象两种格式。
    if isinstance(model_payload, (dict, OrderedDict)):
        num_classes = test_loader.dataset.num_classes
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(model_payload)
        logger.info("Loaded model from state_dict checkpoint")
    else:
        model = model_payload
        logger.info("Loaded serialized model object checkpoint")

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    evaluator = ModelEvaluator(PIPELINE_CONFIG, class_names)
    metrics = evaluator.evaluate(model, test_loader)
    context["task_instance"].xcom_push(
        key="test_metrics", value=_xcom_safe_metrics(metrics)
    )
    logger.info("Model evaluation complete")
    return "Evaluation successful"


def register_model(**context):
    logger.info("Starting model registration...")
    import mlflow

    test_metrics = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="test_metrics"
    )
    if not test_metrics:
        raise ValueError("XCom missing test_metrics from evaluate_model")

    accuracy_threshold = 0.85
    acc = float(test_metrics.get("test_accuracy", 0.0))
    if acc < accuracy_threshold:
        logger.info(
            "Model did not meet criteria (accuracy: %.2f%%)", acc * 100
        )
        return "Model not registered - did not meet criteria"

    mlflow.set_tracking_uri(PIPELINE_CONFIG["mlflow_tracking_uri"])
    experiment = mlflow.get_experiment_by_name(PIPELINE_CONFIG["experiment_name"])
    if experiment is None:
        raise ValueError(
            f"MLflow 中不存在实验: {PIPELINE_CONFIG['experiment_name']!r}"
        )
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs is None or runs.empty:
        raise ValueError("未找到可注册的 MLflow run（search_runs 为空）")
    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name="image_classifier")
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="image_classifier",
        version=result.version,
        stage="Staging",
    )
    logger.info("Registered model version %s", result.version)
    return f"Model registered: version {result.version}"


def notify_pipeline_success(**context):
    """避免未配置 SMTP 时 EmailOperator 导致失败；需要邮件可改回 EmailOperator。"""
    ds = context.get("ds")
    uri = PIPELINE_CONFIG["mlflow_tracking_uri"]
    logger.info("ML 管道已成功完成。ds=%s MLflow=%s", ds, uri)
    return "notification logged"


# ============================================================================
# DAG定义
# ============================================================================

dag = DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline with MLflow tracking",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "production"],
)

with dag:
    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )
    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )
    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )
    task_dvc = PythonOperator(
        task_id="version_data_dvc",
        python_callable=version_data_dvc,
    )
    task_store_features = PythonOperator(
        task_id="store_features",
        python_callable=store_features,
    )
    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )
    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )
    task_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )
    task_notify = PythonOperator(
        task_id="send_success_email",
        python_callable=notify_pipeline_success,
    )

    task_ingest >> task_validate >> task_preprocess >> task_dvc >> task_store_features
    task_store_features >> task_train >> task_evaluate >> task_register >> task_notify


if __name__ == "__main__":
    print(f"DAG: {dag.dag_id}")
    print(f"Schedule: {dag.schedule_interval}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"PIPELINE_CONFIG: {PIPELINE_CONFIG}")
    print(f"Tasks: {len(dag.tasks)}")
    print("\n任务依赖关系：")
    for task in dag.tasks:
        print(
            f"  {task.task_id}: upstream={task.upstream_task_ids}, downstream={task.downstream_task_ids}"
        )
