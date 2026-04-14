"""
用于ML训练管道的Airflow DAG

此DAG编排从数据摄取到模型注册的完整ML管道。

路径与 MLflow 地址通过环境变量配置，便于 conda 本地与 Docker 一致运行：
- ML_PIPELINE_PROJECT_ROOT：项目根（可选；默认取本文件上级目录）
- MLFLOW_TRACKING_URI：默认 http://127.0.0.1:5000（conda）；Docker 可设为 http://mlflow:5000
- FEATURE_STORE_POSTGRES_*：特征库连接（默认与 docker-compose 中 Postgres/mlflow 一致）
"""

from __future__ import annotations

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


def _build_pipeline_config() -> dict:
    root = PROJECT_ROOT
    return {
        "raw_data_path": str(root / "data" / "raw"),
        "processed_data_path": str(root / "data" / "processed"),
        "model_save_path": str(root / "models"),
        "artifacts_path": str(root / "artifacts"),
        "mlflow_tracking_uri": os.environ.get(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
        ),
        "experiment_name": os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "image_classification_pipeline"
        ),
    }


PIPELINE_CONFIG = _build_pipeline_config()


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
        logger.error("DVC versioning failed: %s", e.stderr or e.stdout or e)
        raise
    logger.info("Data versioning complete")
    return "DVC versioning successful"


def store_features(**context):
    logger.info("Starting feature storage...")
    import pandas as pd

    from src.feature_store import FeatureStore

    all_splits_path = context["task_instance"].xcom_pull(
        task_ids="preprocess_data", key="all_splits_csv"
    )
    if not all_splits_path:
        raise ValueError("XCom missing all_splits_csv from preprocess_data")
    df = pd.read_csv(all_splits_path)
    train_df = df[df["split"] == "train"]
    label_col = "label_encoded" if "label_encoded" in train_df.columns else "label"

    host = os.environ.get(
        "FEATURE_STORE_POSTGRES_HOST",
        os.environ.get("POSTGRES_HOST", "localhost"),
    )
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

    all_splits_path = context["task_instance"].xcom_pull(
        task_ids="preprocess_data", key="all_splits_csv"
    )
    if not all_splits_path:
        raise ValueError("XCom missing all_splits_csv")
    root_dir = str(PROJECT_ROOT)
    train_loader, val_loader, _ = create_data_loaders(
        csv_path=all_splits_path,
        root_dir=root_dir,
        batch_size=32,
    )
    num_classes = train_loader.dataset.num_classes
    logger.info("检测到 %s 个类别", num_classes)

    tracker = MLflowTracker(
        tracking_uri=PIPELINE_CONFIG["mlflow_tracking_uri"],
        experiment_name=PIPELINE_CONFIG["experiment_name"],
    )
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
    trainer = ModelTrainer(PIPELINE_CONFIG, tracker)
    model, best_val_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        params=params,
    )
    context["task_instance"].xcom_push(
        key="best_val_acc", value=float(best_val_acc)
    )
    logger.info("Model training complete")
    return "Training successful"


def evaluate_model(**context):
    logger.info("Starting model evaluation...")
    import torch

    from src.dataset import create_data_loaders
    from src.evaluation import ModelEvaluator

    all_splits_path = context["task_instance"].xcom_pull(
        task_ids="preprocess_data", key="all_splits_csv"
    )
    if not all_splits_path:
        raise ValueError("XCom missing all_splits_csv")
    _, _, test_loader = create_data_loaders(
        csv_path=all_splits_path,
        root_dir=str(PROJECT_ROOT),
        batch_size=32,
    )
    model_path = f"{PIPELINE_CONFIG['model_save_path']}/best_model.pth"
    try:
        model = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        model = torch.load(model_path, map_location=torch.device("cpu"))

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
