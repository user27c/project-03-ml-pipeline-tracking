"""
用于ML训练管道的Airflow DAG

此DAG编排从数据摄取到模型注册的完整ML管道。

学习目标：
- 设计具有适当任务依赖关系的Airflow DAG
- 为ML任务实现PythonOperator
- 使用XCom进行任务间通信
- 处理错误和重试
- 调度管道

"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 将项目源代码添加到Python路径
sys.path.insert(0, '/path/to/src')

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DAG配置
# ============================================================================

# 为所有任务定义default_args
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['x22han7@qq.com'],  # TODO: 更新为你的邮箱
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,  # TODO: 设置适当的重试次数
    'retry_delay': timedelta(minutes=5),  # TODO: 设置重试延迟
    # TODO: 添加execution_timeout
    'execution_timeout': timedelta(hours=2),
}

# TODO: 定义管道配置
PIPELINE_CONFIG = {
    'raw_data_path': '/opt/airflow/data/raw',
    'processed_data_path': '/opt/airflow/data/processed',
    'model_save_path': '/opt/airflow/models',
    'artifacts_path': '/opt/airflow/artifacts',
    'mlflow_tracking_uri': 'http://mlflow:5000',
    'experiment_name': 'image_classification_pipeline',
}


# ============================================================================
# 任务函数
# ============================================================================

def ingest_data(**context):
    """
    任务：从源摄取数据。

    TODO：
    1. 导入DataIngestion类
    2. 使用配置初始化
    3. 从CSV（或API、数据库）摄取数据
    4. 保存原始数据
    5. 将数据路径推送到XCom供下一个任务使用
    6. 返回成功消息
    """
    logger.info("Starting data ingestion...")

    # TODO: 导入DataIngestion
    # from src.data_ingestion import DataIngestion

    # TODO: 初始化摄取
    # ingestion = DataIngestion(PIPELINE_CONFIG)

    # TODO: 摄取数据
    # 示例：df = ingestion.ingest_from_csv('/opt/airflow/data/source/dataset.csv')

    # TODO: 保存原始数据
    # output_path = ingestion.save_raw_data(df, 'raw_dataset.csv')

    # TODO: 将路径推送到XCom
    # context['task_instance'].xcom_push(key='raw_data_path', value=str(output_path))

    logger.info("Data ingestion complete")
    return "Data ingestion successful"


def validate_data(**context):
    """
    任务：使用Great Expectations验证数据质量。

    TODO：
    1. 从XCom拉取原始数据路径
    2. 加载数据
    3. 初始化DataValidator
    4. 创建期望套件
    5. 运行验证
    6. 如果验证失败则抛出错误
    7. 返回验证结果
    """
    logger.info("Starting data validation...")

    # TODO: 从上一个任务拉取数据路径
    # raw_data_path = context['task_instance'].xcom_pull(
    #     task_ids='ingest_data',
    #     key='raw_data_path'
    # )

    # TODO: 加载数据
    # import pandas as pd
    # df = pd.read_csv(raw_data_path)

    # TODO: 导入并初始化验证器
    # from src.data_validation import DataValidator
    # validator = DataValidator()

    # TODO: 创建期望
    # validator.create_expectation_suite("data_quality_suite")

    # TODO: 验证
    # validation_passed = validator.validate_data(df, "data_quality_suite")

    # TODO: 如果验证失败则抛出错误
    # if not validation_passed:
    #     raise ValueError("Data validation failed! Check validation report.")

    logger.info("Data validation passed")
    return "Data validation successful"


def preprocess_data(**context):
    """
    任务：预处理数据（清洗、编码、分割）。

    TODO：
    1. 从XCom拉取原始数据路径
    2. 加载数据
    3. 初始化DataPreprocessor
    4. 运行预处理管道
    5. 将完成状态推送到XCom
    6. 返回成功消息
    """
    logger.info("Starting data preprocessing...")

    # TODO: 拉取数据路径
    # raw_data_path = context['task_instance'].xcom_pull(
    #     task_ids='ingest_data',
    #     key='raw_data_path'
    # )

    # TODO: 加载数据
    # import pandas as pd
    # df = pd.read_csv(raw_data_path)

    # TODO: 导入并初始化预处理器
    # from src.preprocessing import DataPreprocessor
    # preprocessor = DataPreprocessor(PIPELINE_CONFIG)

    # TODO: 运行管道
    # train, val, test = preprocessor.run_pipeline(df, label_column='label')

    # TODO: 将状态推送到XCom
    # context['task_instance'].xcom_push(key='preprocessing_complete', value=True)

    logger.info("Data preprocessing complete")
    return "Preprocessing successful"


def version_data_dvc(**context):
    """
    任务：使用DVC版本化处理后的数据。

    TODO：
    1. 在处理后的数据目录上运行dvc add
    2. 将DVC文件提交到git
    3. 推送到DVC远程存储
    4. 使用版本标记
    5. 返回成功消息

    注意：这需要在Airflow容器中设置DVC和Git
    """
    logger.info("Versioning data with DVC...")

    # TODO: 导入subprocess
    # import subprocess

    # TODO: 将处理后的数据添加到DVC
    # try:
    #     subprocess.run(['dvc', 'add', 'data/processed'], check=True)
    #     subprocess.run(['dvc', 'push'], check=True)
    #     subprocess.run(['git', 'add', 'data/processed.dvc', '.gitignore'], check=True)
    #     subprocess.run(
    #         ['git', 'commit', '-m', f'Data version {datetime.now().isoformat()}'],
    #         check=True
    #     )
    # except subprocess.CalledProcessError as e:
    #     logger.error(f"DVC versioning failed: {e}")
    #     raise

    logger.info("Data versioning complete")
    return "DVC versioning successful"


def train_model(**context):
    """
    任务：使用MLflow跟踪训练ML模型。

    TODO：
    1. 初始化MLflowTracker
    2. 加载预处理数据
    3. 创建数据加载器
    4. 定义训练参数
    5. 初始化ModelTrainer
    6. 运行训练
    7. 将最佳验证准确率推送到XCom
    8. 返回成功消息
    """
    logger.info("Starting model training...")

    # TODO: 导入所需的类
    # from src.training import MLflowTracker, ModelTrainer
    # import pandas as pd

    # TODO: 初始化MLflow跟踪器
    # tracker = MLflowTracker(
    #     tracking_uri=PIPELINE_CONFIG['mlflow_tracking_uri'],
    #     experiment_name=PIPELINE_CONFIG['experiment_name']
    # )

    # TODO: 加载处理后的数据
    # train_df = pd.read_csv(f"{PIPELINE_CONFIG['processed_data_path']}/train.csv")
    # val_df = pd.read_csv(f"{PIPELINE_CONFIG['processed_data_path']}/val.csv")

    # TODO: 创建数据加载器
    # 注意：你需要为数据实现一个Dataset类
    # train_loader = ...
    # val_loader = ...

    # TODO: 定义训练参数
    params = {
        'model_name': 'resnet18',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'lr_step_size': 5,
        'lr_gamma': 0.1,
        'early_stopping_patience': 3
    }

    # TODO: 初始化训练器
    # trainer = ModelTrainer(PIPELINE_CONFIG, tracker)

    # TODO: 运行训练
    # model, best_val_acc = trainer.train(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_classes=4,
    #     params=params
    # )

    # TODO: 将指标推送到XCom
    # context['task_instance'].xcom_push(key='best_val_acc', value=best_val_acc)

    logger.info("Model training complete")
    return "Training successful"


def evaluate_model(**context):
    """
    任务：在测试集上评估模型。

    TODO：
    1. 加载测试数据
    2. 加载最佳模型
    3. 初始化ModelEvaluator
    4. 运行评估
    5. 将测试指标推送到XCom
    6. 返回成功消息
    """
    logger.info("Starting model evaluation...")

    # TODO: 导入所需的类
    # from src.evaluation import ModelEvaluator
    # import pandas as pd
    # import torch

    # TODO: 加载测试数据
    # test_df = pd.read_csv(f"{PIPELINE_CONFIG['processed_data_path']}/test.csv")

    # TODO: 创建测试数据加载器
    # test_loader = ...

    # TODO: 加载最佳模型
    # model_path = f"{PIPELINE_CONFIG['model_save_path']}/best_model.pth"
    # model = torch.load(model_path)

    # TODO: 初始化评估器
    # class_names = ['cat', 'dog', 'bird', 'fish']
    # evaluator = ModelEvaluator(PIPELINE_CONFIG, class_names)

    # TODO: 运行评估
    # metrics = evaluator.evaluate(model, test_loader)

    # TODO: 将指标推送到XCom
    # context['task_instance'].xcom_push(key='test_metrics', value=metrics)

    logger.info("Model evaluation complete")
    return "Evaluation successful"


def register_model(**context):
    """
    任务：如果模型满足条件，将其注册到MLflow模型注册表。

    TODO：
    1. 从XCom拉取测试指标
    2. 检查模型是否满足生产标准（例如准确率 >= 85%）
    3. 如果满足，在MLflow中注册模型
    4. 过渡到Staging阶段
    5. 返回注册结果
    """
    logger.info("Starting model registration...")

    # TODO: 拉取测试指标
    # test_metrics = context['task_instance'].xcom_pull(
    #     task_ids='evaluate_model',
    #     key='test_metrics'
    # )

    # TODO: 检查生产标准
    # accuracy_threshold = 0.85
    # if test_metrics['test_accuracy'] >= accuracy_threshold:
    #     # TODO: 导入MLflow
    #     import mlflow
    #
    #     # TODO: 获取最新运行ID
    #     experiment = mlflow.get_experiment_by_name(PIPELINE_CONFIG['experiment_name'])
    #     runs = mlflow.search_runs(
    #         experiment_ids=[experiment.experiment_id],
    #         order_by=["start_time DESC"],
    #         max_results=1
    #     )
    #     run_id = runs.iloc[0]['run_id']
    #
    #     # TODO: 注册模型
    #     model_uri = f"runs:/{run_id}/model"
    #     result = mlflow.register_model(
    #         model_uri=model_uri,
    #         name="image_classifier"
    #     )
    #
    #     # TODO: 过渡到Staging
    #     client = mlflow.tracking.MlflowClient()
    #     client.transition_model_version_stage(
    #         name="image_classifier",
    #         version=result.version,
    #         stage="Staging"
    #     )
    #
    #     logger.info(f"Registered model version {result.version}")
    #     return f"Model registered: version {result.version}"
    # else:
    #     logger.info(f"Model did not meet criteria (accuracy: {test_metrics['test_accuracy']:.2%})")
    #     return "Model not registered - did not meet criteria"

    return "Model registration complete"


# ============================================================================
# DAG定义
# ============================================================================

# TODO: 创建DAG
dag = DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline with MLflow tracking',
    # TODO: 设置调度（每周日午夜）
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,  # 不为过去日期运行
    max_active_runs=1,  # 同时只运行一个
    tags=['ml', 'training', 'production'],
)

# TODO: 定义任务
with dag:
    # 任务1：摄取数据
    task_ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        # TODO: 如果使用Airflow < 2.0则添加provide_context=True
    )

    # 任务2：验证数据
    task_validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )

    # 任务3：预处理数据
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    # 任务4：使用DVC版本化数据
    task_dvc = PythonOperator(
        task_id='version_data_dvc',
        python_callable=version_data_dvc,
    )

    # 任务5：训练模型
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # 任务6：评估模型
    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    # 任务7：注册模型
    task_register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    # 任务8：发送成功邮件
    task_notify = EmailOperator(
        task_id='send_success_email',
        to='mlops@example.com',  # TODO: 更新邮箱
        subject='[成功] ML训练管道 - {{ ds }}',
        html_content="""
        <h3>ML训练管道成功完成</h3>
        <p><strong>执行日期：</strong> {{ ds }}</p>
        <p><strong>状态：</strong> 成功</p>
        <p>在MLflow中查看结果：<a href="http://mlflow:5000">MLflow UI</a></p>
        <p>查看管道：<a href="http://airflow:8080/dags/ml_training_pipeline/grid">Airflow DAG</a></p>
        """,
    )

    # TODO: 定义任务依赖关系
    # 管道应按以下顺序流动：
    # 摄取 → 验证 → 预处理 → 版本化 → 训练 → 评估 → 注册 → 通知

    task_ingest >> task_validate >> task_preprocess >> task_dvc
    task_dvc >> task_train >> task_evaluate >> task_register >> task_notify


# ============================================================================
# DAG测试（用于本地开发）
# ============================================================================

if __name__ == "__main__":
    """
    测试DAG结构而不运行任务。

    TODO：
    1. 打印DAG信息
    2. 验证任务依赖关系
    3. 检查循环
    """
    print(f"DAG: {dag.dag_id}")
    print(f"Schedule: {dag.schedule_interval}")
    print(f"Tasks: {len(dag.tasks)}")
    print("\n任务依赖关系：")
    for task in dag.tasks:
        print(f"  {task.task_id}: upstream={task.upstream_task_ids}, downstream={task.downstream_task_ids}")
