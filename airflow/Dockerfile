FROM apache/airflow:2.7.3

# 配置 pip 镜像源（阿里云 + 清华大学）
# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
#     pip config set install.trusted-host mirrors.aliyun.com

# 安装额外依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir psycopg2-binary pandas scikit-learn matplotlib seaborn onnx

# 训练/评估任务依赖（src/training.py、src/evaluation.py）
RUN pip install --no-cache-dir mlflow

# PyTorch CUDA 依赖与 Airflow 基础镜像的 PIP_CONSTRAINT（hash 锁定）冲突时会报
# “THESE PACKAGES DO NOT MATCH THE HASHES”。对本条 pip 去掉约束环境变量。
RUN env -u PIP_CONSTRAINT -u AIRFLOW_CONSTRAINTS_URL \
    pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 安装 DVC（使用最新稳定版本）
RUN pip install --no-cache-dir dvc dvc-s3

# 复制项目文件
COPY src/ /opt/airflow/src/
COPY dags/ /opt/airflow/dags/

# 设置 PYTHONPATH
ENV PYTHONPATH=/opt/airflow/src:/opt/airflow/dags:${PYTHONPATH}

# 设置用户为 root 以避免权限问题
USER root