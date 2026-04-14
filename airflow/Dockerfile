FROM apache/airflow:2.7.3

# 配置 pip 镜像源（阿里云 + 清华大学）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set install.trusted-host mirrors.aliyun.com

# 安装额外依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir psycopg2-binary pandas scikit-learn matplotlib seaborn

# 安装 PyTorch（使用清华镜像）
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 DVC（使用最新稳定版本）
RUN pip install --no-cache-dir dvc dvc-s3

# 复制项目文件
COPY src/ /opt/airflow/src/
COPY dags/ /opt/airflow/dags/

# 设置 PYTHONPATH
ENV PYTHONPATH=/opt/airflow/src:/opt/airflow/dags:${PYTHONPATH}

# 设置用户为 root 以避免权限问题
USER root