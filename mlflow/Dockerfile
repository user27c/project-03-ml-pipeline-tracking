FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY ./mlflow/requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY src/ ./src/
COPY dags/ ./dags/
COPY data/ ./data/
COPY models/ ./models/
COPY artifacts/ ./artifacts/

# 安装项目依赖
RUN pip install --no-cache-dir mlflow psycopg2-binary scikit-learn matplotlib seaborn pandas numpy torch torchvision

# 暴露端口
EXPOSE 5000

# 启动 MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
