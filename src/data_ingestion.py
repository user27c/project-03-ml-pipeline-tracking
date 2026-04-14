"""
数据摄取模块

该模块处理从多个源（CSV、API、数据库）摄取数据
并将原始数据保存到管道中。

学习目标：
- 实现多源数据摄取
- 处理不同的数据格式
- 实现错误处理和重试
- 记录摄取元数据

"""

import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from datetime import datetime
import time
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    处理来自多个源的数据摄取。

    该类提供从CSV文件、REST API和数据库加载数据的方法，
    具有适当的错误处理和日志记录。

    属性：
        config (Dict[str, Any]): 配置字典
        raw_data_path (Path): 原始数据目录路径
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        使用配置初始化DataIngestion。

        参数：
            config: 配置字典，包含：
                - raw_data_path: 存储原始数据的路径
                - retry_attempts: API调用的重试次数
                - retry_delay: 重试之间的延迟（秒）

        TODO：
        1. 提取配置参数
        2. 如果不存在则创建raw_data_path目录
        3. 初始化重试设置（默认：3次尝试，5秒延迟）
        4. 记录初始化
        """
        self.config = config
        raw_data_path = config['raw_data_path']
        self.raw_data_path = Path(raw_data_path).expanduser().resolve()
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        #  从配置或使用默认值初始化重试设置
        self.retry_attempts = 3
        self.retry_delay = 5

        #  记录成功初始化
        logger.info("DataIngestion已初始化")

    def ingest_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        从CSV文件摄取数据。

        参数：
            file_path: CSV文件路径

        返回：
            包含加载数据的DataFrame

        异常：
            FileNotFoundError: 如果CSV文件不存在
            pd.errors.ParserError: 如果CSV文件格式错误

        TODO：
        1. 记录带有文件路径的摄取尝试
        2. 尝试使用pandas读取CSV文件
        3. 处理潜在错误（FileNotFoundError，ParserError）
        4. 记录带有记录数的成功消息
        5. 返回DataFrame

        示例：
            >>> ingestion = DataIngestion(config)
            >>> df = ingestion.ingest_from_csv('data/raw/dataset.csv')
            >>> print(len(df))
            50000
        """
        logger.info(f"尝试从CSV摄取数据: {file_path}")

        try:
            #  使用pd.read_csv()读取CSV文件
            df = pd.read_csv(file_path)

            #  记录带有记录数的成功消息
            logger.info(f"成功从CSV加载{len(df)}条记录")

            return df

        except FileNotFoundError:
            #  记录错误并重新抛出
            logger.error(f"CSV文件未找到: {file_path}")
            raise

        except pd.errors.ParserError:
            #  记录错误并重新抛出
            logger.error(f"解析CSV文件失败: {file_path}")
            raise

        except Exception as e:
            #  记录意外错误并重新抛出
            logger.error(f"摄取CSV时出现意外错误: {str(e)}")
            raise

    def ingest_from_api(
        self,
        api_url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        带重试逻辑的REST API数据摄取。

        参数：
            api_url: API端点的URL
            params: 可选的查询参数
            headers: 可选的HTTP头

        返回：
            包含API响应数据的DataFrame

        异常：
            requests.exceptions.RequestException: 如果所有重试尝试都失败
            ValueError: 如果响应不是有效的JSON

        TODO：
        1. 实现重试逻辑（最大retry_attempts）
        2. 向API发出GET请求
        3. 处理HTTP错误（4xx，5xx）
        4. 解析JSON响应
        5. 转换为DataFrame
        6. 返回DataFrame

        重试逻辑：
        - 在网络错误和5xx服务器错误时重试
        - 不在4xx客户端错误时重试
        - 使用指数退避（delay * attempt_number）

        示例：
            >>> ingestion = DataIngestion(config)
            >>> df = ingestion.ingest_from_api('https://api.example.com/data')
            >>> print(df.columns)
            Index(['id', 'name', 'value'], dtype='object')
        """
        logger.info(f"尝试从API摄取数据: {api_url}")

        #  实现重试循环
        for attempt in range(self.retry_attempts):
            try:
                #  发出GET请求
                response = requests.get(api_url, params=params, headers=headers, timeout=30)

                #  为HTTP错误抛出异常
                response.raise_for_status()

                #  解析JSON响应
                data = response.json()

                #  转换为DataFrame
                df = pd.DataFrame(data)

                #  记录成功
                logger.info(f"成功从API获取{len(df)}条记录")

                #  返回DataFrame   
                return df

            except requests.exceptions.HTTPError as e:
                #  检查错误是否为4xx（客户端错误）- 不重试
                if 400 <= e.response.status_code < 500:
                    #  对于5xx错误，使用退避重试
                    logger.warning(f"API请求失败（尝试 {attempt + 1}/{self.retry_attempts}）: {e}")

                if attempt < self.retry_attempts - 1:
                    #  计算退避延迟
                    backoff_delay = self.retry_delay * (attempt + 1)
                    logger.info(f"将在{backoff_delay}秒后重试...")
                    time.sleep(backoff_delay)
                else:
                    #  记录最终失败并重新抛出
                    logger.error(f"API的所有重试尝试都失败: {api_url}")
                    raise

            except requests.exceptions.RequestException as e:
                #  使用重试处理网络错误
                logger.warning(f"网络错误（尝试 {attempt + 1}/{self.retry_attempts}）: {e}")

                if attempt < self.retry_attempts - 1:
                    backoff_delay = self.retry_delay * (attempt + 1)
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"API的所有重试尝试都失败: {api_url}")
                    raise

            except ValueError as e:
                #  处理JSON解析错误
                logger.error(f"将API响应解析为JSON失败: {e}")
                raise

    def ingest_from_database(
        self,
        connection_string: str,
        query: str
    ) -> pd.DataFrame:
        """
        使用SQL查询从数据库摄取数据。

        参数：
            connection_string: 数据库连接字符串
                格式：'postgresql://user:password@host:port/database'
            query: 要执行的SQL查询

        返回：
            包含查询结果的DataFrame

        异常：
            sqlalchemy.exc.OperationalError: 如果连接失败
            sqlalchemy.exc.ProgrammingError: 如果查询无效

        说明：
        1. 记录摄取尝试（清理连接字符串！）
        2. 使用pandas执行SQL查询
        3. 处理连接错误
        4. 处理查询错误
        5. 记录带有记录数的成功消息
        6. 返回DataFrame

        安全说明：
        - 永远不要记录密码或敏感凭证
        - 记录前清理连接字符串

        示例：
            >>> ingestion = DataIngestion(config)
            >>> conn_str = 'postgresql://user:pass@localhost:5432/mydb'
            >>> query = 'SELECT * FROM images WHERE split = "train"'
            >>> df = ingestion.ingest_from_database(conn_str, query)
            >>> print(len(df))
            35000
        """
        #  清理连接字符串以用于日志记录
        # 提示：在记录前将密码替换为'***'
        sanitized_conn = self._sanitize_connection_string(connection_string)
        logger.info(f"尝试从数据库摄取数据: {sanitized_conn}")

        try:
            #  使用pd.read_sql()执行查询
            # 提示：df = pd.read_sql(query, connection_string)
            from sqlalchemy import create_engine
            
            # 创建SQLAlchemy引擎
            engine = create_engine(connection_string)
            
            # 执行查询
            df = pd.read_sql(query, engine)
            
            #  记录带有记录数的成功消息
            logger.info(f"成功从数据库查询{len(df)}条记录")

            return df

        except Exception as e:
            #  使用清理后的连接字符串记录错误
            logger.error(f"查询数据库{sanitized_conn}失败: {str(e)}")
            raise

    def save_raw_data(
        self,
        df: pd.DataFrame,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        将原始数据保存到磁盘并附带元数据。

        参数：
            df: 要保存的DataFrame
            filename: 文件名（例如：'dataset.csv'）
            metadata: 可选的与数据一起保存的元数据

        返回：
            保存文件的路径

        说明：
        1. 创建完整输出路径（raw_data_path / filename）
        2. 将DataFrame保存为CSV
        3. 如果未提供则创建元数据字典
        4. 将元数据保存为JSON（相同名称，扩展名为.meta.json）
        5. 记录成功
        6. 返回Path对象

        元数据应包括：
        - filename
        - record_count
        - column_count
        - columns (list)
        - saved_at (timestamp)
        - file_size (bytes)

        示例：
            >>> ingestion = DataIngestion(config)
            >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
            >>> path = ingestion.save_raw_data(df, 'test.csv')
            >>> print(path.exists())
            True
        """
        #  创建完整输出路径
        output_path = self.raw_data_path / filename

        #  将DataFrame保存为CSV（使用processed目录以避免权限问题）
        processed_path = Path('/opt/airflow/data/processed')
        processed_path.mkdir(parents=True, exist_ok=True)
        output_path = processed_path / filename
        df.to_csv(output_path, index=False)

        #  如果未提供则创建元数据字典
        if metadata is None:
            metadata = {
                #  填写元数据字段
                "filename": filename,
                "record_count": len(df),  # 用len(df)替换
                "column_count": len(df.columns),  # 用len(df.columns)替换
                "columns": list(df.columns),  # 用list(df.columns)替换
                "saved_at": datetime.now().isoformat(),  # 用datetime.now().isoformat()替换
                "file_size_bytes": output_path.stat().st_size  # 用output_path.stat().st_size替换
            }

        #  将元数据保存为JSON
        # 提示：使用json.dump()或写入.meta.json文件
        metadata_path = output_path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        #  记录成功
        logger.info(f"将原始数据保存到{output_path}（{len(df)}条记录）")

        return output_path

    def get_ingestion_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        检索先前保存的原始数据文件的元数据。

        参数：
            file_path: 原始数据文件的路径

        返回：
            包含元数据的字典

        TODO：
        1. 构造元数据文件路径（.meta.json）
        2. 加载并解析JSON元数据
        3. 返回元数据字典
        4. 处理元数据文件不存在的情况

        示例：
            >>> ingestion = DataIngestion(config)
            >>> metadata = ingestion.get_ingestion_metadata(Path('data/raw/dataset.csv'))
            >>> print(metadata['record_count'])
            50000
        """
        #  构造元数据文件路径
        metadata_path = file_path.with_suffix('.meta.json')

        #  加载并返回元数据
        metadata = metadata_path.load()

        return metadata

    def _sanitize_connection_string(self, conn_str: str) -> str:
        """
        清理数据库连接字符串以用于安全日志记录。

        参数：
            conn_str: 数据库连接字符串

        返回：
            清理后的连接字符串，密码已替换

        TODO：
        1. 使用正则表达式在连接字符串中查找密码
        2. 将密码替换为'***'
        3. 返回清理后的字符串

        示例：
            >>> conn = 'postgresql://user:password@localhost:5432/db'
            >>> sanitized = self._sanitize_connection_string(conn)
            >>> print(sanitized)
            'postgresql://user:***@localhost:5432/db'
        """
        import re

        # TODO: 实现清理
        # 提示：使用正则表达式模式匹配密码
        # 模式：r':([^@]+)@' 捕获':'和'@'之间的密码
        conn_str = re.sub(r':([^@]+)@', r':***@', conn_str)
        return conn_str  


# 示例用法和测试
if __name__ == "__main__":
    """
    DataIngestion类的示例用法。

    说明：
    1. 创建示例配置
    2. 初始化DataIngestion
    3. 测试CSV摄取
    4. 测试API摄取（使用公共API）
    5. 测试save_raw_data

    尝试使用这些公共API进行测试：
    - JSONPlaceholder: https://jsonplaceholder.typicode.com/users
    - Open Brewery DB: https://api.openbrewerydb.org/breweries
    - REST Countries: https://restcountries.com/v3.1/all
    """

    # 示例配置
    config = {
        'raw_data_path': 'data/raw',
        'retry_attempts': 3,
        'retry_delay': 5
    }

    # 初始化DataIngestion
    ingestion = DataIngestion(config)

    # 测试CSV摄取
    # 首先为测试创建示例CSV文件
    df = ingestion.ingest_from_csv('data/raw/dataset.csv')

    # 测试API摄取
    df = ingestion.ingest_from_api('https://jsonplaceholder.typicode.com/users')

    # 测试save_raw_data
    path = ingestion.save_raw_data(df, 'api_data.csv')

    print("DataIngestion模块已加载。实现结束。")
