"""
特征存储模块

在PostgreSQL中存储和检索处理后的特征向量。

"""

import os
import logging
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    特征存储类，用于在PostgreSQL中存储和检索特征向量。
    
    属性：
        host (str): PostgreSQL主机地址
        port (int): PostgreSQL端口
        database (str): 数据库名称
        user (str): 用户名
        password (str): 密码
        table_name (str): 特征表名
        connection: 数据库连接对象
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'mlflow',
        user: str = 'mlflow',
        password: str = 'mlflow',
        table_name: str = 'features'
    ) -> None:
        """
        初始化特征存储。
        
        参数：
            host: PostgreSQL主机地址
            port: PostgreSQL端口
            database: 数据库名称
            user: 用户名
            password: 密码
            table_name: 特征表名
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table_name = table_name
        self.connection = None
        
        logger.info(f"初始化特征存储: {host}:{port}/{database}")
    
    def connect(self) -> None:
        """
        建立数据库连接。
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info("成功连接到PostgreSQL数据库")
        except Exception as e:
            logger.error(f"连接数据库失败: {e}")
            raise
    
    def close(self) -> None:
        """
        关闭数据库连接。
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("数据库连接已关闭")
    
    def create_table(self) -> None:
        """
        创建特征表（如果不存在）。
        """
        if not self.connection:
            self.connect()
        
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            image_id VARCHAR(255) UNIQUE NOT NULL,
            feature_vector FLOAT[] NOT NULL,
            label INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            version VARCHAR(50) DEFAULT '1.0'
        );
        
        CREATE INDEX IF NOT EXISTS idx_image_id ON {self.table_name}(image_id);
        CREATE INDEX IF NOT EXISTS idx_label ON {self.table_name}(label);
        CREATE INDEX IF NOT EXISTS idx_created_at ON {self.table_name}(created_at);
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_query)
            self.connection.commit()
            logger.info(f"特征表 {self.table_name} 创建成功")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"创建特征表失败: {e}")
            raise
    
    def insert_features(
        self,
        image_ids: List[str],
        feature_vectors: List[List[float]],
        labels: List[int],
        version: str = '1.0'
    ) -> int:
        """
        批量插入特征向量。
        
        参数：
            image_ids: 图像ID列表
            feature_vectors: 特征向量列表
            labels: 标签列表
            version: 版本号
            
        返回：
            int: 插入的记录数
        """
        if not self.connection:
            self.connect()
        
        if len(image_ids) != len(feature_vectors) or len(image_ids) != len(labels):
            raise ValueError("image_ids, feature_vectors, 和 labels 的长度必须相同")
        
        insert_query = f"""
        INSERT INTO {self.table_name} (image_id, feature_vector, label, version)
        VALUES %s
        ON CONFLICT (image_id) DO UPDATE
        SET feature_vector = EXCLUDED.feature_vector,
            label = EXCLUDED.label,
            version = EXCLUDED.version,
            created_at = NOW();
        """
        
        try:
            with self.connection.cursor() as cursor:
                values = [
                    (image_id, vector, label, version)
                    for image_id, vector, label in zip(image_ids, feature_vectors, labels)
                ]
                execute_values(cursor, insert_query, values)
            self.connection.commit()
            logger.info(f"成功插入 {len(values)} 条特征记录")
            return len(values)
        except Exception as e:
            self.connection.rollback()
            logger.error(f"插入特征失败: {e}")
            raise
    
    def get_features_by_image_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        根据图像ID检索特征。
        
        参数：
            image_id: 图像ID
            
        返回：
            dict: 特征记录，如果不存在则返回None
        """
        if not self.connection:
            self.connect()
        
        select_query = f"""
        SELECT id, image_id, feature_vector, label, created_at, version
        FROM {self.table_name}
        WHERE image_id = %s;
        """
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (image_id,))
                result = cursor.fetchone()
            
            if result:
                logger.info(f"成功检索图像 {image_id} 的特征")
                return dict(result)
            else:
                logger.warning(f"未找到图像 {image_id} 的特征")
                return None
        except Exception as e:
            logger.error(f"检索特征失败: {e}")
            raise
    
    def get_features_by_label(self, label: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据标签检索特征。
        
        参数：
            label: 标签值
            limit: 返回记录数限制
            
        返回：
            list: 特征记录列表
        """
        if not self.connection:
            self.connect()
        
        select_query = f"""
        SELECT id, image_id, feature_vector, label, created_at, version
        FROM {self.table_name}
        WHERE label = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (label, limit))
                results = cursor.fetchall()
            
            logger.info(f"检索到 {len(results)} 条标签为 {label} 的特征")
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"检索特征失败: {e}")
            raise
    
    def get_all_features(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取所有特征。
        
        参数：
            limit: 返回记录数限制
            
        返回：
            list: 所有特征记录列表
        """
        if not self.connection:
            self.connect()
        
        query = f"SELECT * FROM {self.table_name}"
        if limit:
            query += f" LIMIT {limit}"
        query += ";"
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
            
            logger.info(f"检索到 {len(results)} 条特征记录")
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"检索特征失败: {e}")
            raise
    
    def get_version_stats(self) -> List[Dict[str, Any]]:
        """
        获取版本统计信息。
        
        返回：
            list: 每个版本的统计信息
        """
        if not self.connection:
            self.connect()
        
        query = f"""
        SELECT 
            version,
            COUNT(*) as record_count,
            MIN(created_at) as first_recorded,
            MAX(created_at) as last_recorded
        FROM {self.table_name}
        GROUP BY version
        ORDER BY version;
        """
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
            
            logger.info(f"获取到 {len(results)} 个版本的统计信息")
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"获取版本统计失败: {e}")
            raise
    
    def delete_features(self, image_ids: List[str]) -> int:
        """
        删除指定图像ID的特征。
        
        参数：
            image_ids: 图像ID列表
            
        返回：
            int: 删除的记录数
        """
        if not self.connection:
            self.connect()
        
        delete_query = f"DELETE FROM {self.table_name} WHERE image_id = ANY(%s);"
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(delete_query, (image_ids,))
                deleted_count = cursor.rowcount
            self.connection.commit()
            logger.info(f"删除了 {deleted_count} 条特征记录")
            return deleted_count
        except Exception as e:
            self.connection.rollback()
            logger.error(f"删除特征失败: {e}")
            raise
    
    def save_features_from_dataframe(
        self,
        df: pd.DataFrame,
        image_id_column: str = 'image_path',
        label_column: str = 'label',
        version: str = '1.0'
    ) -> int:
        """
        从DataFrame保存特征。
        
        参数：
            df: 包含特征的DataFrame
            image_id_column: 图像ID列名
            label_column: 标签列名
            version: 版本号
            
        返回：
            int: 插入的记录数
        """
        # 提取图像ID（从路径中提取文件名）
        image_ids = df[image_id_column].apply(
            lambda x: os.path.basename(x) if isinstance(x, str) else str(x)
        ).tolist()
        
        # 创建虚拟特征向量（实际项目中应该从模型提取）
        # 这里使用简单的像素统计作为示例
        num_samples = len(df)
        feature_dim = 10  # 简化的特征维度
        feature_vectors = [
            np.random.randn(feature_dim).tolist() for _ in range(num_samples)
        ]
        
        labels_raw = df[label_column].tolist()
        if not labels_raw:
            labels = []
        elif isinstance(labels_raw[0], str):
            codes, _ = pd.factorize(labels_raw)
            labels = [int(x) for x in codes]
        else:
            labels = [int(x) for x in labels_raw]

        return self.insert_features(image_ids, feature_vectors, labels, version)
    
    def __enter__(self):
        """上下文管理器入口。"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口。"""
        self.close()
        return False


# 示例用法
if __name__ == "__main__":
    """示例用法。"""
    # 初始化特征存储
    # 注意：使用 docker-compose.yml 中配置的 PostgreSQL 连接参数
    feature_store = FeatureStore(
        host='localhost',
        port=5432,
        database='mlflow',  # 与 docker-compose.yml 中的 POSTGRES_DB 一致
        user='mlflow',      # 与 docker-compose.yml 中的 POSTGRES_USER 一致
        password='mlflow'   # 与 docker-compose.yml 中的 POSTGRES_PASSWORD 一致
    )
    
    # 创建表
    feature_store.create_table()
    
    # 插入示例数据
    image_ids = ['test1.png', 'test2.png', 'test3.png']
    feature_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    labels = [0, 1, 2]
    
    feature_store.insert_features(image_ids, feature_vectors, labels, version='1.0')
    
    # 检索特征
    result = feature_store.get_features_by_image_id('test1.png')
    print(f"检索结果: {result}")
    
    # 获取版本统计
    stats = feature_store.get_version_stats()
    print(f"版本统计: {stats}")
    
    # 关闭连接
    feature_store.close()
