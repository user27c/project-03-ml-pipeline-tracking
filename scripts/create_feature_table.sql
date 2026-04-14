-- 创建特征表
-- 用于存储处理后的特征向量

CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    feature_vector FLOAT[] NOT NULL,
    label INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    version VARCHAR(50) DEFAULT '1.0'
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_image_id ON features(image_id);
CREATE INDEX IF NOT EXISTS idx_label ON features(label);
CREATE INDEX IF NOT EXISTS idx_created_at ON features(created_at);
CREATE INDEX IF NOT EXISTS idx_version ON features(version);

-- 添加注释
COMMENT ON TABLE features IS '存储图像特征向量';
COMMENT ON COLUMN features.image_id IS '图像文件名';
COMMENT ON COLUMN features.feature_vector IS '特征向量（10维简化示例）';
COMMENT ON COLUMN features.label IS '标签索引';
COMMENT ON COLUMN features.version IS '特征版本号';
