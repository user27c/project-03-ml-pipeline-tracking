"""
数据验证模块

使用 Great Expectations 验证数据质量。

验证规则：
- 最少1,000条记录，最多1,000,000条记录
- 必需列：image_path、label、split
- 标签值必须在定义的集合中（CIFAR-10 10个类别）
- 关键列中无空值
- 图像文件路径必须存在

"""

import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据验证器，验证数据质量。
    
    属性：
        config (Dict[str, Any]): 配置字典
        required_columns (List[str]): 必需列列表
        valid_labels (List[str]): 有效标签值列表
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        初始化验证器。
        
        参数：
            config: 配置字典，包含：
                - data_path: 要验证的数据文件路径
                - output_dir: 验证报告输出目录
        """
        self.config = config or {}
        self.data_path = Path(self.config.get('data_path', 'data/raw/cifar-10/dataset.csv'))
        self.output_dir = Path(self.config.get('output_dir', 'data/validation'))
        
        # CIFAR-10 10个类别
        self.valid_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        self.required_columns = ["image_path", "label", "split"]
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataValidator 已初始化")
    
    def validate_record_count(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证记录数量是否在允许范围内。
        
        参数：
            df: DataFrame 数据
            
        返回：
            Dict: 验证结果字典
        """
        min_records = 1000
        max_records = 1000000
        record_count = len(df)
        
        passed = min_records <= record_count <= max_records
        
        return {
            'check': '记录数量验证',
            'passed': passed,
            'message': f"记录数 {record_count} {'在' if passed else '超出'}允许范围 ({min_records}-{max_records})",
            'record_count': record_count,
            'min_value': min_records,
            'max_value': max_records
        }
    
    def validate_required_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证必需列是否存在。
        
        参数：
            df: DataFrame 数据
            
        返回：
            Dict: 验证结果字典
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        passed = len(missing_columns) == 0
        
        return {
            'check': '必需列验证',
            'passed': passed,
            'message': f"所有必需列都存在: {self.required_columns}" if passed else f"缺少必需列: {missing_columns}",
            'required_columns': self.required_columns,
            'missing_columns': missing_columns
        }
    
    def validate_label_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证标签值是否在有效范围内。
        
        参数：
            df: DataFrame 数据
            
        返回：
            Dict: 验证结果字典
        """
        invalid_labels = []
        for label in df['label'].unique():
            if label not in self.valid_labels:
                invalid_labels.append(label)
        
        passed = len(invalid_labels) == 0
        
        return {
            'check': '标签值验证',
            'passed': passed,
            'message': f"所有标签值都有效: {sorted(df['label'].unique())}" if passed else f"发现无效标签值: {invalid_labels}",
            'valid_labels': self.valid_labels,
            'invalid_labels': invalid_labels,
            'unique_labels': sorted(df['label'].unique().tolist())
        }
    
    def validate_no_null_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证关键列中没有空值。
        
        参数：
            df: DataFrame 数据
            
        返回：
            Dict: 验证结果字典
        """
        null_counts = df[self.required_columns].isnull().sum()
        null_columns = null_counts[null_counts > 0].index.tolist()
        
        passed = len(null_columns) == 0
        
        return {
            'check': '空值验证',
            'passed': passed,
            'message': "关键列中没有空值" if passed else f"以下列包含空值: {null_counts[null_columns].to_dict()}",
            'null_counts': null_counts.to_dict(),
            'null_columns': null_columns
        }
    
    def validate_image_paths_exist(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """
        验证图像文件路径是否存在（采样检查）。
        
        参数：
            df: DataFrame 数据
            sample_size: 检查的样本大小
            
        返回：
            Dict: 验证结果字典
        """
        missing_files = []
        check_count = min(sample_size, len(df))
        
        for idx, row in df.head(check_count).iterrows():
            image_path = Path(row['image_path'])
            if not image_path.exists():
                missing_files.append(str(image_path))
        
        passed = len(missing_files) == 0
        
        return {
            'check': '图像文件路径验证',
            'passed': passed,
            'message': f"所有图像文件路径都存在 (检查了 {check_count} 个)" if passed else f"发现 {len(missing_files)} 个不存在的图像文件",
            'checked_count': check_count,
            'missing_count': len(missing_files),
            'missing_files': missing_files[:5]  # 只返回前5个
        }
    
    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证列的数据类型。
        
        参数：
            df: DataFrame 数据
            
        返回：
            Dict: 验证结果字典
        """
        expected_types = {
            'image_path': 'object (string)',
            'label': 'object (string)',
            'split': 'object (string)'
        }
        
        type_issues = []
        for col in self.required_columns:
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != 'object':
                    type_issues.append(f"{col}: 期望 object, 实际 {actual_type}")
        
        passed = len(type_issues) == 0
        
        return {
            'check': '数据类型验证',
            'passed': passed,
            'message': f"所有列的数据类型正确" if passed else f"数据类型问题: {type_issues}",
            'expected_types': expected_types,
            'actual_types': {col: str(df[col].dtype) for col in self.required_columns if col in df.columns},
            'issues': type_issues
        }
    
    def validate_split_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据集划分比例（train/val/test）。
        
        参数：
            df: DataFrame 数据
            
        返回：
            Dict: 验证结果字典
        """
        valid_splits = {'train', 'val', 'test'}
        current_splits = set(df['split'].unique())
        
        # 检查是否有无效的split值
        invalid_splits = current_splits - valid_splits
        passed = len(invalid_splits) == 0
        
        # 计算比例
        split_counts = df['split'].value_counts().to_dict()
        total = len(df)
        split_percentages = {
            split: f"{(count / total) * 100:.1f}%"
            for split, count in split_counts.items()
        }
        
        return {
            'check': '数据集划分验证',
            'passed': passed,
            'message': f"数据集划分有效: {split_counts}" if passed else f"发现无效的split值: {invalid_splits}",
            'valid_splits': list(valid_splits),
            'current_splits': list(current_splits),
            'split_counts': split_counts,
            'split_percentages': split_percentages
        }
    
    def generate_validation_report(self, results: Dict[str, Dict]) -> Path:
        """
        生成验证报告。
        
        参数：
            results: 验证结果字典
            
        返回：
            Path: 报告文件路径
        """
        report_path = self.output_dir / "validation_report.json"
        
        # 添加验证时间
        results['validation_timestamp'] = pd.Timestamp.now().isoformat()
        
        # 计算摘要
        all_checks = [k for k in results.keys() if k != 'validation_timestamp']
        passed_checks = sum(1 for k in all_checks if results[k].get('passed', False))
        
        results['summary'] = {
            'total_checks': len(all_checks),
            'passed': passed_checks,
            'failed': len(all_checks) - passed_checks,
            'all_passed': passed_checks == len(all_checks)
        }
        
        # 保存为 JSON
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"验证报告已保存到 {report_path}")
        
        return report_path
    
    def run_validation(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        运行所有验证检查。
        
        参数：
            df: DataFrame 数据（如果为None，则从data_path读取）
            
        返回：
            Dict: 验证结果
        """
        logger.info(f"开始验证数据: {self.data_path}")
        
        # 读取数据
        if df is None:
            try:
                df = pd.read_csv(self.data_path)
                logger.info(f"成功读取数据，共 {len(df)} 条记录")
            except Exception as e:
                logger.error(f"读取数据失败: {e}")
                return {'error': str(e), 'all_passed': False}
        
        # 运行所有验证检查
        results = {}
        
        results['record_count'] = self.validate_record_count(df)
        results['required_columns'] = self.validate_required_columns(df)
        results['label_values'] = self.validate_label_values(df)
        results['no_null_values'] = self.validate_no_null_values(df)
        results['image_paths_exist'] = self.validate_image_paths_exist(df)
        results['data_types'] = self.validate_data_types(df)
        results['split_distribution'] = self.validate_split_distribution(df)
        
        # 生成验证报告
        report_path = self.generate_validation_report(results)
        
        # 打印验证结果
        self._print_validation_results(results)
        
        # 检查是否全部通过
        # 过滤掉非字典类型的键（如 'all_passed', 'validation_timestamp' 等）
        validation_results = {k: v for k, v in results.items() 
                             if isinstance(v, dict) and 'passed' in v}
        all_passed = all(v.get('passed', False) for v in validation_results.values())
        
        if all_passed:
            logger.info("✅ 数据验证全部通过！")
        else:
            logger.warning("❌ 数据验证有失败项，请检查报告")
        
        results['all_passed'] = all_passed
        
        return results
    
    def _print_validation_results(self, results: Dict[str, Dict]) -> None:
        """打印验证结果摘要。"""
        print("\n" + "=" * 60)
        print("数据验证结果")
        print("=" * 60)
        
        for check_name, result in results.items():
            if isinstance(result, dict) and 'check' in result:
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                print(f"\n{check_name}: {status}")
                print(f"  {result.get('message', '')}")
        
        print("\n" + "=" * 60)
        print(f"摘要: {results.get('summary', {})}")
        print("=" * 60)


# 示例用法
if __name__ == "__main__":
    """
    DataValidator 类的示例用法。
    """
    # 配置
    config = {
        'data_path': 'data/raw/cifar-10/dataset.csv',
        'output_dir': 'data/validation'
    }
    
    # 初始化验证器
    validator = DataValidator(config)
    
    # 运行验证
    results = validator.run_validation()
    
    # 打印结果
    print(f"\n验证结果: {'全部通过' if results.get('all_passed') else '有失败项'}")
