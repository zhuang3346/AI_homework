import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
from logger import log_print

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path):
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            data: 加载的数据
        """
        try:
            self.data = pd.read_csv(file_path)
            log_print(f"数据加载成功，形状: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            log_print(f"文件 {file_path} 未找到，请检查数据集路径！")
            raise
    
    
    def preprocess_data(self):
        """数据预处理"""
        if self.data is None:
            raise ValueError("请先加载数据")
        # 检查缺失值
        if self.data.isnull().sum().sum() > 0:
            log_print("发现缺失值，进行填充...")
            self.data = self.data.fillna(self.data.mean())
        # 选择自变量和目标变量
        X = self.data[config.Config.FEATURE_COLUMNS].values
        y = self.data[config.Config.TARGET_COLUMN].values
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y,
            test_size=config.Config.TEST_SIZE,
            random_state=config.Config.RANDOM_STATE
        )
        log_print(f"训练集大小: {self.X_train.shape}")
        log_print(f"测试集大小: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_info(self):
        """获取数据基本信息"""
        if self.data is None:
            return "数据尚未加载"
        
        info = {
            '数据形状': self.data.shape,
            '特征列': list(self.data.columns),
            '数据类型': self.data.dtypes.to_dict(),
            '缺失值统计': self.data.isnull().sum().to_dict(),
            '基本统计': self.data.describe()
        }
        
        return info