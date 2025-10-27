import os

class Config:
    """配置参数类"""
    
    # 数据路径配置
    DATA_PATH = "crop_growth_dataset.csv"
    SAVE_DIR = "results"
    
    # 模型参数
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 1000
    
    # 数据划分参数
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 特征列配置（根据实际数据集调整）
    FEATURE_COLUMNS = ['Temperature', 'Humidity', 'Soil_Moisture']
    TARGET_COLUMN = 'Growth'
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        if not os.path.exists(cls.SAVE_DIR):
            os.makedirs(cls.SAVE_DIR)

# 初始化目录
Config.create_directories()