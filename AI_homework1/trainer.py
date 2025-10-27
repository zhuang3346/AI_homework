import numpy as np
import time
from linear_regression import LinearRegression
import config
from logger import log_print

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model):
        self.model = model
        self.train_time = None
        
    def train(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            
        Returns:
            loss_history: 训练损失历史
        """
        log_print("开始训练模型...")
        start_time = time.time()
        
        # 训练模型
        self.model.fit(
            X_train, y_train,
            batch_size=config.Config.BATCH_SIZE,
            epochs=config.Config.EPOCHS,
            verbose=True
        )
        
        self.train_time = time.time() - start_time
        log_print(f"训练完成，耗时: {self.train_time:.2f}秒")
        
        return self.model.loss_history
    
