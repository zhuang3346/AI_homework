import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from logger import log_print

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def mean_squared_error(self):
        """计算均方误差 (MSE)"""
        return mean_squared_error(self.y_true, self.y_pred)
    
    def root_mean_squared_error(self):
        """计算均方根误差 (RMSE)"""
        return np.sqrt(self.mean_squared_error())
    
    def r2_score(self):
        """计算R²分数"""
        return r2_score(self.y_true, self.y_pred)
    
    def evaluate_all(self):
        """计算所有评估指标"""
        mse = self.mean_squared_error()
        rmse = self.root_mean_squared_error()
        r2 = self.r2_score()
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2
        }
        
        return metrics
    
    def print_evaluation(self):
        """打印评估结果"""
        metrics = self.evaluate_all()
        
        log_print("\n" + "="*50)
        log_print("模型评估结果")
        log_print("="*50)
        for metric, value in metrics.items():
            log_print(f"{metric}: {value:.4f}")
        log_print("="*50)
