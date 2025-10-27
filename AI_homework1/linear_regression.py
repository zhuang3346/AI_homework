import numpy as np
import matplotlib.pyplot as plt
from logger import log_print

class LinearRegression:
    """线性回归模型"""
    
    def __init__(self, learning_rate=0.01):
        """
        初始化线性回归模型
        
        Args:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def initialize_parameters(self, n_features):
        """初始化模型参数"""
        # 权重初始化为小随机数，偏置初始化为0
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
    def predict(self, X):
        """
        预测函数
        
        Args:
            X: 输入特征 (n_samples, n_features)
            
        Returns:
            y_pred: 预测值 (n_samples,)
        """
        if self.weights is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        """
        计算均方误差损失
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            loss: 均方误差
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def compute_gradients(self, X_batch, y_batch, y_pred):
        """
        计算梯度
        
        Args:
            X_batch: 批量特征
            y_batch: 批量真实值
            y_pred: 批量预测值
            
        Returns:
            dw: 权重梯度
            db: 偏置梯度
        """
        batch_size = X_batch.shape[0]
        
        # 计算误差
        error = y_pred - y_batch
        
        # 计算梯度
        dw = (2 / batch_size) * np.dot(X_batch.T, error)
        db = (2 / batch_size) * np.sum(error)
        
        return dw, db
    
    def update_parameters(self, dw, db):
        """使用梯度下降更新参数"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y, batch_size=32, epochs=1000, verbose=True):
        """
        使用小批量随机梯度下降训练模型
        
        Args:
            X: 训练特征
            y: 训练目标
            batch_size: 批量大小
            epochs: 训练轮数
            verbose: 是否显示训练信息
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.initialize_parameters(n_features)
        
        # 训练循环
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            batch_count = 0
            
            # 小批量训练
            for i in range(0, n_samples, batch_size):
                # 获取当前批次
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # 前向传播
                y_pred = self.predict(X_batch)
                
                # 计算损失
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                batch_count += 1
                
                # 反向传播计算梯度
                dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
                
                # 更新参数
                self.update_parameters(dw, db)
            
            # 记录平均损失
            avg_epoch_loss = epoch_loss / batch_count
            self.loss_history.append(avg_epoch_loss)

            # 每10轮打印一次损失
            if verbose and epoch % 10 == 0:
                    log_print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
    
    def get_parameters(self):
        """获取模型参数"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias,
            'learning_rate': self.learning_rate
        }
    
    def set_parameters(self, weights, bias):
        """设置模型参数"""
        self.weights = weights.copy()
        self.bias = bias