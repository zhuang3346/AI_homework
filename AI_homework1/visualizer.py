import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import config

class ResultVisualizer:
    """结果可视化类"""
    
    def __init__(self, save_dir=config.Config.SAVE_DIR):
        self.save_dir = save_dir
        # 设置中文字体（如果需要）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_loss_history(self, loss_history, title="训练损失曲线"):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('训练轮数')
        plt.ylabel('损失 (MSE)')
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'training_loss.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filepath
    
    def plot_predictions_vs_actual(self, y_true, y_pred, title="预测值 vs 真实值"):
        """绘制预测值与真实值的对比图"""
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.subplot(2, 1, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # 绘制理想线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(2, 1, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'predictions_vs_actual.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filepath
    
    def plot_feature_importance(self, feature_names, weights, title="特征重要性"):
        """绘制特征重要性图"""
        plt.figure(figsize=(10, 6))
        
        # 按绝对值排序
        indices = np.argsort(np.abs(weights))
        sorted_weights = weights[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        colors = ['red' if w < 0 else 'blue' for w in sorted_weights]
        
        plt.barh(range(len(sorted_weights)), sorted_weights, color=colors, alpha=0.7)
        plt.yticks(range(len(sorted_weights)), sorted_features)
        plt.xlabel('权重值')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filepath
    
    def plot_data_distribution(self, data, columns=None):
        """绘制数据分布图"""
        if columns is None:
            columns = data.columns[:4]  # 只显示前4个特征
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        for i, col in enumerate(columns):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{col} 分布')
            plt.xlabel(col)
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'data_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filepath
    
    def plot_correlation_matrix(self, data, columns=None):
        """绘制特征相关性矩阵"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        corr_matrix = data[columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性矩阵')
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'correlation_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filepath