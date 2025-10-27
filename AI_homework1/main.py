import numpy as np
import pandas as pd
from data_loader import DataLoader
from linear_regression import LinearRegression
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
import config
from logger import log_print

def main():
    """主函数"""
    log_print("=" * 60, clear=True)
    log_print("植物生长速度预测系统 - 线性回归模型")
    log_print("=" * 60)
    
    # 1. 加载数据
    log_print("\n1. 数据加载与预处理")
    data_loader = DataLoader()
    data = data_loader.load_data(config.Config.DATA_PATH)
    
    # 显示数据信息
    info = data_loader.get_data_info()
    log_print(f"数据基本信息: {info['数据形状']}")
    log_print(f"特征列: {info['特征列']}")
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test = data_loader.preprocess_data()
    
    # 3. 创建模型
    log_print("\n2. 模型初始化")
    model = LinearRegression(learning_rate=config.Config.LEARNING_RATE)
    trainer = ModelTrainer(model)
    
    # 4. 训练模型
    log_print("\n3. 模型训练")
    loss_history = trainer.train(X_train, y_train)
    
    # 5. 模型预测
    log_print("\n4. 模型预测")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 6. 模型评估
    log_print("\n5. 模型评估")
    
    # 训练集评估
    train_evaluator = ModelEvaluator(y_train, y_pred_train)
    log_print("\n训练集评估:")
    train_evaluator.print_evaluation()
    
    # 测试集评估
    test_evaluator = ModelEvaluator(y_test, y_pred_test)
    log_print("\n测试集评估:")
    test_evaluator.print_evaluation()
    
    # 7. 可视化结果
    log_print("\n6. 结果可视化")
    visualizer = ResultVisualizer()
    
    # 绘制损失曲线
    visualizer.plot_loss_history(loss_history)
    
    # 绘制预测结果
    visualizer.plot_predictions_vs_actual(y_test, y_pred_test)
    
    # 绘制特征重要性
    feature_names = config.Config.FEATURE_COLUMNS
    weights = model.weights
    visualizer.plot_feature_importance(feature_names, weights)
    
    # 绘制数据分布
    visualizer.plot_data_distribution(data)
    visualizer.plot_correlation_matrix(data)
    
    # 8. 保存结果
    log_print("\n7. 保存结果")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '真实值': y_test,
        '预测值': y_pred_test,
        '残差': y_test - y_pred_test
    })
    
    results_path = f"{config.Config.SAVE_DIR}/prediction_results.csv"
    results_df.to_csv(results_path, index=False)
    log_print(f"预测结果已保存到: {results_path}")
    
    # 保存模型参数
    params = model.get_parameters()
    params_df = pd.DataFrame({
        '特征': feature_names + ['偏置项'],
        '权重': list(weights) + [model.bias]
    })
    
    params_path = f"{config.Config.SAVE_DIR}/model_parameters.csv"
    params_df.to_csv(params_path, index=False)
    log_print(f"模型参数已保存到: {params_path}")
    
    
    log_print("\n" + "=" * 60)
    log_print("程序执行完成！")
    log_print("=" * 60)

if __name__ == "__main__":
    main()