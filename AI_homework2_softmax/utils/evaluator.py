import torch
import numpy as np
from .trainer import evaluate
from .plot import plot_confusion_matrix


def _collect_predictions(model, loader, device):
    """在给定数据加载器上收集预测与真实标签。"""
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0) if y_true else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if y_pred else np.array([])
    return y_true, y_pred


def test_model(model, test_loader, device, model_path=None, class_names=None, save_path='confusion_matrix.png'):
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))  # 显式启用安全模式

    # 收集预测
    y_true, y_pred = _collect_predictions(model, test_loader, device)
    if y_true.size == 0:
        print('Test set is empty, skip evaluation.')
        return 0.0

    # 计算准确率
    acc = float((y_true == y_pred).sum() / y_true.size)
    print(f'Test Accuracy: {acc:.4f}')

    # 获取类别名称（优先使用传入的）
    if class_names is None:
        ds = getattr(test_loader, 'dataset', None)
        if ds is not None:
            if hasattr(ds, 'classes'):
                class_names = ds.classes
            elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'classes'):
                class_names = ds.dataset.classes

    # 绘制并保存混淆矩阵
    try:
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, normalize=True, save_path=save_path, show=True)
        print(f'Confusion matrix saved to: {save_path}')
    except Exception as e:
        print(f'Failed to plot confusion matrix: {e}')

    return acc
