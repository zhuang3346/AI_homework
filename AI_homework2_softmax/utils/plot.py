import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

def plot(loss_history: np.ndarray, acc_history: np.ndarray, epoch):

    epochs = np.arange(1, epoch + 1)  # 创建一个从1到epoch的数组

    # 创建画布和子图（1行2列）
    plt.figure(figsize=(12, 5))  # 画布尺寸（宽12英寸，高5英寸）

    # ---------------------------
    # 子图1：Loss曲线
    # ---------------------------
    plt.subplot(1, 2, 1)  # (行数, 列数, 子图索引)
    plt.plot(epochs, loss_history, 'r-', label='Training Loss', linewidth=2)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    plt.legend(fontsize=10)

    # ---------------------------
    # 子图2：Accuracy曲线
    # ---------------------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_history, 'b-', label='Accuracy', linewidth=2)
    plt.title('Validation Accuracy Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    # 调整子图间距
    plt.tight_layout(pad=3)  # 防止标题重叠

    # 保存图片（可选）
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    cmap: str = 'Blues',
    title: Optional[str] = None,
    save_path: Optional[str] = 'confusion_matrix.png',
    show: bool = True,
):
    """
    绘制并保存混淆矩阵。

    参数
    - y_true: 形状为 (N,) 的真实标签数组（int）
    - y_pred: 形状为 (N,) 的预测标签数组（int）
    - class_names: 类别名称列表，用于坐标轴刻度；若为 None，则使用类别索引
    - normalize: 是否按行归一化（每一类的样本数为1），便于观察召回率
    - cmap: 颜色映射
    - title: 标题，默认根据 normalize 自动生成
    - save_path: 保存路径，默认为当前目录下 confusion_matrix.png；若为 None 则不保存
    - show: 是否显示图像
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError('y_true 和 y_pred 必须为一维数组')
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError('y_true 与 y_pred 长度不一致')

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        # 避免标签大于类别数导致下标越界
        num_classes = max(num_classes, len(class_names))

    # 计算混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # 归一化（按行：每个实际类别的比例）
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        fmt = '.2f'
        default_title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        default_title = 'Confusion Matrix'

    if title is None:
        title = default_title

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # 坐标轴与标签
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='Actual', xlabel='Predicted',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # 在每个格子上标注数值
    thresh = (cm_display.max() + cm_display.min()) / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_display[i, j]
            ax.text(j, i, format(value, fmt),
                    ha='center', va='center',
                    color='white' if value > thresh else 'black')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
