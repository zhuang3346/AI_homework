from torchvision import transforms, datasets
from torch.utils.data import random_split
import os
import torch


def get_train_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform


def get_test_transforms():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return test_transform


def load_datasets(data_root, train_ratio=0.8):
    """
    加载数据集：
    - 优先使用 data_root/train, data_root/valid, data_root/test 三段式划分；
    - 若没有 valid 目录，则从 train 中按 train_ratio 划分训练/验证（同一类内随机切分）。
    训练集使用数据增强，验证/测试使用中心裁剪。
    返回：(train_dataset, val_dataset, test_dataset, classes)
    """

    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    train_dir = os.path.join(data_root, 'train')
    valid_dir = os.path.join(data_root, 'valid')
    test_dir = os.path.join(data_root, 'test')

    # 测试集（如不存在则返回空数据集占位）
    test_data = datasets.ImageFolder(test_dir, test_transform) if os.path.isdir(test_dir) else None

    if os.path.isdir(valid_dir):
        # 明确的三段式划分
        train_data = datasets.ImageFolder(train_dir, train_transform)
        val_data = datasets.ImageFolder(valid_dir, test_transform)
        classes = train_data.classes
    else:
        # 从 train 中切分出验证集（注意：两者共享同一底层数据，transform 相同）
        # 为了让验证集使用 test_transform，这里采用两次实例化的策略
        full_train_for_split = datasets.ImageFolder(train_dir, train_transform)
        classes = full_train_for_split.classes

        train_size = int(len(full_train_for_split) * train_ratio)
        val_size = len(full_train_for_split) - train_size
        train_subset, val_subset = random_split(full_train_for_split, [train_size, val_size])

        # 构造两个独立的 ImageFolder 来分别应用不同的 transform，然后通过索引子集进行子采样
        full_train_for_train = datasets.ImageFolder(train_dir, train_transform)
        full_train_for_val = datasets.ImageFolder(train_dir, test_transform)

        # 利用 Subset 的 indices 选择样本
        train_data = torch.utils.data.Subset(full_train_for_train, train_subset.indices)
        val_data = torch.utils.data.Subset(full_train_for_val, val_subset.indices)

    return train_data, val_data, test_data, classes