import torch

class Config:
    data_root = "data"
    batch_size = 32 #批量大小
    num_epochs = 100 #训练轮数
    lr = 0.001
    train_ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_best = True
    model_path = 'best_model.pth'
    image_path = "test.jpg"
    # 四分类（苹果叶片）：按 ImageFolder 的字母序类别映射
    # 默认目录名应为：
    # ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
    num_classes = 4
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy'
    ]