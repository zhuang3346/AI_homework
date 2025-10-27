# 🍏 植物病害四分类项目说明

---

## 1. 使用方法

- **训练**：
	- 运行 `train.py`，使用训练集和验证集进行模型训练。
- **测试**：
	- 运行 `test.py`，在测试集上评估模型准确率，并自动绘制混淆矩阵到 `confusion_matrix.png`。
- **单张/批量预测**：
	- 将需要预测的图片放入 `need_predict` 目录，运行 `predict.py`，预测结果图片会自动生成在 `predicted` 文件夹中。

---

## 2. 模型结构

- 当前默认模型为两层卷积神经网络（见 `models/cnn_model.py`）。
- 处理多分类任务（本项目为苹果叶片四分类：黑星病、黑腐病、赤星病、健康）。

---

## 3. 运行日志示例

```
(base) PS E:\vscode_project\AI_homework2_softmax> D:/anaconda/Scripts/activate
(base) PS E:\vscode_project\AI_homework2_softmax> conda activate pytorch
(pytorch) PS E:\vscode_project\AI_homework2_softmax> & D:/anaconda/envs/pytorch/python.exe e:/vscode_project/AI_homework2_softmax/train.py
Using device: cuda
Epoch 1/100 | Loss: 0.752 | Val Acc: 0.841
Saved best model with acc: 0.841
Epoch 2/100 | Loss: 0.471 | Val Acc: 0.892
Saved best model with acc: 0.892
Epoch 3/100 | Loss: 0.389 | Val Acc: 0.931
Saved best model with acc: 0.931
Epoch 4/100 | Loss: 0.379 | Val Acc: 0.886
Epoch 5/100 | Loss: 0.316 | Val Acc: 0.961
Saved best model with acc: 0.961
Epoch 6/100 | Loss: 0.304 | Val Acc: 0.941
Epoch 7/100 | Loss: 0.308 | Val Acc: 0.893
Epoch 8/100 | Loss: 0.293 | Val Acc: 0.954
Epoch 9/100 | Loss: 0.267 | Val Acc: 0.962
Saved best model with acc: 0.962
Epoch 10/100 | Loss: 0.245 | Val Acc: 0.962
Epoch 11/100 | Loss: 0.261 | Val Acc: 0.963
Saved best model with acc: 0.963
Epoch 12/100 | Loss: 0.238 | Val Acc: 0.962
Epoch 13/100 | Loss: 0.240 | Val Acc: 0.979
Saved best model with acc: 0.979
Epoch 14/100 | Loss: 0.224 | Val Acc: 0.961
Epoch 15/100 | Loss: 0.237 | Val Acc: 0.972
Epoch 16/100 | Loss: 0.218 | Val Acc: 0.971
Epoch 17/100 | Loss: 0.217 | Val Acc: 0.974
Epoch 18/100 | Loss: 0.214 | Val Acc: 0.985
Saved best model with acc: 0.985
Epoch 19/100 | Loss: 0.230 | Val Acc: 0.975
Epoch 20/100 | Loss: 0.214 | Val Acc: 0.986
Saved best model with acc: 0.986
Epoch 21/100 | Loss: 0.209 | Val Acc: 0.987
Saved best model with acc: 0.987
Epoch 22/100 | Loss: 0.206 | Val Acc: 0.982
Epoch 23/100 | Loss: 0.209 | Val Acc: 0.974
Epoch 24/100 | Loss: 0.187 | Val Acc: 0.981
Epoch 25/100 | Loss: 0.176 | Val Acc: 0.979
Epoch 26/100 | Loss: 0.199 | Val Acc: 0.980
Epoch 27/100 | Loss: 0.205 | Val Acc: 0.982
Epoch 28/100 | Loss: 0.199 | Val Acc: 0.980
Epoch 29/100 | Loss: 0.203 | Val Acc: 0.982
Epoch 30/100 | Loss: 0.191 | Val Acc: 0.989
Saved best model with acc: 0.989
Epoch 31/100 | Loss: 0.176 | Val Acc: 0.978
Epoch 32/100 | Loss: 0.185 | Val Acc: 0.970
Epoch 33/100 | Loss: 0.180 | Val Acc: 0.986
Epoch 34/100 | Loss: 0.176 | Val Acc: 0.980
Epoch 35/100 | Loss: 0.166 | Val Acc: 0.979
Epoch 36/100 | Loss: 0.184 | Val Acc: 0.991
Saved best model with acc: 0.991
Epoch 37/100 | Loss: 0.169 | Val Acc: 0.986
Epoch 38/100 | Loss: 0.171 | Val Acc: 0.987
Epoch 39/100 | Loss: 0.173 | Val Acc: 0.985
Epoch 40/100 | Loss: 0.160 | Val Acc: 0.982
Epoch 41/100 | Loss: 0.176 | Val Acc: 0.989
Epoch 42/100 | Loss: 0.149 | Val Acc: 0.996
Saved best model with acc: 0.996
Epoch 43/100 | Loss: 0.229 | Val Acc: 0.975
Epoch 44/100 | Loss: 0.159 | Val Acc: 0.992
Epoch 45/100 | Loss: 0.148 | Val Acc: 0.989
Epoch 46/100 | Loss: 0.166 | Val Acc: 0.990
Epoch 47/100 | Loss: 0.164 | Val Acc: 0.974
Epoch 48/100 | Loss: 0.164 | Val Acc: 0.984
Epoch 49/100 | Loss: 0.154 | Val Acc: 0.989
Epoch 50/100 | Loss: 0.150 | Val Acc: 0.987
Epoch 51/100 | Loss: 0.174 | Val Acc: 0.985
Epoch 52/100 | Loss: 0.156 | Val Acc: 0.991
Epoch 53/100 | Loss: 0.147 | Val Acc: 0.990
Epoch 54/100 | Loss: 0.171 | Val Acc: 0.983
Epoch 55/100 | Loss: 0.137 | Val Acc: 0.992
Epoch 56/100 | Loss: 0.183 | Val Acc: 0.987
Epoch 57/100 | Loss: 0.141 | Val Acc: 0.983
Epoch 58/100 | Loss: 0.157 | Val Acc: 0.983
Epoch 59/100 | Loss: 0.152 | Val Acc: 0.955
Epoch 60/100 | Loss: 0.151 | Val Acc: 0.994
Epoch 61/100 | Loss: 0.150 | Val Acc: 0.984
Epoch 62/100 | Loss: 0.148 | Val Acc: 0.991
Epoch 63/100 | Loss: 0.147 | Val Acc: 0.986
Epoch 64/100 | Loss: 0.155 | Val Acc: 0.990
Epoch 65/100 | Loss: 0.134 | Val Acc: 0.992
Epoch 66/100 | Loss: 0.146 | Val Acc: 0.979
Epoch 67/100 | Loss: 0.127 | Val Acc: 0.987
Epoch 68/100 | Loss: 0.142 | Val Acc: 0.978
Epoch 69/100 | Loss: 0.141 | Val Acc: 0.992
Epoch 70/100 | Loss: 0.140 | Val Acc: 0.990
Epoch 71/100 | Loss: 0.155 | Val Acc: 0.992
Epoch 72/100 | Loss: 0.172 | Val Acc: 0.982
Epoch 73/100 | Loss: 0.138 | Val Acc: 0.989
Epoch 74/100 | Loss: 0.142 | Val Acc: 0.977
Epoch 75/100 | Loss: 0.134 | Val Acc: 0.991
Epoch 76/100 | Loss: 0.137 | Val Acc: 0.996
Epoch 77/100 | Loss: 0.148 | Val Acc: 0.984
Epoch 78/100 | Loss: 0.125 | Val Acc: 0.990
Epoch 79/100 | Loss: 0.119 | Val Acc: 0.983
Epoch 80/100 | Loss: 0.136 | Val Acc: 0.985
Epoch 81/100 | Loss: 0.126 | Val Acc: 0.984
Epoch 82/100 | Loss: 0.137 | Val Acc: 0.993
Epoch 83/100 | Loss: 0.141 | Val Acc: 0.990
Epoch 84/100 | Loss: 0.129 | Val Acc: 0.990
Epoch 85/100 | Loss: 0.119 | Val Acc: 0.992
Epoch 86/100 | Loss: 0.134 | Val Acc: 0.979
Epoch 87/100 | Loss: 0.122 | Val Acc: 0.993
Epoch 88/100 | Loss: 0.120 | Val Acc: 0.993
Epoch 89/100 | Loss: 0.135 | Val Acc: 0.983
Epoch 90/100 | Loss: 0.117 | Val Acc: 0.992
Epoch 91/100 | Loss: 0.115 | Val Acc: 0.992
Epoch 92/100 | Loss: 0.109 | Val Acc: 0.989
Epoch 93/100 | Loss: 0.138 | Val Acc: 0.986
Epoch 94/100 | Loss: 0.142 | Val Acc: 0.994
Epoch 95/100 | Loss: 0.110 | Val Acc: 0.994
Epoch 96/100 | Loss: 0.111 | Val Acc: 0.989
Epoch 97/100 | Loss: 0.115 | Val Acc: 0.992
Epoch 98/100 | Loss: 0.115 | Val Acc: 0.992
Epoch 99/100 | Loss: 0.104 | Val Acc: 0.987
Epoch 100/100 | Loss: 0.113 | Val Acc: 0.992

Training completed in 8839.1s
Best Validation Acc: 0.996
Memory Usage: +-271MB

(pytorch) PS E:\vscode_project\AI_homework2_softmax> & D:/anaconda/envs/pytorch/python.exe e:/vscode_project/AI_homework2_softmax/test.py
Test Accuracy: 1.0000
```