#
# 多项式回归
# 欠拟合和过拟合
# 2023.8.31
#

import math
import numpy as np
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

# 生成数据集
max_degree = 20                                                              # 多项式最大阶数
n_train, n_test = 100, 100                                                   # 训练集和测试集大小
true_w = np.zeros(max_degree)                                                # 多项式权重
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
features = np.random.normal(size=(n_train + n_test, 1))                      # 多项式输入
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))     # 多项式输出
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)                                 # gamma函数是阶乘函数的泛化
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
# NumPy中ndarray类型转换为tensor类型
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

# 加载数据
def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

# 评估损失
def evaluate_loss(net, data_iter, loss):
    loss_all = 0
    nums = 0
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        loss_all = loss_all + l.sum()
        nums = nums + l.numel()
    return loss_all / nums

# 训练一个周期
def train_epoch(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
# 训练
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1,1)),batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1,1)),batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, trainer)
        train_loss.append(evaluate_loss(net, train_iter, loss))
        test_loss.append(evaluate_loss(net, test_iter, loss))
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'轮次{epoch + 1}的训练误差和测试误差为{evaluate_loss(net, train_iter, loss)}和{evaluate_loss(net, test_iter, loss)}')
    print('weight:', net[0].weight.data.numpy())
    return train_loss, test_loss

# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# va1, va2 = train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
# x = range(len(va1))
# plt.plot(x, torch.tensor(va1), x, torch.tensor(va2))

# 从多项式特征中选择前2个维度，即1和x
# 欠拟合
# va1, va2 = train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
# x = range(len(va1))
# plt.plot(x, torch.tensor(va1), x, torch.tensor(va2))

# 从多项式特征中选取所有维度
# 过拟合
va1, va2 = train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)
x = range(len(va1))
plt.plot(x, torch.tensor(va1), x, torch.tensor(va2))