#
# 权重衰减（缓解过拟合）（从零实现）
# 2023.9.1
# 

import torch
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt

# 生成数据集
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape([-1,1])
def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)

def linreg(X, w, b):
    return torch.matmul(X, w) + b
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
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
# L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            print(f'轮次{epoch + 1}的训练误差和测试误差为{evaluate_loss(net, train_iter, loss)}和{evaluate_loss(net, test_iter, loss)}')
        train_loss.append(evaluate_loss(net, train_iter, loss))
        test_loss.append(evaluate_loss(net, test_iter, loss))
    print('w的L2范数是：', torch.norm(w).item())
    return train_loss, test_loss

# 不带惩罚项
# va1, va2 = train(lambd=0)
# x = range(len(va1))
# plt.plot(x, torch.tensor(va1), x, torch.tensor(va2))
# plt.show()

# 带惩罚项
va1, va2 = train(lambd=3)
x = range(len(va1))
plt.plot(x, torch.tensor(va1), x, torch.tensor(va2))
plt.show()