#
# 权重衰减（缓解过拟合）
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
def train(lambd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay':lambd},
        {"params":net[0].bias}], lr=lr)
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            print(f'轮次{epoch + 1}的训练误差和测试误差为{evaluate_loss(net, train_iter, loss)}和{evaluate_loss(net, test_iter, loss)}')
        train_loss.append(evaluate_loss(net, train_iter, loss))
        test_loss.append(evaluate_loss(net, test_iter, loss))
    print('w的L2范数：', net[0].weight.norm().item())
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