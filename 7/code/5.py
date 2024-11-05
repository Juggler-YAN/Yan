#
# 多层感知机（从零实现）
# 2023.8.31
#

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torch import nn
from torchvision import transforms

# dataset
def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

# model
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)

# accuracy
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    acc_nums = 0
    nums = 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_nums = acc_nums + accuracy(net(X), y)
            nums = nums + y.numel()
    return acc_nums / nums

if __name__ == '__main__':
    
    # dataset
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    
    lr = 0.1
    num_epochs = 10

    # 
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    # loss
    loss = nn.CrossEntropyLoss(reduction='none')
    # opti
    updater = torch.optim.SGD(params, lr=lr)

    # train
    train_loss_arr = [0.0] * num_epochs
    train_acc_arr = [0.0] * num_epochs
    test_acc_arr = [0.0] * num_epochs
    for epoch in range(num_epochs):
        if isinstance(net, torch.nn.Module):
            net.train()
        train_loss = 0
        train_acc = 0
        train_nums = 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            train_loss = train_loss + float(l.sum())
            train_acc =  train_acc + accuracy(y_hat, y)
            train_nums = train_nums + y.numel()
        train_metrics = [train_loss / train_nums, train_acc / train_nums]
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'轮次{epoch + 1}的训练损失和训练精度为{train_metrics[0]}和{train_metrics[1]}')
        print(f'轮次{epoch + 1}的测试精度为{test_acc}')
        train_loss_arr[epoch] = train_metrics[0]
        train_acc_arr[epoch] = train_metrics[1]
        test_acc_arr[epoch] = test_acc

    x = range(num_epochs)
    plt.plot(x, train_loss_arr, x, train_acc_arr, x, test_acc_arr)
    plt.show()