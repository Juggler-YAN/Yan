#
# LeNet
# 2024.11.4
#

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torch import nn
from torchvision import transforms

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# dataset
def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

# accuracy
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    acc_nums = 0
    nums = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            acc_nums = acc_nums + accuracy(net(X), y)
            nums = nums + y.numel()
    return acc_nums / nums

if __name__ == '__main__':
    
    # dataset
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    learning_rate = 0.1
    num_epochs = 2

    # device
    device = try_gpu()
    # model
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    # loss
    loss = nn.CrossEntropyLoss()
    # opti
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

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
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            train_loss = train_loss + l.cpu().detach() * X.shape[0]
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
    plt.plot(x, train_loss_arr, label='train_loss')
    plt.plot(x, train_acc_arr, label='train_acc')
    plt.plot(x, test_acc_arr, label='test_acc')
    plt.legend()
    plt.show()
