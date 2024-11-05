#
# 多层感知机
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

    # arg
    num_epochs = 10

    # model
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights);
    # loss
    loss = nn.CrossEntropyLoss(reduction='none')
    # opti
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

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
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
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
