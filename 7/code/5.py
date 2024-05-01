#
# 多层感知机（从零实现）
# 2023.8.31
#

%matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn

# 生成数据集
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 定义模型
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)
# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]
# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 精度计算
# 定义预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    acc_nums = 0     # 正确预测数
    nums = 0         # 预测总数
    with torch.no_grad():
        for X, y in data_iter:
            acc_nums = acc_nums + accuracy(net(X), y)
            nums = nums + y.numel()
    return acc_nums / nums
# 训练
# 训练一个周期
def train_epoch(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    train_loss = 0         # 训练损失
    train_acc = 0          # 训练准确度
    train_nums = 0         # 训练样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        train_loss = train_loss + float(l.sum())
        train_acc =  train_acc + accuracy(y_hat, y)
        train_nums = train_nums + y.numel()
    # 返回训练损失和训练精度
    return train_loss / train_nums, train_acc / train_nums
# 训练
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    train_loss_arr = [0.0] * num_epochs
    train_acc_arr = [0.0] * num_epochs
    test_acc_arr = [0.0] * num_epochs
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'轮次{epoch + 1}的训练损失和训练精度为{train_metrics[0]}和{train_metrics[1]}')
        print(f'轮次{epoch + 1}的测试精度为{test_acc}')
        train_loss_arr[epoch] = train_metrics[0]
        train_acc_arr[epoch] = train_metrics[1]
        test_acc_arr[epoch] = test_acc
    return train_loss_arr, train_acc_arr, test_acc_arr
    
lr = 0.1
num_epochs = 10
updater = torch.optim.SGD(params, lr=lr)
va1, va2, va3 = train(net, train_iter, test_iter, loss, num_epochs, updater)

# x = range(10)
# plt.plot(x, va1, x, va2, x, va3)
# 测试
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
for X, y in test_iter:
    plt.imshow(X[0].reshape(28,28))
    print(text_labels[y[0].item()])
    print(text_labels[(net(X).argmax(axis=1))[0].item()])
    break