import torch
from torch import nn

# 切换存储和计算的设备
torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
# 计算可用的GPU数量
print(torch.cuda.device_count())

# 指定某一gpu是否可用
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# 返回所有可用gpu
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
print(try_gpu(), try_gpu(10), try_all_gpus())

# 查询张量所在的设备
# 存储在cpu上
x = torch.tensor([1, 2, 3])
print(x.device)
# 存储在gpu上
X = torch.ones(2, 3, device=try_gpu())
print(X.device)
# 存储在gpu1上
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y.device)

# 移动张量到gpu1上
Z = X.cuda(1)
print(X)
print(Y)
print(Z)
# print(X + Z)
print(Y + Z)
print(Z.cuda(1) is Z)

# 移动网络到gpu
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)


