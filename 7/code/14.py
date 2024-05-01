import torch
from torch import nn

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# 未初始化
print(net[0].weight)
# 初始化
X = torch.rand(2, 20)
net(X)
print(net[0].weight.shape)

# 设置延后初始化
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# 未初始化
print(net[0].weight)
# 初始化
X = torch.rand(2, 20)
net(X)
print(net[0].weight.shape)