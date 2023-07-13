import numpy as np
import math

# 1.数据集
data_x = np.array([
    [0.697,0.460],
    [0.774,0.376],
    [0.634,0.264],
    [0.608,0.318],
    [0.556,0.215],
    [0.403,0.237],
    [0.481,0.149],
    [0.437,0.211],
    [0.666,0.091],
    [0.243,0.267],
    [0.245,0.057],
    [0.343,0.099],
    [0.639,0.161],
    [0.657,0.198],
    [0.360,0.370],
    [0.593,0.042],
    [0.719,0.103]
])
data_y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

# 迭代计算
steps = 100
beta = np.zeros((3,1))
x = np.mat(np.c_[data_x, np.ones(len(data_x))]).T
y = np.mat(data_y)
print(beta)
print(x)
print(y)
print(x[:,2])
print(y[0,2])
def p1(x,beta):
    return math.exp(beta.T*x)/(1+math.exp(beta.T*x))
# print(x.shape[1])
for step in range(steps):
    # 计算一阶导
    one_level = np.zeros((3,1))
    for i in range(x.shape[1]):
        one_level = one_level-x[:,i]*(y[0,i]-p1(x[:,i],beta))
# #     # 计算二阶导