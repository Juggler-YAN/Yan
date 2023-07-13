import numpy as np
import matplotlib.pyplot as plt
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

def p1(x, beta):
    return math.exp(beta.T*x)/(1+math.exp(beta.T*x))

# 迭代计算
steps = 100
beta = np.mat([0.0]*3).T
data_x = np.mat(np.c_[data_x, np.ones(len(data_x))])
for step in range(steps):
    one_level = np.zeros((3,1))
    for i in range(len(data_x)):
        one_level = one_level-data_x[i].T*(data_y[i]-p1(data_x[i].T, beta))
    tow_level = np.zeros((3,1))
    for i in range(len(data_x)):
        tow_level = tow_level+data_x[i].T*data_x[i]*p1(data_x[i].T, beta)*(1-p1(data_x[i].T, beta))
    prev_beta = beta
    beta = prev_beta-tow_level.I*one_level
    if np.linalg.norm(prev_beta.T-beta.T) < 1e-6:
        print(step)
        break
print(beta)
print(prev_beta)

# 绘图
for i in range(len(data_x)):
    if data_y[i] == 0:
        plt.plot(data_x[i,0], data_x[i,1], 'ob')
    else:
        plt.plot(data_x[i,0], data_x[i,1], 'or')
w0 = beta[0,0]
w1 = beta[1,0]
b = beta[2,0]
plt.plot([-b/w0,0],[0,-b/w1])
plt.savefig("1.jpg")
plt.show()