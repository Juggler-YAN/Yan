# 
# 线性回归
# 2023.8.29
# 

import torch
from torch.utils import data
from torch import nn

# dataset
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# extract data in small batches
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__ == '__main__':

    # datset
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # args
    batch_size = 10
    num_epochs = 3
    learning_rate = 0.03

    # model
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    # loss
    loss = nn.MSELoss()
    # opti
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    # train
    for epoch in range(num_epochs):
        for X, y in load_array((features, labels), batch_size):
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    # res
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)