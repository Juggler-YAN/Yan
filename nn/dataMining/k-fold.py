import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

if __name__ == '__main__':

    # args
    k = 5
    num_epochs = 2000
    lr = 0.1
    weight_decay = 0
    batch_size = 64

    # data
    train_data = pd.read_csv('./data/kaggle_house_pred_train.csv')
    test_data = pd.read_csv('./data/kaggle_house_pred_test.csv')
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # 标准化
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # 缺失值处理
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # 划分
    n_train = train_data.shape[0]
    train_features_data = [[float(item) for item in sublist] for sublist in all_features[:n_train].values]
    train_features = torch.tensor(train_features_data, dtype=torch.float32)
    test_features_data = [[float(item) for item in sublist] for sublist in all_features[n_train:].values]
    test_features = torch.tensor(test_features_data, dtype=torch.float32)
    train_labels_data = [[float(item) for item in sublist] for sublist in train_data.SalePrice.values.reshape(-1, 1)]
    train_labels = torch.tensor(train_labels_data, dtype=torch.float32)

    # net
    in_features = train_features.shape[1]
    net = nn.Sequential(nn.Linear(in_features,1))

    # loss
    loss = nn.MSELoss()

    # train —— K折寻找最优参数
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train, y_train, X_test, y_test = get_k_fold_data(k, i, train_features, train_labels)
        train_ls, valid_ls = [], []
        train_iter = load_array((X_train, y_train), batch_size)
        # opti
        optimizer = torch.optim.Adam(net.parameters(),
                                    lr = lr,
                                    weight_decay = weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
            train_ls.append(log_rmse(net, X_train, y_train))
            if y_test is not None:
                valid_ls.append(log_rmse(net, X_test, y_test))
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.plot(train_ls, label='train')
            plt.plot(valid_ls, label='test')
            plt.legend()
            plt.savefig("1.png")
        print(f'折{i + 1}，训练log rmse: {float(train_ls[-1]):f}, '
            f'验证log rmse: {float(valid_ls[-1]):f}')
    train_l, valid_l = train_l_sum / k, valid_l_sum / k
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
        f'平均验证log rmse: {float(valid_l):f}')

    # 最优参数训练
    train_ls = []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = lr,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
    print(f'训练log rmse：{float(train_ls[-1]):f}')

    # 预测
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
