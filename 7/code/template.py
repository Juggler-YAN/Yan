# 导入包以及设置随机种子
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 二、以类的方式定义超参数
class argparse():
    pass

args = argparse()
args.epochs, args.learning_rate, args.patience = [30, 0.001, 4]
args.hidden_size, args.input_size= [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]

# 三、定义自己的模型
class Your_model(nn.Module):
    def __init__(self):
        super(Your_model, self).__init__()
        pass
        
    def forward(self,x):
        pass
        return x

# 四、定义早停类(此步骤可以省略)
class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss

# 五、定义自己的数据集Dataset,DataLoader
class Dataset_name(Dataset):
    def __init__(self, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.__load_data__()

    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

    def __load_data__(self, csv_paths: list):
        pass
        print(
            "train_X.shape:{}\ntrain_Y.shape:{}\nvalid_X.shape:{}\nvalid_Y.shape:{}\n"
            .format(self.train_X.shape, self.train_Y.shape, self.valid_X.shape, self.valid_Y.shape))

train_dataset = Dataset_name(flag='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
valid_dataset = Dataset_name(flag='valid')
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)

# 六、实例化模型，设置loss，优化器等
model = Your_model().to(args.device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Your_model.parameters(),lr=args.learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

early_stopping = EarlyStopping(patience=args.patience,verbose=True)

# 七、开始训练以及调整lr
for epoch in range(args.epochs):
    Your_model.train()
    train_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(train_dataloader,0):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = Your_model(data_x)
        optimizer.zero_grad()
        loss = criterion(data_y,outputs)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx%(len(train_dataloader)//2)==0:
            print("epoch={}/{},{}/{}of train, loss={}".format(
                epoch, args.epochs, idx, len(train_dataloader),loss.item()))
    train_epochs_loss.append(np.average(train_epoch_loss))
    
    #=====================valid============================
    Your_model.eval()
    valid_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = Your_model(data_x)
        loss = criterion(data_y,outputs)
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    #==================early stopping======================
    early_stopping(valid_epochs_loss[-1],model=Your_model,path=r'c:\\your_model_to_save')
    if early_stopping.early_stop:
        print("Early stopping")
        break
    #====================adjust lr========================
    lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

# 八、绘图
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()

#九、预测
# 此处可定义一个预测集的Dataloader。也可以直接将你的预测数据reshape,添加batch_size=1
Your_model.eval()
predict = Your_model(data)

# 1.导入包
# 2.设置随机数种子
# 3.定义超参数
# 4.定义数据集
# 5.定义模型
# 6.实例化模型
# 7.训练
# 8.绘图
# 9.预测