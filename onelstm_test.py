import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import *
import bazimen_EMD
import pandas as pd



device=torch.device('cuda:0')
datas=bazimen_EMD.imfs
datas= np.reshape(datas, (-1, 1))


seq=2
def creat_dataset(dataset,seqnum):
    data_x=[]
    data_y=[]
    for i in range(len(dataset)-seqnum):
        data_x.append(dataset[i:i+seqnum])
        data_y.append(dataset[i+seqnum])
    return  np.asarray(data_x),np.asarray(data_y)
dataX,dataY=creat_dataset(datas,seq)
train_size=int(len(dataX)*0.8)
x_train=dataX[:train_size]
y_train=dataY[:train_size]
print(dataX)
x_train = x_train.reshape(-1, seq, 1) #将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.reshape(-1, 1)
x_train=torch.from_numpy(x_train).float().to(device)
y_train=torch.from_numpy(y_train).float().to(device)
x_test=(torch.tensor(dataX[train_size:])).reshape(-1,seq,1).float().to(device)
y_test=(torch.tensor(dataY[train_size:])).reshape(-1,1).float().to(device)


model_file='D:/model/ZG110lstm2.pth'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=12, num_layers=1, batch_first=True)
        # 输入格式是1，输出隐藏层大小是32
        self.linear = nn.Linear(12*seq, 1)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 12*seq)
        x = self.linear(x)
        return x

model=Net()
model=model.to(device)
model.load_state_dict(torch.load(model_file))
test_pred = list(model(x_test).data.reshape(-1))

train_pred = model(x_train).data.reshape(-1)
pred_data=list(datas[:2])+list(model(x_train).data.reshape(-1)) + list(model(x_test).data.reshape(-1))

plt.plot(datas[train_size:], label="real")
# 原数据
plt.plot(pred_data[train_size:], label="pred")
# 预测数据
plt.title('onelstm')
plt.xlabel('point')
plt.ylabel('displacement')
# plt.savefig('D:/picture/ZG118/onelstm2.png')
plt.show()

