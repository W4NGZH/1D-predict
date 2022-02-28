import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import *
import EMD
import pandas as pd



device=torch.device('cuda:0')
seed = 0
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子

datas=EMD.imfs
datas= np.reshape(datas, (-1, 1))

seq=3
lr_rate=0.001
wd_rate=0.001
step=2000
gamma_size = 0.99
Epochs=20000

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
x_train = x_train.reshape(-1, seq, 1) #将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.reshape(-1, 1)
x_train=torch.from_numpy(x_train).float().to(device)
y_train=torch.from_numpy(y_train).float().to(device)
x_test=(torch.tensor(dataX[train_size:])).reshape(-1,seq,1).float().to(device)
y_test=(torch.tensor(dataY[train_size:])).reshape(-1,1).float().to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        # 输入格式是1，输出隐藏层大小是32
        # self.linear = nn.Linear(6*seq, 1)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 6*seq)
        x = self.linear(x)
        return x

model_cp='D:/model'
model=Net()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate,weight_decay=wd_rate)
# optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate)
scheduler = lr_scheduler.StepLR(optimizer,step_size=step,gamma = gamma_size)
loss_func=nn.MSELoss()
lr_list=[]
losses = []
losses1 = []
for epoch in range(Epochs):
    model.train()
    output = model(x_train)
    loss = loss_func(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    if epoch % 2000 == 0 and epoch > 0:
        print("epoch:{}, loss:{}".format(epoch, loss/train_size))
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    losses.append(loss/train_size)
    model.eval()
    testout = model(x_test)
    loss1 = loss_func(testout,y_test)
    losses1.append(loss1/(len(dataX)-train_size))

# torch.save(model.state_dict(), '{0}/ZG110reslstm.pth'.format(model_cp))

model.eval()
test_loss = loss_func(model(x_test), y_test)/(len(dataX)-train_size)
print(test_loss)
test_pred = list(model(x_test).data.reshape(-1))
test_pred.insert(0,list(model(x_train).data.reshape(-1))[-1])
pred_data= list(datas[:3])+list(model(x_train).data.reshape(-1)) + list(model(x_test).data.reshape(-1))
# print(len(datas))
# print(len(pred_data))
plt.subplot(1,2,1)
plt.plot(datas, label="real")
# 原数据
plt.plot(pred_data, label="pred")
# 预测数据
plt.title('onelstm')
plt.xlabel('point')
plt.ylabel('displacement')
# plt.legend(loc='best')
# plt.plot(xlabel = 'num')
# plt.plot(ylabel =  'data')

plt.subplot(1,2,2)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus']=False
plt.plot(losses,color = 'y')
plt.plot(losses1,color = 'r')
plt.title('MSE')
plt.xlabel('epoch')
plt.ylabel = ('loss')
# plt.legend(loc='best')
# plt.plot(xlabel = 'num')
# plt.plot(ylabel =  'data')
# plt.plot(range(Epochs),lr_list,color = 'r')
# plt.title('learning rate')
# plt.xlabel('epoch')
# plt.ylabel('learning rate')
# plt.savefig('D:/data/2007-12baishuihe/picture/ZG93onelstm.png',dpi=600)
plt.show()