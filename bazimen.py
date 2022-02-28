import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import *
import pandas as pd


df=pd.read_csv('D:/data/2007-12baishuihe/bazimen/bazimen.csv')
datas=np.array(df['ZG111'])
datas= np.reshape(datas, (-1, 1))

# datas=bazimen_EMD.imfs
# datas= np.reshape(datas, (-1, 1))

seed = 0
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
device=torch.device('cuda:0')

seq=2
lr_rate=0.01
wd_rate=0.001
step=1000
gamms_size = 0.95
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
x_train=dataX
y_train=dataY
x_train = x_train.reshape(-1,seq)
y_train = y_train.reshape(-1, 1)
x_train=torch.from_numpy(x_train).float().to(device)
y_train=torch.from_numpy(y_train).float().to(device)
x_test=(torch.tensor(dataX[train_size:])).reshape(-1,seq).float().to(device)
y_test=(torch.tensor(dataY[train_size:])).reshape(-1,1).float().to(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(seq, 15).to(device),
            torch.nn.Tanhshrink(),
            # torch.nn.Linear(10, 5).cuda(),
            # torch.nn.Tanhshrink(),
            torch.nn.Linear(15, 1).to(device),
        )
    def forward(self,x):
        x = self.net(x)
        return x

model_cp='D:/model'
model=Net()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate,weight_decay=wd_rate)
scheduler = lr_scheduler.StepLR(optimizer,step_size=step,gamma = gamms_size)
loss_func=nn.MSELoss()
lr_list=[]
losses = []
losses1 = []
for epoch in range(Epochs):
    output = model(x_train)
    loss = loss_func(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    if epoch % 2000 == 0 and epoch > 0:
        print("epoch:{}, loss:{}".format(epoch, loss/train_size))


torch.save(model.state_dict(), '{0}/ZG111.pth'.format(model_cp))

model.eval()

pred_data= list(datas[:seq])+list(model(x_train).data.reshape(-1))



plt.plot(datas, label="real")
plt.plot(pred_data, label="pred")
plt.title('ANN')
plt.xlabel('point')
plt.ylabel('displacement')

plt.show()