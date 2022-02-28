import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import *
import bazimen_EMD
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# datas=pd.read_csv('D:/111(1)(1).csv')
# datas=np.array(datas).reshape(-1,1)[:,0]

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
x_train = x_train.reshape(-1,seq)
y_train = y_train.reshape(-1, 1)
x_train=torch.from_numpy(x_train).float().to(device)
y_train=torch.from_numpy(y_train).float().to(device)
x_test=(torch.tensor(dataX[train_size:])).reshape(-1,seq).float().to(device)
y_test=(torch.tensor(dataY[train_size:])).reshape(-1,1).float().to(device)

model_file='D:/model/ZG110imfs.pth'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(seq, 6).to(device),
            torch.nn.Tanhshrink(),
            # torch.nn.Linear(10, 5).cuda(),
            # torch.nn.Tanhshrink(),
            torch.nn.Linear(6, 1).to(device),
        )
    def forward(self,x):
        x = self.net(x)
        return x

model=Net()
model=model.to(device)
model.load_state_dict(torch.load(model_file))

train_pred = model(x_train).data.reshape(-1)
test_pred = list(model(x_test).data.reshape(-1))
# test_pred.insert(0,list(model(x_train).data.reshape(-1))[-1])
pred_data= list(datas[:3])+list(model(x_train).data.reshape(-1)) + list(model(x_test).data.reshape(-1))

rmse=mean_squared_error(dataY[train_size:],test_pred)**0.5
r2=r2_score(dataY[train_size:],test_pred)
print(rmse)
print(r2)

restrain =train_pred
restest = test_pred
pred1 = [x.cpu().numpy() for x in restrain]
pred2 = [x.cpu().numpy() for x in restest]
pred = list(datas[:2]) + pred1 + pred2
pred = np.array(pred).reshape(1,-1)
pred = pred[0]
# with open('D:/data/2007-12baishuihe/txtfiles/annZG118model4.txt','w') as f1:
#     for i in range(len(datas)):
#         p=str(pred[i])
#         f1.write(p)
#         f1.write('\n')

plt.plot(dataY[train_size:], color='r')
# 原数据
plt.plot(test_pred, color='y')
# 预测数据
plt.title('ANN')
plt.xlabel('point')
plt.ylabel('displacement')
# plt.savefig('D:/picture/ZG118/ANN1.png')
plt.show()