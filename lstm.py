import torch
import torch.nn as nn
from torch.optim import *
import matplotlib.pyplot as plt
import numpy as np
import EMD

device=torch.device('cuda:0')

datas=EMD.imfs
datas= np.reshape(datas, (-1, 1))
seq=2
def creat_dataset(dataset,seqnum):
    data_x=[]
    data_y=[]
    for i in range(len(dataset)-seqnum-1):
        data_x.append(dataset[i:i+seqnum])
        data_y.append(dataset[i+seqnum:i+seqnum+2])
    return  np.asarray(data_x),np.asarray(data_y)
dataX,dataY=creat_dataset(datas,seq)


train_size=int(len(dataX)*0.8)
x_train=dataX[:train_size]
y_train=dataY[:train_size]
x_train = x_train.reshape(-1, seq, 1) #将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.reshape(-1, seq)
x_train=torch.from_numpy(x_train).float().to(device)
y_train=torch.from_numpy(y_train).float().to(device)
x_test=(torch.tensor(dataX[train_size:])).reshape(-1,seq,1).float().to(device)
y_test=(torch.tensor(dataY[train_size:])).reshape(-1,seq).float().to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        # 输入格式是1，输出隐藏层大小是32

        self.linear = nn.Linear(32*seq, 2)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 32*seq)
        x = self.linear(x)
        return x


model=Net()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
scheduler = lr_scheduler.StepLR(optimizer,step_size=1000,gamma = 0.8)
loss_func=nn.MSELoss()
lr_list=[]
Epochs=6000
for epoch in range(Epochs):
    output = model(x_train)
    loss = loss_func(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    if epoch % 1000 == 0 and epoch > 0:
        print("epoch:{}, loss:{}".format(epoch, loss))

model.eval()

test_loss = loss_func(model(x_test), y_test)
print(test_loss)
pred_train= list(model(x_train).data.reshape(-1))
pred_test=list(model(x_test).data.reshape(-1))

# t_test=list(range(len(pred_train)+2,len(datas)))
pred_data= list(model(x_train).data.reshape(-1)) + list(model(x_test).data.reshape(-1))
pred=[]
for i in range(1,len(pred_data)-1):

    pred.append(pred_data[i])
pred=np.array(pred).reshape(-1,2)
para_a=1
datatrain=datas[3:len(datas)-1]

finalloss=10000
# predloss=[]
while para_a>0:
    predmean = []
    para_b = 1 - para_a
    for i in range(len(pred)):
        y1=pred[i,0]*para_a+pred[i,1]*para_b

        predmean.append(y1)
    predmean=(torch.tensor(predmean)).reshape(-1,1)
    datatrain=(torch.tensor(datatrain)).reshape(-1,1)
    losssum=loss_func(datatrain,predmean)
    if losssum<finalloss:
        finalloss=losssum
        final_a=para_a
    # predloss.append(para_a)
    # predloss.append(para_b)
    # predloss.append(losssum)
    para_a=para_a-0.01

print(final_a)

pred_final=[]
final_b=1-final_a
for i in range(len(pred)):
    ytrue = pred[i, 0] * final_a + pred[i, 1] * final_b

    pred_final.append(ytrue)

pred_final=list([datas[0],datas[1],pred_data[0]])+list(pred_final)+list([pred_data[len(pred_data)-1]])

plt.subplot(1,2,1)
plt.plot(datas, label="real")
plt.plot(pred_final, label="pred_test",color='yellow')
plt.title('manylstm')
plt.xlabel('point')
plt.ylabel('displacement')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(range(Epochs),lr_list,color = 'r')
plt.title('learning rate')
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.savefig('D:/data/2007-12baishuihe/picture/ZG93manylstm.png',dpi=600)
plt.show()