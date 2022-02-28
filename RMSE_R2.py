import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data=[]
with open('D:/data/2007-12baishuihe/txtfiles/ZG111.txt','r') as f:
    d1=f.readlines()
for x1 in d1:
    x2=x1.split()
    for x3 in x2:
        x4=float(x3)
        data.append(x4)

data_pred=[]
with open('D:/data/2007-12baishuihe/txtfiles/ZG111pred.txt','r') as f:
    d2=f.readlines()
for y1 in d2:
    y2=y1.split()
    for y3 in y2:
        y4=float(y3)
        data_pred.append(y4)

#
# # datas=res+imf
#
#
# # with open('D:/data/2007-12baishuihe/txtfiles/DX01pred.txt','w') as f1:
# #     for i in range(len(datas)):
# #         p=str(datas[i])
# #         f1.write(p)
# #         f1.write('\n')



rmse_imf=mean_squared_error(data,data_pred)**0.5
r2_imf=r2_score(data,data_pred)
print('final',rmse_imf)
print('final',r2_imf)

plt.plot(data)
plt.plot(data_pred)
plt.show()