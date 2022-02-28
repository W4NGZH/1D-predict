import numpy as np
import matplotlib.pyplot as plt
from pyhht.emd import EMD
import pandas as pd
# from pyEMD import EMD,Visualisation
from pyhht.visualization import plot_imfs


# data_x=[]
# with open('D:/data/2007-12baishuihe/baishuihe_CSV/ZG92.txt','r') as f:
#     d=f.readlines()
# for x1 in d:
#     x2=x1.split()
#     for x3 in x2:
#         x4=float(x3)
#         data_x.append(x4)
# data=np.array(data_x).reshape(-1,1)

df=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG93.csv')
data=df['ΔF(mm)']
data=np.array(data)
# data=data[0:-1:3]
# data=[]
# for i in range(len(datas)):
#     if i%12!=11:
#         data.append(datas[i])
# data=np.array(data).reshape(-1,1)
# print(len(data))

t=np.arange(len(data))
#EMD经验模态分解
emd = EMD(data)
imf = emd.decompose()
row=imf.shape[0]




imfs=imf[2:row-1,:]
imfs=imfs.sum(axis=0)
res=imf[row-1,:]
#
#
plt.subplot(1,2,1)
plt.scatter(t,data,color='red')
plt.plot(t,res,color='yellow')
plt.title('EMD')
plt.ylabel('res')

#
# plt.subplot(1,2,2)
#
#
# plt.plot(t,imfs,color='red')
# plt.title('EMD')
# plt.ylabel('imf')
# plt.savefig('D:/picture/ZG92/EMD1.png',dpi=600)
plt.show()

