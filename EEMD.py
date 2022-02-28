import numpy as np
from PyEMD import EEMD, EMD, Visualisation
import pylab as plt
import pandas as pd

df=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/DX01.csv')
data=df['ΔF(mm)']
data=np.array(data)

t=np.arange(len(data))
#EMD经验模态分解
eemd = EEMD()
imf = eemd.eemd(data)
row=imf.shape[0]
print(row)

#
# for num in range(row):
#     plt.subplot(row, 1, num+1)
#     plt.plot(t,imf[num], 'g')
#     plt.title("Imf " + str(num + 1))
#
# plt.show()



# imfs=imf[0:row-1,:]
# imfs=imfs.sum(axis=0)
# res=imf[row-1,:]

# print(imfs)
# vis = Visualisation()
# vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
# plt.show()

# with open('D:/data/2007-12baishuihe/EEMD_txt/DX01imfs.txt','w') as f1:
#     for i in range(len(data)):
#         p=str(imfs[i])
#         f1.write(p)
#         f1.write('\n')
#
# with open('D:/data/2007-12baishuihe/EEMD_txt/DX01res.txt', 'w') as f2:
#     for j in range(len(data)):
#         q = str(res[j])
#         f2.write(q)
#         f2.write('\n')

# plt.subplot(1,2,1)
# plt.scatter(t,data,color='red')
# plt.plot(t,res,color='yellow')
# plt.title('EMD')
# plt.ylabel('res')
#
#
# plt.subplot(1,2,2)
#
# plt.plot(t,data,color='yellow')
# plt.plot(t,imfs,color='red')
# plt.title('EMD')
# plt.ylabel('imf')
# # plt.savefig('D:/picture/ZG91/EEMD1.png',dpi=600)
# plt.show()
