import numpy as np
import matplotlib.pyplot as plt
import EMD
# import prophet0712
import ANN_test
import math
import annres
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# df=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG92.csv')
# datas=df['ΔF(mm)']
# datas=np.array(datas)
datas = EMD.data
datasimf = EMD.imfs
datasres = EMD.res
trainsize = int(0.8*(len(datas)))

restrain = annres.train_pred
restest =annres.test_pred
res1 = [x.cpu().numpy() for x in restrain]
res2 = [x.cpu().numpy() for x in restest]
res = list(datasres[:3]) + res1 + res2
res = np.array(res).reshape(1,-1)
res = res[0]



# res1 = prophet0712.pre['yhat']
# res = np.array(res1)

imftrain = ANN_test.train_pred
imftest = ANN_test.test_pred
imf1 = [x.cpu().numpy() for x in imftrain]
imf2 = [x.cpu().numpy() for x in imftest]
imf = list(datasimf[:3]) + imf1 + imf2
imf = np.array(imf).reshape(1,-1)
imf = imf[0]

imf=imf


data_pred = res + imf

rmse_imf=mean_squared_error(datasimf,imf)**0.5
r2_imf=r2_score(datasimf,imf)
print('final',rmse_imf)
print('final',r2_imf)

# with open('D:/data/2007-12baishuihe/txtfiles/ZG93imf_pred.txt','w') as f1:
#     for i in range(len(datas)):
#         p=str(imf[i])
#         f1.write(p)
#         f1.write('\n')

# data = np.array(EMD.imfs)
# data=data.sum(axis=0)
# data = datas
data = datasimf + EMD.res
n = len(data)

rmse=mean_squared_error(datas,data_pred)**0.5
r2=r2_score(datas,data_pred)
print('final',rmse)
print('final',r2)

# plt.subplot(1,2,1)
plt.plot(data,color='y',label = 'real')
plt.plot(data_pred,color='r',label = 'predict')
plt.legend(loc='best')

# plt.subplot(1,2,2)
# plt.plot(resreal)
# plt.plot(res)
# plt.savefig('D:/picture/ZG118/imf_new.png')
plt.show()