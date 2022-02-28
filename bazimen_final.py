import numpy as np
import matplotlib.pyplot as plt
import bazimen_EMD
import onelstm_test
import math
import annres
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import ANN_test

# df=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG92.csv')
# datas=df['ΔF(mm)']
# datas=np.array(datas)
datas = bazimen_EMD.data
datasres = bazimen_EMD.res
datasimf = bazimen_EMD.imfs
trainsize = int(0.8*(len(datas)))

restrain = annres.train_pred
restest =annres.test_pred
res1 = [x.cpu().numpy() for x in restrain]
res2 = [x.cpu().numpy() for x in restest]
res = list(datasres[:2]) + res1 + res2
res = np.array(res).reshape(1,-1)
res = res[0]




imftrain = onelstm_test.train_pred
imftest = onelstm_test.test_pred
imf1 = [x.cpu().numpy() for x in imftrain]
imf2 = [x.cpu().numpy() for x in imftest]
imf = list(datasimf[:2]) + imf1 + imf2
imf = np.array(imf).reshape(1,-1)
imf = imf[0]

imf=imf




data_pred = res + imf

# rmse_imf=mean_squared_error(datasres[trainsize:],res[trainsize:])**0.5
# r2_imf=r2_score(datasres[trainsize:],res[trainsize:])
# print('imfrmse',rmse_imf)
# print('imfr2',r2_imf)


rmse_imf=mean_squared_error(datasimf[trainsize:],imf[trainsize:])**0.5
r2_imf=r2_score(datasimf[trainsize:],imf[trainsize:])
print('imfrmse',rmse_imf)
print('imfr2',r2_imf)


n = len(datas)

rmse=mean_squared_error(datas[trainsize:],data_pred[trainsize:])**0.5
r2=r2_score(datas[trainsize:],data_pred[trainsize:])
print('allrmse',rmse)
print('allr2',r2)


plt.plot(datas[trainsize:],color='y',label = 'real')
plt.plot(data_pred[trainsize:],color='r',label = 'predict')
plt.legend(loc='best')

plt.show()