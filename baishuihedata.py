import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhht.emd import EMD


df=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG118.csv')
data=df['ΔF(mm)']
data=np.array(data)
t=np.arange(len(data))

decomposer = EMD(data)
imfs = decomposer.decompose()
row=imfs.shape[0]
imf=imfs[0:row-1,:]
imf=imf.sum(axis=0)
res=imfs[row-1,:]

plt.subplot(1,2,1)
plt.scatter(t,data,color='yellow')
plt.plot(t,res,color='red')
plt.title('trend')

plt.subplot(1,2,2)
plt.plot(t,imf,color='green')
plt.title('random')
plt.show()
