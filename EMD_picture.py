import math
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack
import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib
font = {
    'family':'SimHei',
    'weight':'bold',
    'size':9
}
matplotlib.rc("font", **font)

# 判定当前的时间序列是否是单调序列
def ismonotonic(x):
    max_peaks = signal.argrelextrema(x, np.greater)[0]
    min_peaks = signal.argrelextrema(x, np.less)[0]
    all_num = len(max_peaks) + len(min_peaks)
    if all_num > 0:
        return False
    else:
        return True


# 寻找当前时间序列的极值点
def findpeaks(x):
    #     df_index=np.nonzero(np.diff((np.diff(x)>=0)+0)<0)

    #     u_data=np.nonzero((x[df_index[0]+1]>x[df_index[0]]))
    #     df_index[0][u_data[0]]+=1

    #     return df_index[0]
    return signal.argrelextrema(x, np.greater)[0]


# 判断当前的序列是否为 IMF 序列
def isImf(x):
    N = np.size(x)
    pass_zero = np.sum(x[0:N - 2] * x[1:N - 1] < 0)  # 过零点的个数
    peaks_num = np.size(findpeaks(x)) + np.size(findpeaks(-x))  # 极值点的个数
    if abs(pass_zero - peaks_num) > 1:
        return False
    else:
        return True


# 获取当前样条曲线
def getspline(x):
    N = np.size(x)
    peaks = findpeaks(x)
    #     print '当前极值点个数：',len(peaks)
    peaks = np.concatenate(([0], peaks))
    peaks = np.concatenate((peaks, [N - 1]))
    if (len(peaks) <= 3):
        #         if(len(peaks)<2):
        #             peaks=np.concatenate(([0],peaks))
        #             peaks=np.concatenate((peaks,[N-1]))
        #             t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
        #             return interpolate.splev(np.arange(N),t)
        t = interpolate.splrep(peaks, y=x[peaks], w=None, xb=None, xe=None, k=len(peaks) - 1)
        return interpolate.splev(np.arange(N), t)
    t = interpolate.splrep(peaks, y=x[peaks])
    return interpolate.splev(np.arange(N), t)


#     f=interp1d(np.concatenate(([0,1],peaks,[N+1])),np.concatenate(([0,1],x[peaks],[0])),kind='cubic')
#     f=interp1d(peaks,x[peaks],kind='cubic')
#     return f(np.linspace(1,N,N))


# 经验模态分解方法
def emd(x):
    imf = []
    while not ismonotonic(x):
        x1 = x
        sd = np.inf
        while sd > 0.1 or (not isImf(x1)):
            #             print isImf(x1)
            s1 = getspline(x1)
            s2 = -getspline(-1 * x1)
            x2 = x1 - (s1 + s2) / 2
            sd = np.sum((x1 - x2) ** 2) / np.sum(x1 ** 2)
            x1 = x2

        imf.append(x1)
        x = x - x1
    imf.append(x)
    return imf

df1=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG93.csv')
data1=df1['ΔF(mm)']
data1=np.array(data1)

df2=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG118.csv')
data2=df2['ΔF(mm)']
data2=np.array(data2)

df3=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/DX01.csv')
data3=df3['ΔF(mm)']
data3=np.array(data3)

# data_x=[]
# with open('D:/data/2007-12baishuihe/baishuihe_CSV/ZG91.txt','r') as f:
#     d=f.readlines()
# for x1 in d:
#     x2=x1.split()
#     for x3 in x2:
#         x4=float(x3)
#         data_x.append(x4)
# data=np.array(data_x).reshape(-1,1)
# data=data.flatten()

# t1=pd.date_range(start='2006-12',end='2012-1',freq='M')
t1=df1['month']
imf1=emd(data1)
row1=len(imf1)
print(row1)
imfs=np.array(imf1[0:row1-1])
imfs=imfs.sum(axis=0)
res=imf1[-1]

t2=df2['时间']
# t2=pd.date_range(start='2006-12',end='2013-1',freq='M')
imf2=emd(data2)
row2=len(imf2)
imfs2=np.array(imf2[0:row2-1])
imfs2=imfs2.sum(axis=0)
res2=imf2[-1]


imf3=emd(data3)
row3=len(imf3)

imfs3=np.array(imf3[0:row3-1])
imfs3=imfs3.sum(axis=0)
res3=imf3[-1]

# for j in range(len(imfs)):
#     if imfs[j]<-800:
#         imfs[j] = (imfs[j-1] + imfs[j+1])/2

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.subplot(4,1,1)
# plt.plot(t,data)
# plt.xlabel('Monitoring period /month')
# plt.ylabel('data/mm')
#
# plt.subplot(4,1,2)
# plt.plot(t,imf1[0])
# plt.xlabel('monitoring period /month')
# plt.ylabel('imf1/mm')
#
# plt.subplot(4,1,3)
# plt.plot(t,imf1[1])
# plt.xlabel('Monitoring period /month')
# plt.ylabel('imf2/mm')
#
# plt.subplot(4,1,4)
# plt.plot(t,imf1[2])
# plt.xlabel('Monitoring period /month')
# plt.ylabel('res/mm')

# plt.savefig('D:/picture/DX01/EMDfinal1.png',dpi=600)
# plt.show()

plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(6,6))
fig1,ax=plt.subplots(3,2,figsize=(7.5,6))
ax[0][0].plot(t1,res,color='black')
ax[0][0].set_xlabel('日期/年-月')
ax[0][0].set_ylabel('趋势项/mm')
ax[0][0].set_title('ZG93')
for tick in ax[0][0].get_xticklabels():
    tick.set_rotation(330)
ax[0][0].xaxis.set_major_locator(plt.MaxNLocator(nbins=8))


ax[0][1].plot(t1,imfs,color='black')
ax[0][1].set_xlabel('日期/年-月')
ax[0][1].set_ylabel('波动项/mm')
ax[0][1].set_title('ZG93')
for tick in ax[0][1].get_xticklabels():
    tick.set_rotation(330)
ax[0][1].xaxis.set_major_locator(plt.MaxNLocator(nbins=8))



ax[1][0].plot(t2,res2,color='black')
ax[1][0].set_xlabel('日期/年-月')
ax[1][0].set_ylabel('趋势项/mm')
ax[1][0].set_title('ZG118')
for tick in ax[1][0].get_xticklabels():
    tick.set_rotation(330)
ax[1][0].xaxis.set_major_locator(plt.MaxNLocator(nbins=8))


ax[1][1].plot(t2,imfs2,color='black')
ax[1][1].set_xlabel('日期/年-月')
ax[1][1].set_ylabel('波动项/mm')
ax[1][1].set_title('ZG118')
for tick in ax[1][1].get_xticklabels():
    tick.set_rotation(330)
ax[1][1].xaxis.set_major_locator(plt.MaxNLocator(nbins=8))


ax[2][0].plot(t2,res3,color='black')
ax[2][0].set_xlabel('日期/年-月')
ax[2][0].set_ylabel('趋势项/mm')
ax[2][0].set_title('XD01')
for tick in ax[2][0].get_xticklabels():
    tick.set_rotation(330)
ax[2][0].xaxis.set_major_locator(plt.MaxNLocator(nbins=8))


ax[2][1].plot(t2,imfs3,color='black')
ax[2][1].set_xlabel('日期/年-月')
ax[2][1].set_ylabel('波动项/mm')
ax[2][1].set_title('XD01')
for tick in ax[2][1].get_xticklabels():
    tick.set_rotation(330)
ax[2][1].xaxis.set_major_locator(plt.MaxNLocator(nbins=8))

# plt.savefig('D:/EMDfinal.png',dpi=600)
plt.show()
