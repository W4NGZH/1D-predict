import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prophet0712
import matplotlib
font = {
    'family':'SimHei',
    'weight':'bold',
    'size':15
}
matplotlib.rc("font", **font)

trend=prophet0712.pre['yhat']
df1=pd.read_csv('D:/data/2007-12baishuihe/baishuihe_CSV/ZG93.csv')
date1=df1['month']
df=pd.read_csv('D:/anasis.csv')
data=df['数据']
date=df['日期']



plt.rcParams['axes.unicode_minus'] = False
fig,ax=plt.subplots(2,1,figsize=(8,8))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.3)

ax[0].plot(date1,trend,'red',label='预测值')
ax[0].plot(date1,prophet0712.trenddata,'black',label='测量值')
ax[0].legend()
ax[0].set_xlabel('日期/年-月')
ax[0].set_ylabel('趋势/mm')
for tick in ax[0].get_xticklabels():
    tick.set_rotation(340)
ax[0].xaxis.set_major_locator(plt.MaxNLocator(nbins=12))


ax[1].plot(date,data,'black')
ax[1].set_xlabel('日期/月-日')
ax[1].set_ylabel('年季节性变化/mm')
# for tick in ax[0].get_xticklabels():
#     tick.set_rotation(330)
ax[1].xaxis.set_major_locator(plt.MaxNLocator(nbins=14))
plt.savefig('D:/prophet.tif',dpi=600)
plt.show()


# fig,ax=plt.subplots()
# ax.plot(date,datas['ZG93'],'green',label='ZG93')
# ax.plot(date,datas['ZG118'],'orange',label='ZG118')
# ax.plot(date,datas['XD01'],'pink',label='XD01')

# fig1,ax=plt.subplots()
# ax.plot(date,datas['ZG93_real'][:12],color='black',marker='s',ms=3,label='实际值')
# ax.plot(date,datas['ZG93_predict'][:12],color='red',marker='^',ms=3,label='预测值')
# ax.set_xlabel('时间/年-月')
# ax.set_ylabel('趋势项/mm')
# ax.legend()
# for tick in ax.get_xticklabels():
#     tick.set_rotation(330)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)



# ax[0][1].plot(t1,imfs,color='black')
# ax[0][1].set_xlabel('Monitoring period /month')
# ax[0][1].set_ylabel('imf/mm')
# ax[0][1].set_title('ZG93')
# for tick in ax[0][1].get_xticklabels():
#     tick.set_rotation(330)
#
#
#
#
# ax[1][0].plot(t2,res2,color='black')
# ax[1][0].set_xlabel('Monitoring period /month')
# ax[1][0].set_ylabel('res2/mm')
# ax[1][0].set_title('ZG118')
# for tick in ax[1][0].get_xticklabels():
#     tick.set_rotation(330)
#
#
# ax[1][1].plot(t2,imfs2,color='black')
# ax[1][1].set_xlabel('Monitoring period /month')
# ax[1][1].set_ylabel('imf/mm')
# ax[1][1].set_title('ZG118')
# for tick in ax[1][1].get_xticklabels():
#     tick.set_rotation(330)
#
#
# ax[2][0].plot(t2,res3,color='black')
# ax[2][0].set_xlabel('Monitoring period /month')
# ax[2][0].set_ylabel('res/mm')
# ax[2][0].set_title('XD01')
# for tick in ax[2][0].get_xticklabels():
#     tick.set_rotation(330)
#
#
# ax[2][1].plot(t2,imfs3,color='black')
# ax[2][1].set_xlabel('Monitoring period /month')
# ax[2][1].set_ylabel('imf/mm')
# ax[2][1].set_title('XD01')
# for tick in ax[2][1].get_xticklabels():
#     tick.set_rotation(330)

plt.show()