import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import pandas as pd
import datetime
import dataload
import warnings
warnings.filterwarnings("ignore")

# data=dataload.data
data=dataload.res


date_list=pd.date_range(start='2006-11-15',periods=62,freq='M')

trainsize=int(len(data)*0.8)


t=date_list
df1=pd.DataFrame(t,columns=['ds'])
df2=pd.DataFrame(data,columns=['y'])
df=pd.concat([df1,df2],axis=1)
# df['cap']=700
df_train=df.iloc[:trainsize,:]
testnum=len(data)-trainsize
print(trainsize)
print(testnum)

pro = Prophet(growth='linear')
pro.fit(df_train)
future = pro.make_future_dataframe(periods=testnum,freq='M')

# future['cap'] = 700
future.tail()
forecast = pro.predict(future)
pre=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
print(pre)
pro.plot(forecast)

pro.plot_components(forecast)
plt.show()
