import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import warnings
import EMD
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.rcParams['lines.color'] = 'r'

trenddata=EMD.res
date=pd.date_range(start='2006-12',end='2011-12',freq='M')
t=pd.DataFrame(date,columns=['ds'])
tre=pd.DataFrame(trenddata,columns=['y'])
Fdata=pd.concat([t,tre],axis=1)
# Fdata.columns=['ds','y']
trainsize=int(0.8*Fdata.shape[0])
testsize=Fdata.shape[0]-trainsize
traindata=Fdata.loc[:trainsize,:]
testdata=Fdata.loc[trainsize:,:]

pro = Prophet(growth='linear',yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False,changepoint_prior_scale=0.8,seasonality_prior_scale=25,holidays_prior_scale=0)
# pro = Prophet(growth='linear')
pro.fit(traindata)

future=pro.make_future_dataframe(periods=testsize-1,freq='M')

future.tail()
forecast = pro.predict(future)
pre=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# pro.plot(forecast)
# pro.plot_components(forecast,figsize=(4,4))
# plt.xticks(rotation=330)
# # resreal = trenddata
# # res = pre['yhat']
# # #
# # x = np.arange(len(resreal))
# # plt.plot(x,resreal,color = 'y')
# # plt.plot(x,res,color = 'r')
# # plt.savefig('D:/picture/ZG94/prophet11.png')
# plt.show()