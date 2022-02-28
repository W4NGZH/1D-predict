import datetime
import pandas as pd
import dataload
import matplotlib.pyplot as plt

data=dataload.data
datestart=datetime.datetime.strptime("2003-07-10",'%Y-%m-%d')
dateend=datetime.datetime.strptime("2011-01-01",'%Y-%m-%d')
date_list = []
date_list.append(datestart.strftime('%Y-%m-%d'))
while datestart<dateend:
# 日期叠加一天
    datestart+=datetime.timedelta(days=+6)
# 日期转字符串存入列表
    date_list.append(datestart.strftime('%Y-%m-%d'))

t=date_list
df1=pd.DataFrame(t,columns=['ds'])
df2=pd.DataFrame(data,columns=['y'])
df=pd.concat([df1,df2],axis=1)
df.to_csv('D:/data/baishuihe/landslide1.csv',sep=',',header=True,index=True)

