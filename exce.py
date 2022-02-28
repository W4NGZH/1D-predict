import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import dataload
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

trainsize=int(len(dataload.res)*0.85)
t=list(range(0,len(dataload.res)))
t = np.reshape(t, (-1, 1))
x_random=np.array(dataload.res)
x_random= np.reshape(x_random, (-1, 1))
t_train=t[:trainsize]
t_test=t[trainsize:]

# mm=MinMaxScaler()
# x_random1=mm.fit_transform(x_random)
x_train=x_random[:trainsize]


regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                        param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                    "gamma": np.logspace(-2, 2, 5)})
# regressor =SVR(kernel='rbf')
regressor.fit(t_train, x_train)
x_random_pred=regressor.predict(t)
x_random_pred= np.reshape(x_random_pred, (-1, 1))
# x_random_pred=mm.inverse_transform(x_random_pred)

print(x_random_pred)

plt.plot(t, x_random, color = 'red')
plt.plot(t, x_random_pred, color = 'blue')
plt.title('SVR')
plt.xlabel('t')
plt.ylabel('zpred')

# plt.savefig('z927SVR.png',dpi=600)
plt.show()