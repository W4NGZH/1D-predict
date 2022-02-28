import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from pycaret.regression import *
import warnings
warnings.filterwarnings("ignore")


datas=pd.read_csv('D:/ZG93.csv')
t=np.array(datas['t']).reshape(-1,1)
y=np.array(datas['F']).reshape(-1,1)
trainsize=int(0.8*len(t))
sc_t = StandardScaler()
t= sc_t.fit_transform(t)
train_t=t[:trainsize]
train_y=y[:trainsize]
test_t=t[trainsize:]
test_y=y[trainsize:]

# svr=SVR()
# svr.fit(train_t,train_y)
# pred=svr.predict(t)
#
# plt.plot(y,color='y')
# plt.plot(pred,color='r')
# plt.show()



plt.rcParams['figure.figsize'] = (8, 5)


class RELM_HiddenLayer:
    """
        正则化的极限学习机
        :param x: 初始化学习机时的训练集属性X
        :param num: 学习机隐层节点数
        :param C: 正则化系数的倒数
    """

    def __init__(self, x, num, C=10):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()
        # 权重w
        self.w = rnd.uniform(-1, 1, (columns, num))
        # 偏置b
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        self.H0 = np.matrix(self.sigmoid(np.dot(x, self.w) + self.b))
        self.C = C
        self.P = (self.H0.H * self.H0 + len(x) / self.C).I

    @staticmethod
    def sigmoid(x):
        """
            激活函数sigmoid
            :param x: 训练集中的X
            :return: 激活值
        """
        return 1.0 / (1 + np.exp(-x))

    # 回归问题 训练
    def regressor_train(self, T):
        """
            初始化了学习机后需要传入对应标签T
            :param T: 对应属性X的标签T
            :return: 隐层输出权值beta
        """
        all_m = np.dot(self.P, self.H0.H)
        self.beta = np.dot(all_m, T)
        return self.beta

    # 回归问题 测试
    def regressor_test(self, test_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:特征
            :return: 预测值
        """
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result


# 产生数据集


j = 0
# 绘制原始散点图
# plt.plot(t, y, 'or')

# 不同隐藏层线条设置不同的颜色
color = ['g', 'b', 'y', 'c', 'm']

# 比较不同隐藏层拟合效果

my_EML = RELM_HiddenLayer(train_t,1)
my_EML.regressor_train(train_y)
y_pred = my_EML.regressor_test(test_t)
plt.plot(test_y,color='y')
plt.plot(y_pred,color='r')
plt.title('EML_regress')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
print(y_pred)