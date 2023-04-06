import numpy as np
import matplotlib.pyplot as plt
# 线性回归模型
from sklearn.linear_model import LinearRegression
# 拆分训练集
from sklearn.model_selection import train_test_split

'''
reshape(1,-1)转化成1行： 
reshape(2,-1)转换成两行： 
reshape(-1,1)转换成1列： 
reshape(-1,2)转化成两列
'''
# time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1)
# scores = np.array([56, 83, 47, 93, 47, 82, 45, 78, 55, 67, 57, 4, 12]).reshape(-1, 1)
# # print(time_studied)
# model = LinearRegression()  # 使用线性回归模型
# # 拟合/训练
# model.fit(time_studied, scores)
# print(model.predict(np.array([56]).reshape(-1, 1)))
# plt.scatter(time_studied, scores)
# # 从0到70,显示模型的预测(回归线),颜色是红色 r
# plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
# plt.ylim(0, 100)
# plt.show()
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1, 1)
# 拆分
time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.2)
model = LinearRegression()
model.fit(time_train, score_train)
# 看看准确率(如果出现负数,可能是训练和测试样本太小)
print(model.score(time_test, score_test))
