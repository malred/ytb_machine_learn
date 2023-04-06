from sklearn.datasets import load_breast_cancer
# k近邻分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 乳腺癌数据集
data = load_breast_cancer()
print(data.feature_names)
print(data.target_names)
# 拆分数据集
# 所有用于预测的参数作为x(np.array),所有目标作为y(np.array)
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)
# print(x_test)
# print(y_test)
# n_neighbors -> 需要多少个距离最近的邻居
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
# 预测并评分
print(clf.score(x_test, y_test))

