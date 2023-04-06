from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 支持向量机
from sklearn.svm import SVC
# k近邻(用于比较)
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()
x = data.data
y = data.target
# 如果random_state=0,表示不会每次都打乱数据集再拆分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# 分类器
## kernel: 核函数; C: 软边距
clf = SVC(kernel='linear', C=3)
clf.fit(x_train, y_train)
print(f'svc: {clf.score(x_test, y_test)}')
# k近邻
clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)
print(f'knn: {clf2.score(x_test, y_test)}')
