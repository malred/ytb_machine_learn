from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
# 数字数据集
from sklearn.datasets import load_digits

digits = load_digits()
# 标准化数据集
data = scale(digits.data)
print(data)
# 参数: 聚类数量,初始化方式,初始点个数
model = KMeans(n_clusters=10, init='random', n_init=10)
model.fit(data)
# 预测
# model.predict()
