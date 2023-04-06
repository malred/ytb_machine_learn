import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# # 手写数字样本
# mnist = tf.keras.datasets.mnist
# # 拆分数据集
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # 缩小,或者说规范化数据集
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# # 1*输入层 2*密集层(2*隐藏层) 1*输出层
# model = tf.keras.models.Sequential()  # 创建基本网络
# # 展平, 因为输入的是图像(input_shape=28*28)
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # 密集层: 神经元和下一层的每个神经元都连接 -> 复杂度高
# ## units: 该层的神经元数量,越多越复杂(64,128,256...)
# ## activation: 激活函数
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# ## softmax: 根据该层神经元输出的值,统计不同可能(分类)的概率,并且这些概率加起来为1
# model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
# # 编译, (指定优化器,损失函数,感兴趣的指标)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # 训练
# ## epochs是训练轮数
# model.fit(x_train, y_train, epochs=3)
# # 评估
# loss, accuracy = model.evaluate(x_test, y_test)
# print(accuracy)
# print(loss)
# # 保存模型
# model.save('digits.model')
model = tf.keras.models.load_model('digits.model')
for x in range(1, 6):
    img = cv.imread(f'pics/{x}.png')[:, :, 0]
    # invert是翻转,为了让数字为黑色
    img = np.invert(np.array([img]))
    # 预测
    prediction = model.predict(img)
    # argmax 得到 输出里 最高值 的索引
    print(f'The result is probably: {np.argmax(prediction)}')
    # cmap=plt.cm.binary使图像为黑白色
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
