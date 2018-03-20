import tensorflow as tf
import numpy as np
# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.preprocessing import StandardScaler

# # データを生成
x_data = np.asarray([[10,20,30], [24,80,10], [30,40,9], [40,25,15], [51,80,70], [60,80,50]], dtype=np.float32)
#x_data = x_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける
# # 物件の家賃
y_data = np.asarray([[103], [242], [304], [402], [519], [625]], dtype=np.float32)
#y_data = y_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける#正規化

n_1 = len(x_data)#6
n_2 = len(x_data[0])#3 (目的変数の数)

X = tf.placeholder(tf.float32, shape = [None, n_2], name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

# モデル
W = tf.Variable(tf.zeros([n_2, 1]))
b = tf.Variable(tf.zeros([1]), name = "bias")
y_hat = tf.add(tf.matmul(X, W), b)
#tf.transpose()

# 目的関数
loss = tf.reduce_sum(tf.square(y_data - y_hat))

# 確率的最急勾配法
rate = 0.5
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#変数初期化

    for step in range(120001):
        sess.run(train, feed_dict={X: x_data, Y:y_data})
        if step % 10000 == 0:
            loss_val = sess.run(loss, feed_dict={X: x_data, Y:y_data}) 
            print('Step: %03d,   Loss: %5.4f' % (step,loss_val))
            w_val = sess.run(W)
            b_val = sess.run(b)
            print(w_val)
            print(b_val)

# # 線形回帰予測関数
# def predict(x):
#     result = 0.0
#     for n in range(0,5):
#         result += w_val[n][0] * x**n
#     return result

# fig = plt.figure()
# subplot = fig.add_subplot(1,1,1)

# plt.scatter(x_data,y_data)
# linex = np.linspace(0,1,100)
# liney = predict(linex)
# subplot.plot(linex,liney)
# plt.show()

# tf.exit()