import tensorflow as tf
import numpy as np
# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt

# データを生成
x_data = np.asarray([[10,20,30], [24,80,10], [30,40,9], [40,25,15], [51,80,70], [60,80,50]])
#x_data = x_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける
# 物件の家賃
y_data = np.asarray([[103], [242], [304], [402], [519], [625]])
#y_data = y_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける

n_1 = len(x_data)#6
n_2 = len(x_data[0])#3

# モデル
#x_d = tf.placeholder(tf.float32, [None,n])
#y_d = tf.placeholder(tf.float32, [None,1])
w = tf.Variable(tf.zeros([n_1, n_2]))
b = tf.Variable(tf.zeros([1]), name = "bias")
y_hat = w * x_data + b

# 目的関数
loss = tf.reduce_sum(tf.square(y_data - (y_hat)))

# 確率的最急勾配法
rate = 0.5
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# 変数初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(42001):
    loss_val = sess.run(train)
    if step % 2000 == 0:
        loss_val = sess.run(loss)
        print('Step: %03d,   Loss: %5.4f' % (step,loss_val), sess.run(w))
        sess.run(w)
        sess.run(b)

# # 目的関数
# loss = tf.reduce_sum(tf.square(y_data - y_hat))

# # 確率的最急勾配法
# rate = 0.5
# optimizer = tf.train.AdamOptimizer()
# train = optimizer.minimize(loss)


# # 変数初期化
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# for step in range(12001):
#     if step % 1000 == 0:
#         loss_val = sess.run(loss, feed_dict={y_hat:y_train}) 
#         print('Step: %03d,   Loss: %5.4f' % (step,loss_val))
#         w_val = sess.run(w)
#     sess.run(train, feed_dict={x_d:x_train,y_d:y_train})

# sess.close()

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