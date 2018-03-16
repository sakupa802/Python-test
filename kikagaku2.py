import tensorflow as tf
import numpy as np
# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt

# データを生成
n=10
x_data = np.random.rand(n).astype(np.float32)
y_data = 0.14* x_data**4  -  0.35* x_data**3  +  0.2 * x_data*2 + 0.01*x_data + 2

# 　ノイズを加える
y_data = y_data + 0.009 * np.random.randn(n)

# モデル
xt = tf.placeholder(tf.float32, [None,5])
yt = tf.placeholder(tf.float32, [None,1])
w  = tf.Variable(tf.zeros([5,1]))
y  = tf.matmul(xt,w)

# 目的関数
loss = tf.reduce_sum(tf.square(y-yt))

#最急勾配法
rate = 0.5
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# トレーニングデータ
y_train = y_data.reshape([n,1])
x_train = np.zeros([n,5])
for i in range(0,n):
    for j in range(0,5):
        x_train[i][j] = x_data[i]**j

# 変数初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(12001):
    if step % 1000 == 0:
        loss_val = sess.run(loss, feed_dict={xt:x_train, yt:y_train}) 
        print('Step: %03d,   Loss: %5.4f' % (step,loss_val))   
        w_val = sess.run(w)
    sess.run(train, feed_dict={xt:x_train,yt:y_train})

# 線形回帰予測関数
def predict(x):
    result = 0.0
    for n in range(0,5):
        result += w_val[n][0] * x**n
    return result

fig = plt.figure()
subplot = fig.add_subplot(1,1,1)

plt.scatter(x_data,y_data)
linex = np.linspace(0,1,100)
liney = predict(linex)
subplot.plot(linex,liney)
plt.show()

# ノイズ付きデータを描画
# plt.scatter(x_data,y_data)
# plt.show()

tf.exit()