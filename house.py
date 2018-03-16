import tensorflow as tf
import numpy as np
# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt

x_data = np.asarray([10, 22, 30, 40, 46, 53])
x_data = x_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける
# 物件の家賃
y_data = np.asarray([100, 240, 300, 400, 510, 620])
y_data = y_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける

#n_samples = y_data.shape[0]
#y_data = x_data * 0.1 + 0.3

# x_data = np.random.rand(100).astype(np.float32) # 0~1の乱数を100個生成し、計算用に型変換
# y_data = x_data * 0.1 + 0.3 # y = 0.1^x + 0.3 で表される直線上に乗る点を用意

# 線形回帰のモデル定義
# a = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # a の初期値(-1.0から1.0の間のランダム)
# b = tf.Variable(tf.zeros([1])) # b の初期値(0)
a = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = a * x_data + b # 一次方程式 a(傾き) b(切片)から y を求める

# 目的関数の定義
loss = tf.reduce_mean(tf.square(y - y_data)) # 全ての点における誤差の平均(予測データ y と実データ y_data の差の二乗)
#loss = tf.reduce_sum(tf.pow(activation-y, 2))/(2*n_samples)  # 二乗誤差
optimizer = tf.train.GradientDescentOptimizer(0.5) # アルゴリズム最急降下法(勾配降下法)を使う。引数の 0.5 は学習率(偏微分)
#optimizer = tf.train.AdamOptimizer(0.5) # 
train = optimizer.minimize(loss) # loss を最小化

# Sessionの開始前にVariables(変数)を初期化する
init = tf.global_variables_initializer()

# 実行
sess = tf.Session()
sess.run(init)

for step in range(1001):
   sess.run(train)
   if step % 20 == 0:
       print("loss=", "{:.9f}".format(sess.run(loss)))
       print(step, sess.run(a), sess.run(b))
       print((sess.run(a)*53+sess.run(b)))#15平米の時の家賃結果

# 散布図
plt.scatter(x_data, y_data)

# 回帰直線
plt.plot(x_data, (sess.run(a)*x_data+sess.run(b)))
plt.show()

sess.close()