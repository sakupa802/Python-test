import tensorflow as tf
import numpy as np
# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt

# トレーニングデータの準備
# 物件の専有面積(m2)平方メートル
x_data = np.asarray([10, 11.55, 20.46, 23.89, 23.89, 44.36, 43.79, 40.8, 26.35, 22.4, 26.6, 37.6, 39.09, 54.51, 56.45, 40.23, 40.23, 42.71, 42.71, 42.71, 37.6, 36.79, 15.5, 25.11, 22.43, 21.42, 22.04, 28, 27.64, 57.78])
x_data = x_data * 0.01 # 結果がNaN(無限大)になるので、0.01掛ける
# 物件の月額家賃(円)
y_data = np.asarray([30000, 31000, 58000, 72000, 74000, 136000, 137000, 164000, 109000, 112000, 133000, 143000, 155000, 205000, 206000, 130000, 131000, 133000, 134000, 134000, 143000, 143000, 58000, 95000, 107000, 80000, 89000, 105000, 110000, 150000])
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
# y = a * x_data + b # 一次方程式 a(傾き) b(切片)から y を求める
y = b * tf.exp(x_data / (a+x_data))

# 目的関数の定義
loss = tf.reduce_mean(tf.square(y - y_data)) # 全ての点における誤差の平均(予測データ y と実データ y_data の差の二乗)
#loss = tf.reduce_sum(tf.pow(activation-y, 2))/(2*n_samples)  # 二乗誤差
optimizer = tf.train.GradientDescentOptimizer(0.5) # アルゴリズム最急降下法(勾配降下法)を使う。引数の 0.5 は学習率
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

# 散布図
plt.scatter(x_data, y_data)

# 回帰直線
plt.plot(x_data, (sess.run(b) * tf.exp(x_data / (sess.run(a)+x_data))))
plt.show()

sess.close()