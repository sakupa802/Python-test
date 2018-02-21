import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
# トレーニングデータの準備
x_data = np.random.rand(100).astype(np.float32) # 0~1の乱数を100個生成し、計算用に型変換
y_data = x_data * 0.1 + 0.3 # y = 0.1^x + 0.3 で表される直線上に乗る点を用意

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
# 線形回帰のモデル定義
a = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # a の初期値(-1.0から1.0の間のランダム)
b = tf.Variable(tf.zeros([1])) # b の初期値(0)

#一次方程式 a(傾き) b(切片)から y を求める
y = a * x_data + b

# Minimize the mean squared errors.
# コスト関数の定義
loss = tf.reduce_mean(tf.square(y - y_data)) # 全ての点における誤差の平均(予測データ y と実データ y_data の差の二乗)
optimizer = tf.train.GradientDescentOptimizer(0.5) # アルゴリズム最急降下法を使う。引数の 0.5 は学習率
train = optimizer.minimize(loss) # loss を最小化#

# Before starting, initialize the variables.  We will 'run' this first.
# 実行前の初期化
init = tf.global_variables_initializer()#

# Launch the graph.
# 実行フェイズ
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
   sess.run(train)
   if step % 20 == 0:
       print(step, sess.run(a), sess.run(b))#

# Learns best fit is a: [0.1], b: [0.3]#

# Close the Session when we're done.
sess.close()