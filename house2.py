import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

#ボストンデータ読み込み
boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['target'] = boston.target

# RM（部屋数）とPRICE（価格）で回帰分析
# X_RM = boston.data[:, [5]] # RM(部屋数)
# y_PRICE = boston.target # target(価格)

#学習データ準備
f_num = df.shape[1] - 1
train_X = np.array(df.iloc[:, :f_num])
train_Y = np.array(df.iloc[:, f_num: f_num + 1])
n_samples = train_X.shape[0]

#正規化
ss = StandardScaler()
ss.fit(train_X)
train_X = ss.transform(train_X)
#print(train_X)

#プレースホルダー
with tf.name_scope('data'):
	X = tf.placeholder(tf.float32, shape = [None, f_num], name = "X")
	Y = tf.placeholder(tf.float32, name = "Y")

#変数（パラメータ）
with tf.name_scope('parameter'):
	W = tf.Variable(tf.zeros([f_num, 1]), name = "weight")
	b = tf.Variable(tf.zeros([1]), name = "bias")

#モデル
with tf.name_scope('model'):
	pred = tf.add(tf.matmul(X, W), b) #matmulは行列の積
	#pred = X*W+b
	# 売上額　　 回帰係数　　　　　  回帰係数　　　　   回帰係数			  定数項
	# y　= 0.12×店舗面積 + 0.34×従業員数 + 0.56×座席数 ～～～～～ + 0.89

#損失関数
with tf.name_scope('loss'):
	# Mean squared error
	loss = tf.reduce_mean(tf.square(pred - Y))
	tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_step = optimizer.minimize(loss)

#決定係数(R2)
with tf.name_scope('r2'):
	r2 = 1 - (tf.reduce_sum(tf.square(Y - pred)) / tf.reduce_sum(tf.square(Y - tf.reduce_mean(Y))))
	tf.summary.scalar('r2', r2)

with tf.Session() as sess:
	# ログの設定
	summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter("boston_log", sess.graph)
	
	sess.run(tf.global_variables_initializer())#変数初期化
	
	for i in range(1000):
		sess.run(train_step, feed_dict={X: train_X, Y: train_Y})
		if i != 0 and i % 200 == 0: # 200ステップごとに精度を出力
			train_summary, train_loss, train_r2 = sess.run([summary, loss, r2], feed_dict={X: train_X, Y:train_Y})# コストと精度を出力
			writer.add_summary(train_summary, i) #summaryの更新
			
			print("Step:", '%04d' % (i), "loss=", "{:.9f}".format(train_loss), "r2=", "{:.9f}".format(train_r2), "W=", sess.run(W), "b=", sess.run(b))			
			
	training_cost, training_r2 = sess.run([loss,r2], feed_dict={X: train_X, Y: train_Y})
	print("Training cost=", training_cost, "Training r2=", training_r2, "W=", sess.run(W), "b=", sess.run(b), '\n')
