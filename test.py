import tensorflow as tf

# 定数
a = tf.constant(3)

# 変数
b = tf.Variable(0)

# プレースホルダー
c = tf.placeholder(tf.float32)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
added = tf.add(a, b)
with tf.Session() as sess:
    print(sess.run(added, feed_dict = {a: 3.0, b: 5.0}))