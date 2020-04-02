import tensorflow as tf

b = tf.constant(2)
a = tf.constant(3)
c = tf.multiply(b, a)
d = tf.add(b, a)
f = tf.add(d, c)
e = tf.subtract(d, c)
g = tf.divide(f, e)
with tf.Session() as sess:
    outs = sess.run(g)
    sess.close()
    print("oust = {}".format(outs))

b1 = tf.constant(3.0)
a1 = tf.constant(4.0)
c1 = tf.multiply(b1, a1)
d1 = tf.sin(c1)
e1 = tf.divide(b1, d1)
with tf.Session() as sess:
    outs1 = sess.run(e1)
    sess.close()
    print("outs1 = {}".format(outs1))
