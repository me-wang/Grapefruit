# mnist fc 1层隐藏层  test acc 95-96%
# 用CNN实现mnist,看一看acc能提高多少

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('mnist_data', one_hot=True)


# 定义网路结构，全连接
# W,b tensor Variable
# 定义一个函数来封装一下构建层的功能
# z = w * x + b
# 激活!!
# 这个函数被调用完是不是相当于把这一层的向前传播给做了？
def add_layer(input, input_size, output_size, activation_function=None):
    # 从正态分布里面采样,前人得出的结论
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))      # 一般去比较小的常数0.1
    # numpy 广播机制
    z = tf.add(tf.matmul(input, w), b)
    if activation_function == None:
        a = z
    else:
        a = activation_function(z)
    return a

# 损失函数，softmax之后的值
def crossentropy(label, output):
    return -tf.reduce_sum(tf.multiply(label, tf.log(output)), axis=1)

# 验证或者叫预测
def predict(output, label):
    #找最大的
    correct_prediction = tf.equal(tf.argmax(output, axis=1), tf.argmax(label, axis=1))
    accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuarcy:", accuarcy)
    return accuarcy

# 输入层
with tf.name_scope('input_layer'):
    input_images = tf.placeholder(tf.float32, [None, 784])
# 隐藏层
with tf.name_scope('hidden_layer'):
    hidden = add_layer(input_images, 784, 16, tf.nn.relu)
# 输出层
with tf.name_scope('output_layer'):
    output_layer = add_layer(hidden, 16, 10)
    output = tf.nn.softmax(output_layer)
# 输入的label
with tf.name_scope('accuarcy'):
    label = tf.placeholder(tf.float32, [None, 10])
    accuarcy = predict(output, label)

# 优化，损失
with tf.name_scope('loss'):
    loss = tf.reduce_mean(crossentropy(label, output))
    tf.summary.scalar('loss:', loss)

with tf.name_scope('optimizer'):
    op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

merged = tf.summary.merge_all()

# 运行我们的图
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log/', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch_image, batch_label = mnist.train.next_batch(100)
        _, merged_summary = sess.run([op, merged], feed_dict={input_images:batch_image, label: batch_label})
        writer.add_summary(merged_summary, i)
        if i % 100 ==0:
            train_result = sess.run(accuarcy, feed_dict={input_images: mnist.train.images, label: mnist.train.labels})
            test_result = sess.run(accuarcy, feed_dict={input_images: mnist.test.images, label: mnist.test.labels})

            print("step %d, train_accuarcy= %g, test_accuarcy=%g" % (i, train_result, test_result))


# # 定义一层卷积层
# # w, b
# # 卷积核3*3， 3个通道，10个卷积核
#
# w = tf.Variable(tf.truncated_normal([3, 3, 3, 10], stddev=0.1))
# b = tf.Variable(tf.constant(0.1, shape=[10]))
# # fc
# # a(xw + b)
# # conv
#
# z = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b)
# u = tf.nn.relu(tf.nn.conv2d(z, w, strides=[1, 1, 1, 1], padding='SAME') + b)
#
# # pooling
# p = tf.nn.max_pool(u, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
# # fc,扁平
# tf.reshape(p, [-1,])