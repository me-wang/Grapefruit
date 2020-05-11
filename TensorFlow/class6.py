

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('mnist_data', one_hot=True)
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


# x代表input， f代表filter
def conv2d(x, f):
    return tf.nn.conv2d(x, f, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# 加卷积层
def add_conv_layer(input, shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    z = tf.nn.relu(conv2d(input, w) + b)

    # 池化
    return max_pool_2x2(z)


# 输入层
with tf.name_scope('input_layer'):
    input_images = tf.placeholder(tf.float32, [None, 784])
    images = tf.reshape(input_images, (-1, 28, 28, 1))
# 卷积层1
with tf.name_scope('conv_layer1'):
    conv1 = add_conv_layer(images, [5, 5, 1, 32])

# 卷积层2
with tf.name_scope('conv_layer2'):
    conv2 = add_conv_layer(conv1, [5, 5, 32, 64])
# 扁平
with tf.name_scope('flatten'):
    reshape = tf.reshape(conv2, [-1, 7*7*64])
# 隐藏层
with tf.name_scope('hidden_layer'):
    hidden = add_layer(reshape, 7*7*64, 1024, tf.nn.relu)
# 输出层
with tf.name_scope('output_layer'):
    output_layer = add_layer(hidden, 1024, 10)
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
    op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

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
            test_result = sess.run(accuarcy, feed_dict={input_images: mnist.test.images, label: mnist.test.labels})

            print("step %d,  test_accuarcy=%g" % (i, test_result))

