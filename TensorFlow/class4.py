import tensorflow
# 要用到MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from skimage import data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 训练集
print("Train data")
print(mnist.train.images.shape, mnist.train.labels.shape)
# 验证集
print("validation data")
print(mnist.validation.images.shape, mnist.validation.labels.shape)
# 测试集
print("test data")
print(mnist.test.images.shape, mnist.test.labels.shape)
# 数据已经做了归一化
print("Train data example", mnist.train.images[0])

# matplotlib展示前8张图
for i in range(8):
    print("。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。")
    img = data.astronaut()
    plt.figure(num='astronaut', figsize=(8,8 ))
    plt.subplot(1, 8, i+1)
    plt.title('nunber pic')
    plt.imshow(mnist.train.images[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    print(mnist.train.images[i].reshape(28, 28))
    # print(mnist.train.labels[i])

# 定义网路结构，全连接
# W,b tensor  Variable
# def add_layer(input, input_size, output_size,):
