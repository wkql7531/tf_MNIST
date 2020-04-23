import tensorflow as tf
from pkg.tensorflow.tensorflow.examples.tutorials.mnist import input_data

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.math.log(y))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()
sess.run(init)

# Learning
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))