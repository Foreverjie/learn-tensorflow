import tensorflow as tf
from input_data import mnist

x = tf.placeholder("float", [None, 784])

w = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))
#预测值
y = tf.nn.softmax(tf.matmul(x, w) + b)
#正确值
y_ = tf.placeholder("float", [None, 10])
#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


