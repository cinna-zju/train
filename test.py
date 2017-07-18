import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x+b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    if i % 50 == 0:
        print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


curr_w, curr_b, curr_loss = sess.run([W, b, loss],{x:[1,2,3,4], y:[0,-1,-2,-3]})
print("w:%s b:%s loss:%s" % (curr_w, curr_b, curr_loss))

