import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


xy = np.loadtxt('wchi5.csv', delimiter=',', dtype=np.float32)
x_data = xy[:7000, :-1]
y_data = xy[:7000, [-1]]

x_test_data = xy[7000:, :-1]
y_test_data = xy[7000:, [-1]]

scaler = MinMaxScaler(feature_range=(0,1))
x_data = scaler.fit_transform(x_data)
y_data = scaler.fit_transform(y_data)
x_test_data = scaler.fit_transform(x_test_data)
y_test_data = scaler.fit_transform(y_test_data)

nb_classes = 8
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, nb_classes])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([nb_classes, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Hypothesis
hypothesis = tf.matmul(X, W) + b
# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Start populating the filename queue.
for step in range(7001):
   cost_val, hy_val, _ = sess.run(
       [cost, hypothesis, train],
       feed_dict={X: x_data, Y: y_data})
   if step % 1000 == 0:
       print(step, "Cost: ", cost_val,"\nPrediction:\n", hy_val)
pred = sess.run(hypothesis,feed_dict={X: x_test_data})
pred = scaler.inverse_transform(pred)
print("Prediction : ", pred)
