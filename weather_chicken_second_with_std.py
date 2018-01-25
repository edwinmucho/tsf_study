import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# read csv file
xy = np.loadtxt('wchi5_second.csv', delimiter=',', dtype=np.float32)

# Training Data
x_raw = xy[:7000, :-1]
y_raw = xy[:7000, [-1]]

# Test Data
x_test_raw = xy[7000:, :-1]
y_test_raw = xy[7000:, [-1]]

# Make Standardization Scaler
x_scale = preprocessing.StandardScaler().fit(x_raw)
y_scale = preprocessing.StandardScaler().fit(y_raw)

# Transform each data using MinMaxScale (scope 0 to 1)
x_data = x_scale.transform(x_raw)
y_data = y_scale.transform(y_raw)
x_test_data = x_scale.transform(x_test_raw)
y_test_data = y_scale.transform(y_test_raw)

nb_classes = 10
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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

# Test
pred = sess.run(hypothesis,feed_dict={X: x_test_data})
pred = y_scale.inverse_transform(pred)
print("Prediction : ", pred)

