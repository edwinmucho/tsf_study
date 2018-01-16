import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# read csv file
xy = np.loadtxt('wchi5.csv', delimiter=',', dtype=np.float32)

# ready to use MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))  # Scale scope 0 to 1

# Training Data
x_data = xy[:7000, :-1]
y_data = xy[:7000, [-1]]

# Test Data
x_test_data = xy[7000:, :-1]
y_test_data = xy[7000:, [-1]]

# Transform each data using MinMaxScale (scope 0 to 1)
x_data = scaler.fit_transform(x_data)
y_data = scaler.fit_transform(y_data)
x_test_data = scaler.fit_transform(x_test_data)
y_test_data = scaler.fit_transform(y_test_data)

# Why don't transform together?
# -> 아 영작 싫다. MinMaxScale 써서 0과 1사이의 숫자로 나타낸 값을
#    다시 되돌리기 위해서는 Shape 이 같아야 하는데!
#    한꺼번에 MinMaxScale을 하면 shape=[None, 9] 로 설정이되서
#    필요한 y 값만 가지고 되돌릴수 없다.( y 값의 shape=[None,1] 이기 때문!)
#    왜 이렇게 되는지는 세세히 파악을 하지 못해
#    차후 해당 부분이 해결된 소스를 보면 그걸로 대체하는걸로 진행할까함.


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

# Test
pred = sess.run(hypothesis,feed_dict={X: x_test_data})
pred = scaler.inverse_transform(pred)
print("Prediction : ", pred)

