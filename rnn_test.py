import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)
# with tf.variable_scope('initial_state') as scope:
#     batch_size = 3

#     pp.pprint(x_data)
#
#     # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch: 3
#     hidden_size = 2
#     cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
#     initial_state = cell.zero_state(batch_size, tf.float32)
#     outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
#                                          initial_state=initial_state, dtype=tf.float32)
#     sess.run(tf.global_variables_initializer())
#     pp.pprint(outputs.eval())

with tf.variable_scope('MultiRNNCell') as scope:
    # Make rnn
    # cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
    def lstm_cell():
        cell = rnn.BasicLSTMCell(5, state_is_tuple=True)
        return cell
    cells = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)  # 3 layers
    # print(x_data)
    # rnn in/out
    outputs, _states = tf.nn.dynamic_rnn(cells, x_data, dtype=tf.float32)
    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size