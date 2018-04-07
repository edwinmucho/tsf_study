import tensorflow as tf

class ABC:
    pass

class Cnn:
    def __init__(self, sess, name, learning_rate=0.01):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            self.training = tf.placeholder(tf.bool)

            # W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            # L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
            # L1 = tf.nn.relu(L1)
            # L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1. 2. 2. 1], padding="SAME")

            # conv1 = tf.layers.conv2d(input=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
            # dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)
            #
            # conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            # dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training = self.training)
            #
            # conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], activation=tf.nn.relu)
            # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
            # dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

            cl1 = self.cnn_layer(X_img, 32)
            cl2 = self.cnn_layer(cl1, 64)
            cl3 = self.cnn_layer(cl2, 128)

            flat = tf.reshape(cl3, [-1, 4 * 4 * 128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            self.hypothesis = tf.layers.dense(inputs=dropout4, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def cnn_layer(self,input_data, f_units):
        conv = tf.layers.conv2d(inputs=input_data, filters=f_units, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], padding="SAME", strides=2)
        return tf.layers.dropout(inputs=pool, rate=0.7, training=self.training)

    def predict(self,x_test, training=False):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})