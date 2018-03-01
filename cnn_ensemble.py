import cnn_model as cm
import tensorflow as tf
import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

sess = tf.Session()

model_num = 2
models = []

for num in range(model_num):
    models.append(cm.Cnn(sess, "model"+str(num), learning_rate=0.001))

sess.run(tf.global_variables_initializer())
print("Ensemble Learning Started !! ")
total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print("Epoch : {:04d}  cost = {}".format(epoch + 1, avg_cost_list))

print("Ensemble Learning Finished !!")

test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
for m_idx, m in enumerate(models):
    print("{:04d} Accuracy : {}".format(m_idx, m.get_accuracy(mnist.test.images, mnist.test.labels)))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

print("Ensemble accuracy : {}: ".format(sess.run(ensemble_accuracy)))