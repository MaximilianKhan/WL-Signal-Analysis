import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np 
import pandas as pd 
import tensorflow as tf

# Import all of our data contained in the csv
training_data = pd.read_csv('training-data' + os.sep + 'training-data.csv', header=None)

examples = len(training_data)
features = 100
X_train = training_data.loc[:, 0:99].as_matrix()
X_train = np.array(X_train)
y_train = training_data.loc[:, 100].as_matrix()
y_train = np.array(y_train).reshape(examples, 1)

test_data = pd.read_csv('training-data' + os.sep + 'messy-training-data.csv', header=None)
test_examples = len(test_data)
X_test = test_data.loc[:, 0:99].as_matrix()
X_test = np.array(X_test)
y_test = test_data.loc[:, 100].as_matrix()
y_test = np.array(y_test).reshape(test_examples, 1)

# I need to also get test data...

# Define the neural network model
hidden_layer_nodes = 10
X = tf.placeholder(tf.float32, shape=[None, features], name='X')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# Layer 1
w1 = tf.Variable(tf.random_normal(shape=[features, hidden_layer_nodes]), name='w1')
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]), name='b1')

# Layer 2
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]), name='w2')
b2 = tf.Variable(tf.random_normal(shape=[1, 1]), name='b2')
hidden_output = tf.nn.relu(tf.add(tf.matmul(X, w1), b1), name='hidden_output')
y_hat = tf.nn.relu(tf.add(tf.matmul(hidden_output, w2), b2), name='y_hat')
loss = tf.reduce_mean(tf.square(y - y_hat), name='loss')

# Setup gradient descent
learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# Session
sess = tf.Session()
writer = tf.summary.FileWriter('./tf-log')
writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())

# Add tf logging to allow for the visualization of weights and biases over epochs. 

epochs = 50000
batch_size = 5

# Before training
curr_loss = sess.run(loss, feed_dict={X:X_train, y:y_train})
print('Loss before training:', curr_loss)

# Train
for i in range(epochs):
    rand_index = np.random.choice(examples, size=batch_size)
    sess.run(train_step, feed_dict={X:X_train[rand_index], y:y_train[rand_index]})

# After training:
curr_loss = sess.run(loss, feed_dict={X:X_train, y:y_train})
print('Loss after training:', curr_loss)

test_loss = sess.run(loss, feed_dict={X:X_test, y:y_test})
print('Loss for test:', test_loss)