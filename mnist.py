## Loading MNIST from keras

import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-1000], X_train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
## Printing dimensions
print(X_train.shape, y_train.shape)
## Visualize the first digit
plt.imshow(X_train[0], cmap="Greys");
plt.show()

## Preprocessing, changing dimension of input impages from N*28*28 to N*784
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

print('Train dimension:');print(X_train.shape)
print('Test dimension:');print(X_test.shape)

## Changing labels to one-hot encoded vector
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print('Train labels dimension:');print(y_train.shape)
print('Test labels dimension:');print(y_test.shape)

## Importing required libraries
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
s = tf.InteractiveSession()

## Defining various initialization parameters for 784-512-256-10 MLP model
num_classes = y_train.shape[1]
num_features = X_train.shape[1]
num_output = y_train.shape[1]
num_layers_0 = 512
num_layers_1 = 256
starter_learning_rate = 0.001
regularizer_rate = 0.1

# Placeholders for the input data
input_X = tf.placeholder('float32', shape = (None, num_features), name="input_X")
input_y = tf.placeholder('float32', shape = (None, num_classes), name="input_Y")
## for dropout layer
keep_prob = tf.placeholder(tf.float32)

## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
weights_0 = tf.Variable(tf.random_normal([num_features, num_layers_0], stddev=
                                         (1/tf.sqrt(float(num_features)))))
bias_0 = tf.Variable(tf.random_normal([num_layers_0]))

weights_1 = tf.Variable(tf.random_normal([num_layers_0, num_layers_1], stddev=
                                         (1/tf.sqrt(float(num_layers_0)))))
bias_1 = tf.Variable(tf.random_normal([num_layers_1]))

weights_2 = tf.Variable(tf.random_normal([num_layers_1, num_output], stddev=
                                         (1/tf.sqrt(float(num_layers_1)))))
bias_2 = tf.Variable(tf.random_normal([num_output]))

## Initializing weights and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X, weights_0) + bias_0)
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)

hidden_output_1 =tf.nn.relu(tf.matmul(hidden_output_0_0, weights_1)+bias_1)
hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)

predicted_y = tf.sigmoid(tf.matmul(hidden_output_1_1,weights_2) + bias_2)

# Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y, labels=input_y)) + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))


## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
## Adam optimizer for finding the rihgt weight
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = [weights_0, weights_1, weights_2, bias_0, bias_1, bias_2])

## Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


## Training parameters
batch_size = 128
epochs = 14
dropout_prob = 0.6

training_accuracy = []
training_loss = []
testing_accuracy =[]

s.run(tf.global_variables_initializer())
for epoch in range(epochs):
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0,X_train.shape[0], batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]], input_y: y_train[arr[index:index+batch_size]], keep_prob:dropout_prob})

    training_accuracy.append(s.run(accuracy, feed_dict = {input_X: X_train, input_y: y_train,keep_prob: 1}))
    training_loss.append(s.run(loss, {input_X: X_train, input_y: y_train,keep_prob: 1}))

    ## Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1), s.run(predicted_y, {input_X: X_test, keep_prob:1}).argmax(1)))
    print("Epoch:{0}, Train_loss: {1:.2f} Train acc: {2:.3f}, Test acc: {3:.3f}".format(epoch, training_loss[epoch], training_accuracy[epoch], testing_accuracy[epoch]))

# Plotting chart of training and testing accuracy as a function of iterations
iterations = list(range(epochs))
plt.plot(iterations, training_accuracy, label='Train')
plt.plot(iterations, testing_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.show()
print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
print("Test Accuracy: {0:.2f}".format(testing_accuracy[-1]))
