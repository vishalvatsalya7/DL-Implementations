import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # reads in the MNIST dataset


# a function that shows examples from the dataset. If num is specified (between 0 and 9), then only pictures with those labels will beused
def show_pics(mnist, num = None):
    to_show = list(range(10)) if not num else [num]*10 # figure out which numbers we should show
    for i in range(100):
        batch = mnist.train.next_batch(1) # gets some examples
        pic, label = batch[0], batch[1]
        if np.argmax(label) in to_show:
            # use matplotlib to plot it
            pic = pic.reshape((28,28))
            plt.title("Label: {}".format(np.argmax(label)))
            plt.imshow(pic, cmap = 'binary')
            plt.show()
            to_show.remove(np.argmax(label))
            
            
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

learning_rate = 0.1
hidden_layer_neurons = 50
num_iterations = 5000

x = tf.placeholder(tf.float32, shape = [None, 784]) # none = the size of that dimension doesn't matter. why is that okay here? 
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# create our weights and biases for our first hidden layer
W_1, b_1 = weight_variable([784, hidden_layer_neurons]), bias_variable([hidden_layer_neurons])

# compute activations of the hidden layer
h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2_hidden = weight_variable([hidden_layer_neurons, 30])
b_2_hidden = bias_variable([30])
h_2 = tf.nn.relu(tf.matmul(h_1, W_2_hidden) + b_2_hidden)
# create our weights and biases for our output layer
W_2, b_2 = weight_variable([30, 10]), bias_variable([10])
# compute the of the output layer
y = tf.matmul(h_2,W_2) + b_2

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

# create an optimizer to minimize our cross entropy loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

# functions that allow us to gauge accuracy of our model
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # creates a vector where each element is T or F, denoting whether our prediction was right
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) # maps the boolean values to 1.0 or 0.0 and calculates the accuracy

# we will need to run this in our session to initialize our weights and biases. 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # initializes our variables
    for i in range(num_iterations):
        # get a sample of the dataset and run the optimizer, which calculates a forward pass and then runs the backpropagation algorithm to improve the weights
        batch = mnist.train.next_batch(100)
        optimizer.run(feed_dict = {x: batch[0], y_: batch[1]})
        # every 100 iterations, print out the accuracy
        if i % 100 == 0:
            # accuracy and loss are both functions that take (x, y) pairs as input, and run a forward pass through the network to obtain a prediction, and then compares the prediction with the actual y.
            acc = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1]})
            loss = cross_entropy_loss.eval(feed_dict = {x: batch[0], y_: batch[1]})
            print("Epoch: {}, accuracy: {}, loss: {}".format(i, acc, loss))
            
     # evaluate our testing accuracy       
    acc = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
    print("testing accuracy: {}".format(acc))
