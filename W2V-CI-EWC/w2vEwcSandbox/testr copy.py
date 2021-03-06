print('Importing Libraries...')
import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#mnist = tf.keras.datasets.mnist
sess = tf.InteractiveSession()


num_iter = 800
trainset = mnist
testsets = [mnist]
disp_freq = 20
lams=[0]

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#model = Model(x, y_) # from the model class


# classification accuracy plotting
def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

test_accs = [] # initialize test accuracy array for each task 
for task in range(len(testsets)): #allocate test_accs & store testing data for each task
    test_accs.append(np.zeros(int(num_iter/disp_freq)))

print('Running Sim...')
for iter in range(num_iter): # ~800 iterations
    batch = trainset.train.next_batch(100)
    
    in_dim = int(x.get_shape()[1]) # 784 for MNIST
    out_dim = int(y_.get_shape()[1]) # 10 for MNIST
    #self.x = x # input placeholder

    # simple 2-layer network
    W1 = weight_variable([in_dim,50])
    b1 = bias_variable([50])

    W2 = weight_variable([50,out_dim])
    b2 = bias_variable([out_dim])

    h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
    y = tf.matmul(h1,W2) + b2 # output layer

    var_list = [W1, b1, W2, b2]
    # vanilla single-task loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # set loss (self.train_step )
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


    # performance metrics
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # probe preformance
##    plots = []
##    for task in range(len(testsets)):
##        feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
##        test_accs[task][int(iter/disp_freq)] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval(feed_dict=feed_dict)
##        plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq)+1], colors[task], label="Acc ")
##        plots.append(plot_h)
##    plot_test_acc(plots)
    
    if iter % 100 == 0:
        print(iter, accuracy)
        
plt.show()
    
##plt.subplot(1, len(lams), l+1)
##plots = []
##colors = ['r', 'b', 'g']
##for task in range(len(testsets)):
##    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
##    test_accs[task][int(iter/disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)
##    c = chr(ord('A') + task)
##    plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq)+1], colors[task], label="task " + c)
##    plots.append(plot_h)
##plot_test_acc(plots)
    





    # train model
    #model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})



#train_task(model, 800, 20, mnist, [mnist], x, y_, lams=[0])
