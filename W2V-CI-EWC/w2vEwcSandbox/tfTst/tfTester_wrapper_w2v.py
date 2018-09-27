print('Importing Libraries...') # because mnist takes ages
import tensorflow as tf
import numpy as np
from copy import deepcopy
#from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tfTester_class import Net
from corpusFunc import Corpus



# classification accuracy plotting
def plot_helper(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    
def trainNetwork(model, num_iter, disp_freq, trainset, testsets, x, y_, lams=[0]):
    l = 0
    model.set_vanilla_loss()
    
    test_accs = [] # initialize test accuracy data array for each task 
    for task in range(len(testsets)): #allocate test_accs & store testing data for each task
        test_accs.append(np.zeros(int(num_iter/disp_freq)))

    # train on current task
    for i in range(0,len(trainset)):
    #for iter in range(num_iter): # ~800 iterations
        #batch = trainset.train.next_batch(100) # need to write a batching function
        # train model
        model.train_step.run(feed_dict={x: trainset[i], y_: testsets[0][i]})

        #ACC task
        if iter % disp_freq == 0:
            plt.subplot(1, len(lams), l+1)
            plots = []
            colors = ['r', 'b', 'g']
            for task in range(len(testsets)): # test accuracy for each task
                feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
                test_accs[task][int(iter/disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)
                c = chr(ord('A') + task)
                plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq)+1], colors[task], label="task " + c)
                plots.append(plot_h)
            plot_helper(plots)
            plt.title("Training Acc: vanilla sgd")
               
    print('num_iter =', num_iter)
    print('test_accs:', np.shape(test_accs))
    print('len plt handles',len(plots))
    print(test_accs)
    plt.show()
    


# init vocab functions
c = Corpus()
# Grab training data
tst,trn = c.fetchData([0,0],'rand', 10)

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#mnist = tf.keras.datasets.mnist
sess = tf.InteractiveSession()

dataShape = len(c.vocab)
outputShape = 0
embeddingSize = 10
iterations = 800
dispFreq = 20

# from prudvhi's code
#train_inputs = tf.placeholder(tf.int32, shape = [1])
#train_labels = tf.placeholder(tf.int32, shape = [1, 1])

x = tf.placeholder(tf.float32, shape=[1]) # why not length of vocab?
y_ = tf.placeholder(tf.float32, shape=[1,1]) # why not length of output?

# instantiate new model, simple 2-layer network
net = Net(x, y_,embeddingSize) 

# initialize variables
sess.run(tf.global_variables_initializer())

print("\nFirst Task:") # training 1st task
print("Training network...")
# 
trainNetwork(net,iterations,dispFreq,trn,[tst], x, y_,lams=[0])









##
##
##
##num_iter = 800
##trainset = mnist
##testsets = [mnist]
##disp_freq = 20
##lams=[0]
##
##x = tf.placeholder(tf.float32, shape=[None, 784])
##y_ = tf.placeholder(tf.float32, shape=[None, 10])
###model = Model(x, y_) # from the model class
##
##
### classification accuracy plotting
##def plot_test_acc(plot_handles):
##    plt.legend(handles=plot_handles, loc="center right")
##    plt.xlabel("Iterations")
##    plt.ylabel("Test Accuracy")
##    plt.ylim(0,1)
##
### variable initialization functions
##def weight_variable(shape):
##    initial = tf.truncated_normal(shape, stddev=0.1)
##    return tf.Variable(initial)
##
##def bias_variable(shape):
##    initial = tf.constant(0.1, shape=shape)
##    return tf.Variable(initial)
##
##test_accs = [] # initialize test accuracy array for each task 
##for task in range(len(testsets)): #allocate test_accs & store testing data for each task
##    test_accs.append(np.zeros(int(num_iter/disp_freq)))
##
##print('Running Sim...')
##for iter in range(num_iter): # ~800 iterations
##    batch = trainset.train.next_batch(100)
##    
##    in_dim = int(x.get_shape()[1]) # 784 for MNIST
##    out_dim = int(y_.get_shape()[1]) # 10 for MNIST
##    #self.x = x # input placeholder
##
##    # simple 2-layer network
##    W1 = weight_variable([in_dim,50])
##    b1 = bias_variable([50])
##
##    W2 = weight_variable([50,out_dim])
##    b2 = bias_variable([out_dim])
##
##    h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
##    y = tf.matmul(h1,W2) + b2 # output layer
##
##    var_list = [W1, b1, W2, b2]
##    # vanilla single-task loss function
##    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
##
##    # set loss (self.train_step )
##    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
##
##
##    # performance metrics
##    # reduce the mean of the noise-contrastive estimation training loss
##    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
##    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##
##    # probe preformance
####    plots = []
####    for task in range(len(testsets)):
####        feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
####        test_accs[task][int(iter/disp_freq)] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval(feed_dict=feed_dict)
####        plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq)+1], colors[task], label="Acc ")
####        plots.append(plot_h)
####    plot_test_acc(plots)
##    
##    if iter % 100 == 0:
##        print(iter, accuracy)
##        
##plt.show()
##    
####plt.subplot(1, len(lams), l+1)
####plots = []
####colors = ['r', 'b', 'g']
####for task in range(len(testsets)):
####    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
####    test_accs[task][int(iter/disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)
####    c = chr(ord('A') + task)
####    plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq)+1], colors[task], label="task " + c)
####    plots.append(plot_h)
####plot_test_acc(plots)
##    
##
##
##
##
##
##    # train model
##    #model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
##
##
##
###train_task(model, 800, 20, mnist, [mnist], x, y_, lams=[0])
##
