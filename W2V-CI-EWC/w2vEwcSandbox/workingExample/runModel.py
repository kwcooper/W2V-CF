# code to run the EWC model
# battery 43% -> 39%

print("Importing libraries...") # because it takes awhile
import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
# this is depreciated, and will throw an error but it works,
# the new one has bugs as of august, may be worth a new look

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from EWCnet import Net # this is the nn with EWC


# TD
# use alternatives such as official/mnist/dataset.py from tensorflow/models

# Use to display MNIST data
# input is a 1D array of length 784
# could probably just sqrt?
def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = np.arange(mnist.train.images.shape[1]) # k convert to numpy array
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

# classification accuracy plotting
def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    
# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, lams=[0]):
    # run twice, once with EWC and once without for compareison
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss()
        else:
            model.update_ewc_loss(lams[l])
        # initialize test accuracy array for each task 
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(int(num_iter/disp_freq)))
        # train on current task
        for iter in range(num_iter):
            # grab next set of MNIST images
            batch = trainset.train.next_batch(100)
            model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l+1)
                plots = []
                colors = ['r', 'b', 'g']
                # check the accuracy on all of the previous tasks. 
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
                    test_accs[task][int(iter/disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)
                    c = chr(ord('A') + task)
                    plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq)+1], colors[task], label="task " + c)
                    plots.append(plot_h)
                plot_test_acc(plots)
                if l == 0: 
                    plt.title("vanilla sgd")
                else:
                    plt.title("ewc")
    plt.show()




mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#mnist = tf.keras.datasets.mnist

sess = tf.InteractiveSession()

# define input and target placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

##x = tf.placeholder(tf.int32, shape=[1]) # train_inputs
##y_ = tf.placeholder(tf.int32, shape=[1,1]) #train_labels


# instantiate new model
model = Net(x, y_) # simple 2-layer network

# initialize variables
sess.run(tf.global_variables_initializer())

print("\nFirst Task:")

# training 1st task
print("Training network...")
train_task(model, 800, 20, mnist, [mnist], x, y_, lams=[0])

# Fisher information
print("Computing Fisher information...")
model.compute_fisher(mnist.validation.images, sess, num_samples=200, plot_diffs=True) # use validation set for Fisher computation
F_row_mean = np.mean(model.F_accum[0], 1)
mnist_imshow(F_row_mean)
plt.title("W1 row-wise mean Fisher");

model.star() # save current optimal weights

print("\nSecond Task:")
# permuting mnist for 2nd task
mnist2 = permute_mnist(mnist)

if 0:
    plt.subplot(1,2,1)
    mnist_imshow(mnist.train.images[5])
    plt.title("original task image")
    plt.subplot(1,2,2)
    mnist_imshow(mnist2.train.images[5])
    plt.title("new task image");
    plt.show()

# training 2nd task
print("Training network...")
train_task(model, 800, 20, mnist2, [mnist, mnist2], x, y_, lams=[0, 15])

# Fisher information for 2nd task
print("Computing Fisher information...")
model.compute_fisher(mnist2.validation.images, sess, num_samples=200, plot_diffs=True)

F_row_mean = np.mean(model.F_accum[0], 1)
mnist_imshow(F_row_mean)
plt.title("W1 row-wise mean Fisher");
plt.show()

print("\nThird Task:")
# permuting mnist for 3rd task
mnist3 = permute_mnist(mnist)

model.star() # save current optimal weights

# training 3rd task
print("Training network...")
train_task(model, 800, 20, mnist3, [mnist, mnist2, mnist3], x, y_, lams=[0, 15])



