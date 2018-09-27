import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Net:
    def __init__(self, x, y_,h):
        
        #print(x.get_shape())
        in_dim = int(x.get_shape()[1]) # int(x.get_shape()[1]) 
        out_dim = int(y_.get_shape()[1])  #y_.get_shape()[1])

        self.x = x # input placeholder

        # simple 2-layer network
        W1 = weight_variable([in_dim,h])
        b1 = bias_variable([h])

        W2 = weight_variable([h,out_dim])
        b2 = bias_variable([out_dim])

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
        self.y = tf.matmul(h1,W2) + b2 # output layer

        self.var_list = [W1, b1, W2, b2]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        # train model with standard  gradient descent
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

    


