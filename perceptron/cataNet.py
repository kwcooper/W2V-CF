import random
import numpy as np
import matplotlib.pyplot as plt

# Question, do perceptrons suffer from CF?

class Perceptron:

    def __init__(self):
        self.w1 = random.random()*2-1
        self.w2 = random.random()*2-1
        self.bias = random.random()*2-1
        self.lc = 0.1  # learning constant

    def forward(self, i1, i2):
        netInput = (i1 * self.w1) + (i2 * self.w2) + self.bias
        return self.step_function(netInput)
        #return self.sigmoid(netInput)

    def train(self, i1, i2, target):
        output = self.forward(i1,i2)
        error = target - output
        self.w1 += error * i1 * self.lc
        self.w2 += error * i2 * self.lc
        self.bias += error * 1 * self.lc
        return abs(error)

    def step_function(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

def testNet(x):
    tests = [0,0,0,0]
    if x.forward(1,1) == 1:
        tests[0] = 1
    if x.forward(1,0) == 0:
        tests[1] = 1
    if x.forward(0,1) == 0:
        tests[2] = 1
    if x.forward(0,0) == 0:
        tests[3] = 1
    return tests

# Define a network and train it in full
# This will be our interleved comparison
print("\nInterleaved")
a = Perceptron()
t = 0
epochs = 150
errList = []
print("Training")
print("Weights init: ", a.w1,a.w2)
while t < epochs:
    error = 0
    error += a.train(0,0,0)   # first training AND 
    error += a.train(0,1,0)   # second training AND
    error += a.train(1,0,0)   # third training AND
    error += a.train(1,1,1)   # fourth training AND
    if t % 10 == 0:
        print("Epoch ", t, ": ", error)
    errList.append(error)
    t += 1
print("Weights end: ", a.w1,a.w2)

# Tests
print(a.forward(1,1), " (1)")
print(a.forward(1,0), " (0)")
print(a.forward(0,1), " (0)")
print(a.forward(0,0), " (0)")

### Plot the error
##plt.plot(errList)
##title = "Error: Interleaved, Step, Epochs: " + str(epochs)
##plt.title(title)
##plt.show()

print("\nSequential")
b = Perceptron()
t = 0
errList = []
print("Training 1")
print("Weights init: ", b.w1,b.w2)
while t < 50:
    error = 0
    error += b.train(0,0,0)   # first training AND 
    error += b.train(0,1,0)   # second training AND
    error += b.train(0,0,0)   # first training AND 
    error += b.train(0,1,0)   # second training AND
    if t % 10 == 0:
        print("Epoch ", t, ": ", error)
    errList.append(error)
    t += 1
print("Weights end: ", b.w1,b.w2)

# Tests
print(b.forward(1,1), " (1)")
print(b.forward(1,0), " (0)")
print(b.forward(0,1), " (0)")
print(b.forward(0,0), " (0)")

t = 0
print("Training 2")
print("Weights init: ", b.w1,b.w2)
while t < 50:
    error = 0
    error += b.train(1,0,0)   # third training AND
    error += b.train(1,1,1)   # fourth training AND
    if t % 10 == 0:
        print("Epoch ", t, ": ", error)
    t += 1
print("Weights end: ", b.w1,b.w2)

# Tests
print(b.forward(1,1), " (1)")
print(b.forward(1,0), " (0)")
print(b.forward(0,1), " (0)")
print(b.forward(0,0), " (0)")



tst = testNet(b)
print(tst)
    

