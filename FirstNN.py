__author__ = 'diegoinsydo'
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

x = T.dvector()
y = T.dscalar()

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.1 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

theta1 = theano.shared(np.array(np.random.rand(3,3), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(4,1), dtype=theano.config.floatX))

hid1 = layer(x, theta1)

out1 = T.sum(layer(hid1, theta2))
fc = (out1 - y)**2

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),
        (theta2, grad_desc(fc, theta2))])

run_forward = theano.function(inputs=[x], outputs=out1)

inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2)
exp_y = np.array([1, 1, 0, 0])
cur_cost = 0

for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

print(run_forward([0,1]))
print(run_forward([1,1]))
print(run_forward([1,0]))
print(run_forward([0,0]))

print "end"