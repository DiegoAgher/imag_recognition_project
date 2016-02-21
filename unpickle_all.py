
from __future__ import print_function
import numpy as np
import theano
import time
from unpickle import unpickle

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test)

    # loop over all test rows
    for i in xrange(num_test):
        distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
        min_index = np.argmin(distances)
        Ypred[i] = self.ytr[min_index]
        print("iteration number"+str(i))
    return Ypred


data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
test = unpickle('cifar-10-batches-py/test_batch')


def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i


def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(np.transpose(x))
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  print("done one")
  return loss_i

def L(X, y, W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    # evaluate loss over all examples in X without using any for loops
    # left as exercise to reader in the assignment
    delta = 1.0

    scores = W.dot(np.transpose(X))
    hlp = np.arange(y.shape[0])
    margins = scores - scores[y, hlp] + 1
    margins[np.where(scores < 0)] = 0
    total_loss = margins.sum()

    return total_loss




train_set_1 = data_batch_1["data"]
Xtr_rows = train_set_1.reshape(train_set_1.shape[0], 32 * 32 * 3)
Ytr = np.asarray(data_batch_1["labels"])

test_set = test["data"]
Xte_rows = test_set.reshape(train_set_1.shape[0], 32 * 32 * 3)
Yte = test["labels"]

b = np.ones((10, 1))


def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.1

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    print("ix: "+str(ix))
    print("is is over?!", it.finished)
    it.iternext() # step to next dimension

  return grad

W = np.random.rand(10, 3072) * 0.001

def CIFAR10_loss_fun(W):
  return L(Xtr_rows, Ytr, W)

print("made it this far")
W = np.random.rand(10, 3072) * 0.001 # random weight vector
print("then this shit")
df = eval_numerical_gradient(CIFAR10_loss_fun, W) # get the gradient

print("the gradient is "+str(df))

