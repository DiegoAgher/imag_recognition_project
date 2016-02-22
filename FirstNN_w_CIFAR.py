from __future__ import print_function
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from LogisticRegression import LogisticRegression, load_data
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from unpickle import unpickle

data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
test = unpickle('cifar-10-batches-py/test_batch')

train_set_1 = data_batch_1["data"]
Xtr_rows = train_set_1.reshape(train_set_1.shape[0], 32 * 32 * 3)
Xtr_rows = (Xtr_rows - Xtr_rows.mean()) / Xtr_rows.std()
sub_x = Xtr_rows[0:250, :]
Ytr = np.asarray(data_batch_1["labels"])
Ytr_6 = (Ytr == 6)*1
sub_y = (Ytr[0:250] == 6)*1

test_set = test["data"]
Xte_rows = test_set.reshape(train_set_1.shape[0], 32 * 32 * 3)
Yte = np.asarray(test["labels"])

x = T.dvector()
y = T.dscalar()
NN = True

if NN:

    # start-snippet-1
    class HiddenLayer(object):
        def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                     activation=T.tanh):
            """
            Typical hidden layer of a MLP: units are fully-connected and have
            sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
            and the bias vector b is of shape (n_out,).

            NOTE : The nonlinearity used here is tanh

            Hidden unit activation is given by: tanh(dot(input,W) + b)

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.dmatrix
            :param input: a symbolic tensor of shape (n_examples, n_in)

            :type n_in: int
            :param n_in: dimensionality of input

            :type n_out: int
            :param n_out: number of hidden units

            :type activation: theano.Op or function
            :param activation: Non linearity to be applied in the hidden
                               layer
            """
            self.input = input
            # end-snippet-1

            # `W` is initialized with `W_values` which is uniformely sampled
            # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
            # for tanh activation function
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            # Note : optimal initialization of weights is dependent on the
            #        activation function used (among other things).
            #        For example, results presented in [Xavier10] suggest that you
            #        should use 4 times larger initial weights for sigmoid
            #        compared to tanh
            #        We have no info for other function, so we use the same as
            #        tanh.
            if W is None:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4

                W = theano.shared(value=W_values, name='W', borrow=True)

            if b is None:
                b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b', borrow=True)

            self.W = W
            self.b = b

            lin_output = T.dot(input, self.W) + self.b
            self.output = (
                lin_output if activation is None
                else activation(lin_output)
            )
            # parameters of the model
            self.params = [self.W, self.b]


    # start-snippet-2
    class MLP(object):
        """Multi-Layer Perceptron Class

        A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.
        Intermediate layers usually have as activation function tanh or the
        sigmoid function (defined here by a ``HiddenLayer`` class)  while the
        top layer is a softmax layer (defined here by a ``LogisticRegression``
        class).
        """

        def __init__(self, rng, input, n_in, n_hidden, n_out):
            """Initialize the parameters for the multilayer perceptron
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.TensorType
            :param input: symbolic variable that describes the input of the
            architecture (one minibatch)

            :type n_in: int
            :param n_in: number of input units, the dimension of the space in
            which the datapoints lie

            :type n_hidden: int
            :param n_hidden: number of hidden units

            :type n_out: int
            :param n_out: number of output units, the dimension of the space in
            which the labels lie

            """

            # Since we are dealing with a one hidden layer MLP, this will translate
            # into a HiddenLayer with a tanh activation function connected to the
            # LogisticRegression layer; the activation function can be replaced by
            # sigmoid or any other nonlinear function
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
            )

            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out
            )
            # end-snippet-2 start-snippet-3
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.logRegressionLayer.W).sum()
            )

            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.logRegressionLayer.W ** 2).sum()
            )

            # negative log likelihood of the MLP is given by the negative
            # log likelihood of the output of the model, computed in the
            # logistic regression layer
            self.negative_log_likelihood = (
                self.logRegressionLayer.negative_log_likelihood
            )
            # same holds for the function computing the number of errors
            self.errors = self.logRegressionLayer.errors

            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = self.hiddenLayer.params + self.logRegressionLayer.params
            # end-snippet-3

            # keep track of model input
            self.input = input


    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=Xtr_rows.shape[0],
        n_hidden=4,
        n_out=10
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    L1_reg = 0.0
    L2_reg = 0.0
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )


    MLP( Xtr_rows, Xtr_rows.shape[0], 3, 2)




else:

    rng = np.random

    N = Xtr_rows.shape[0]                                   # training sample size
    feats = Xtr_rows.shape[1] # number of input variables

    # generate a dataset: D = (input_values, target_class)
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    D = (Xtr_rows, Ytr_6)


    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")

    # initialize the weight vector w randomly
    #
    # this and the following bias variable b
    # are shared so they keep their values
    # between training iterations (updates)
    w = theano.shared(rng.randn(feats), name="w")

    # initialize the bias term
    b = theano.shared(0., name="b")

    print("Initial model:")
    print(w.get_value())
    print(b.get_value())

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
    prediction = p_1 > 0.5                    # The prediction thresholded
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
    gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                              # w.r.t weight vector w and
                                              # bias term b
                                              # (we shall return to this in a
                                              # following section of this tutorial)

    # Compile
    train = theano.function(
              inputs=[x,y],
              outputs=[prediction, xent],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    training_steps = 800
    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])

    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))