from __future__ import print_function
import os
import sys
import timeit

import numpy

import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression, load_data
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import six.moves.cPickle as pickle
from unpickle import unpickle

data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
test = unpickle('cifar-10-batches-py/test_batch')

train_set_1 = data_batch_1["data"]
train_set_2 = data_batch_2["data"]
train_set_3 = data_batch_3["data"]
train_set_4 = data_batch_4["data"]
train_set_5 = data_batch_5["data"]
X_train = np.concatenate((train_set_1, train_set_2, train_set_3, train_set_4, train_set_5), axis=0)

y_train = np.concatenate((data_batch_1["labels"],data_batch_2["labels"],data_batch_3["labels"],data_batch_4["labels"],
                          data_batch_5["labels"]))



test_set = test["data"]
Xte_rows = test_set.reshape(train_set_1.shape[0], 32 * 32 * 3)
Yte = np.asarray(test["labels"])


Xval_rows = X_train[:7500, :] # take first 1000 for validation
Yval = y_train[:7500]
Xtr_rows = X_train[7500:50000, :] # keep last 49,000 for train
Ytr = y_train[7500:50000]


mean_train = Xtr_rows.mean(axis=0)
stdv_train = Xte_rows.std(axis=0)
Xtr_rows = (Xtr_rows - mean_train) / stdv_train
Xval_rows = (Xval_rows - mean_train) / stdv_train
Xte_rows = (Xte_rows - mean_train) / stdv_train

train_set = (Xtr_rows,Ytr)
valid_set = (Xval_rows,Yval)
test_set = (Xte_rows,Yte)


x = T.dvector()
y = T.dscalar()


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
            activation=theano.tensor.tanh
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


# compute number of minibatches for training, validation and testing
batch_size = 20

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels

rng = numpy.random.RandomState(1234)

# construct the MLP class
classifier = MLP(
    rng=rng,
    input=x,
    n_in=Xtr_rows.shape[1],
    n_hidden=400,
    n_out=10
)

# start-snippet-4
# the cost we minimize during training is the negative log likelihood of
# the model plus the regularization terms (L1 and L2); cost is expressed
# here symbolically
L1_reg = 0.000
L2_reg = 0.005
learning_rate = theano.shared(0.01)
cost = (
    classifier.negative_log_likelihood(y)
    + L1_reg * classifier.L1
    + L2_reg * classifier.L2_sqr
)
# end-snippet-4

# compiling a Theano function that computes the mistakes that are made
# by the model on a minibatch

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)
datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

gparams = [T.grad(cost, param) for param in classifier.params]


# eval_grad = theano.function(
#            inputs=[param for param in classifier.params],
#            outputs=[gparams])

updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(classifier.params, gparams)
]

train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
cost_aux = 4

patience = 100  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()


epoch = 0
n_epochs = 1000
done_looping = False
prev_cost = 10
print("before the while")
e_e = 0
f_f = 0
epoch_loss_list = []
epoch_val_list = []
f_graph = np.zeros([1,2])
s_graph = np.zeros([1,2])
t_graph = np.zeros([1,2])
while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(200):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number

            ite = (epoch - 1) * n_train_batches + minibatch_index
            epoch_loss_entry = [ite,epoch,float(minibatch_avg_cost)]
            epoch_loss_list.append(epoch_loss_entry)



            if (ite + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                epoch_val_entry = [ite,epoch,this_validation_loss]
                epoch_val_list.append(epoch_val_entry)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, ite * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = ite

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if epoch >3:
                done_looping = True
                break


end_time = timeit.default_timer()
print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print(('The code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

epoch_loss_np = np.reshape(epoch_loss_list,newshape=(len(epoch_loss_list),3))

plt(epoch_loss_np)
epoch_val_np = np.reshape(epoch_loss_list,newshape=(len(epoch_val_list),3))

#trying to plot this this



"""
for i in range(10000):
    cost = train_model(Xtr_rows, Ytr)
    if cost_aux > cost:
        learning_rate.set_value(learning_rate.get_value()*1.1)
    else:
        learning_rate.set_value(learning_rate.get_value()*0.9)
    cost_aux = cost
    print("cost is: ", cost, "learning rate: ", learning_rate.get_value())
"""
W1,b1,W2,b2 = classifier.params
W1 = W1.container.data
b1 = b1.container.data
W2 = W2.container.data
b2 = b2.container.data
params = [W1, b1, W2, b2]


W = T.matrix('W')
b = T.vector('b')
W_aux = T.matrix('W_aux')
b_aux = T.vector('b_aux')
z = T.dot(x, W) + b
s = T.tanh(z)
z2 = T.dot(s, W_aux) + b_aux
s2 = T.tanh(z2)
evaluate_layer = theano.function(inputs=[W, b, W_aux, b_aux, x], outputs=s2)

preds = np.argmax(evaluate_layer(W1, b1, W2, b2, Xte_rows),axis=1)
compare = (preds == Yte)*1
mean_accurance =compare.mean()
print("true_mean: "+ str(mean_accurance))

print("end")
