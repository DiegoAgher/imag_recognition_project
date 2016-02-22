__author__ = 'diegoinsydo'
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
NN=False

if NN :

    def layer(x, w):
        b = np.array([1], dtype=theano.config.floatX)
        new_x = T.concatenate([x, b])
        m = T.dot(w.T, new_x)
        h = nnet.sigmoid(m)
        return h

    def grad_desc(cost, theta):
        alpha = 0.1 #learning rate
        return theta - (alpha * T.grad(cost, wrt=theta))

    theta1 = theano.shared(np.array(np.random.rand(3073,3), dtype=theano.config.floatX))
    theta2 = theano.shared(np.array(np.random.rand(4,1), dtype=theano.config.floatX))


    hid1 = layer(x, theta1)

    out1 = T.sum(layer(hid1, theta2))


    xent = -y * T.log(out1) - (1-y) * T.log(1-out1) # Cross-entropy loss function
    xent_f = xent.mean() + 0.01 * (theta2 ** 2).sum()

    cost = theano.function(inputs=[x, y], outputs=xent_f, updates=[
            (theta1, grad_desc(xent_f, theta1)),
            (theta2, grad_desc(xent_f, theta2))])

    run_forward = theano.function(inputs=[x], outputs=out1)

    for i in range(100):
        for k in range(len(Xtr_rows)):
            cur_cost = cost(Xtr_rows[k], Ytr_6[k]) #call our Theano-compiled cost function, it will auto update weights
            print('Cost: %s' % (cur_cost,))

    for i in range(Xtr_rows.shape[0]):
        print "prob: ", run_forward(Xtr_rows[i,:])," real ", Ytr[i]

    print "end"
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