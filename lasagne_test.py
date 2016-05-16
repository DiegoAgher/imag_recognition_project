import numpy as np
import theano
import theano.tensor as T

import time
import numpy
import lasagne
from unpickle import unpickle



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



def load_dataset():
    batch_size = 500
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
    X_train = numpy.concatenate((train_set_1, train_set_2, train_set_3, train_set_4, train_set_5), axis=0)

    y_train = numpy.concatenate((data_batch_1["labels"], data_batch_2["labels"], data_batch_3["labels"],
                                 data_batch_4["labels"], data_batch_5["labels"]))

    test_set = test["data"]
    Xte_rows = test_set.reshape(train_set_1.shape[0], 32 * 32 * 3)
    Yte = numpy.asarray(test["labels"])

    Xval_rows = X_train[:7500, :]  # take first 1000 for validation
    Yval = y_train[:7500]
    Xtr_rows = X_train[7500:50000, :]  # keep last 49,000 for train
    Ytr = y_train[7500:50000]

    mean_train = Xtr_rows.mean(axis=0)
    stdv_train = Xte_rows.std(axis=0)
    Xtr_rows = (Xtr_rows - mean_train) / stdv_train
    Xval_rows = (Xval_rows - mean_train) / stdv_train
    Xte_rows = (Xte_rows - mean_train) / stdv_train
    train_set = (Xtr_rows, Ytr)
    valid_set = (Xval_rows, Yval)
    test_set = (Xte_rows, Yte)

    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # train_set_x, train_set_y = shared_dataset(train_set)
    # datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
    #             (test_set_x, test_set_y)]
    #
    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    return (Xtr_rows, Ytr, Xval_rows , Yval, Xte_rows, Yte)


def build_ImgNet(input_var=None):

    # network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
    #                                     input_var=input_var)
    # # This time we do not apply input dropout, as it tends to work less well
    # # for convolutional layers.
    #
    # # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # # convolutions are supported as well; see the docstring.
    # network = lasagne.layers.Conv2DLayer(
    #         network, num_filters=32, filter_size=(5, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         W=lasagne.init.GlorotUniform())
    # # Expert note: Lasagne provides alternative convolutional layers that
    # # override Theano's choice of which implementation to use; for details
    # # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
    #
    # # Max-pooling layer of factor 2 in both dimensions:
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    #
    # # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    # network = lasagne.layers.Conv2DLayer(
    #         network, num_filters=32, filter_size=(5, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    #
    # # A fully-connected layer of 256 units with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=.5),
    #         num_units=256,
    #         nonlinearity=lasagne.nonlinearities.rectify)
    #
    # # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=.5),
    #         num_units=10,
    #         nonlinearity=lasagne.nonlinearities.softmax)
    #
    # return network
    l_in = lasagne.layers.InputLayer(shape=(500, 3, 32, 32),
                                     input_var=input_var)
    conv1 = lasagne.layers.Conv2DLayer(
        l_in, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    max_pool1 = lasagne.layers.MaxPool2DLayer(conv1, pool_size=(2, 2))

    conv2 = lasagne.layers.Conv2DLayer(
        max_pool1, num_filters=80, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    max_pool2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))

    l_drop1 = lasagne.layers.DropoutLayer(max_pool2, p=0.2)

    conv3 = lasagne.layers.Conv2DLayer(
        l_drop1, num_filters=200, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    Chape = lasagne.layers.get_output_shape(conv3)

    print "shape: ", Chape

    #max_pool3 = lasagne.layers.MaxPool2DLayer(conv3, pool_size=(2, 2))
    #
    # MLP = lasagne.layers.DenseLayer(
    #     lasagne.layers.dropout(max_pool3, p=.5),
    #     num_units=256,
    #     nonlinearity=lasagne.nonlinearities.rectify)

    final = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(conv3, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return final


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(model='Imgnet', num_epochs=100):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()



    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')


    network = build_ImgNet(input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.15, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.15, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            inputs = inputs.reshape((500,3,32,32))
            targets = targets.astype(np.int32)
            train_err += train_fn(inputs, targets)
            train_batches += 1


        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            inputs = inputs.reshape((500,3,32,32))
            targets = targets.astype(np.int32)
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        inputs = inputs.reshape((500,3,32,32))
        targets = targets.astype(np.int32)
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)



main(model='Imgnet')