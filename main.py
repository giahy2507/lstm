import theano
import theano.tensor as T
from theano import config
import numpy as np
from lasagne.updates import adam
import imdb


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(config.floatX)

def tanh(x):
    return T.tanh(x)

def linear(x):
    return x

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def hysigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def hytanh(x):
    return np.tanh(x)



class ProjectionLayer(object):
    def __init__(self, input, vocab_size, embsize, input_shape):
        self.words_embedding = theano.shared(value= norm_weight(vocab_size,embsize),
                                             name = "wordembedding",
                                             borrow=True)
        self.input = input
        self.output_shape = (input.shape[0],input.shape[1],embsize)
        self.output = self.words_embedding[input.flatten()].reshape(self.output_shape)
        self.params = [self.words_embedding]
        self.L2 = (self.words_embedding**2).sum()

class LSTM(object):


    def __init__(self, input, mask , embdim, hiddendim, name = "LSTM"):
        """
        Params:
            input: state below
            input_shape: shape of state below (n_step, batch_size, emb_size)
        """
        self.input = input
        self.mask = mask
        self.embdim = embdim
        self.hiddendim = hiddendim
        self.name = name
        self.init_params()


    def init_params(self):
        W = np.concatenate([norm_weight(self.embdim, self.hiddendim),
                           norm_weight(self.embdim, self.hiddendim),
                           norm_weight(self.embdim, self.hiddendim),
                           norm_weight(self.embdim, self.hiddendim)], axis=1)
        self.W = theano.shared(value=W, name=self.name + "W", borrow=True)

        U = np.concatenate([norm_weight(self.hiddendim, self.hiddendim),
                           norm_weight(self.hiddendim, self.hiddendim),
                           norm_weight(self.hiddendim, self.hiddendim),
                           norm_weight(self.hiddendim, self.hiddendim)], axis=1)
        self.U = theano.shared(value=U, name=self.name + "U", borrow=True)
        b = np.zeros((4 * self.hiddendim,))
        self.b = theano.shared(value=b, name=self.name + "b", borrow=True)
        self.params = [self.W, self.U, self.b]

    @staticmethod
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]


    def _step(self, m_, x_, h_, c_):
        preact = T.dot(h_, self.U)
        preact += x_
        i = T.nnet.sigmoid(LSTM._slice(preact, 0, self.hiddendim))
        f = T.nnet.sigmoid(LSTM._slice(preact, 1, self.hiddendim))
        o = T.nnet.sigmoid(LSTM._slice(preact, 2, self.hiddendim))
        c = T.tanh(LSTM._slice(preact, 3, self.hiddendim))
        c = f * c_ + i * c

        # unroll last state (h,c) until nstep with mask
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    def build_lstm(self):
        nsteps = self.input.shape[0]
        n_samples = self.input.shape[1]
        assert self.mask is not None

        # (nstep, minibatch_size, embsize)
        self.state_below = (T.dot(self.input, self.W) + self.b)

        # (nstep, minibatch_size, 4*hiddensize)
        rval, updates = theano.scan(self._step,
                                sequences=[self.mask, self.state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),n_samples,self.hiddendim),
                                              T.alloc(numpy_floatX(0.),n_samples,self.hiddendim)],
                                name=self.name,
                                n_steps=nsteps)
        # rval have 2 matrix: (h, c)
        # [h = (nstep, minibatch_size, hiddensize) , c =(nstep, minibatch_size, hiddensize)]
        self.output = rval[0]

class SoftmaxLayer(object):

    def __init__(self, input , n_in, n_out):
        self.W = theano.shared(value= np.asarray(np.random.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=config.floatX),
                               name = "W",
                               borrow=True)
        self.b = theano.shared(value= np.asarray(np.random.rand(n_out,) ,dtype=config.floatX),
                               name ="b",
                               borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.input = input
        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def build_model(vocab_size = 200,
                embsize = 25,
                hiddensize = 50, ydim =2 ):

    X = T.matrix('X', dtype='int64')
    Mask = T.matrix('mask', dtype=config.floatX)
    Y = T.vector('Y', dtype='int64')

    nstep = X.shape[0]
    mini_batch_size = X.shape[1]

    emblayer = ProjectionLayer(X,vocab_size,embsize,(nstep,mini_batch_size))

    lstmlayer = LSTM(emblayer.output,Mask,embsize,hiddensize,name ="lstm-encode")
    lstmlayer.build_lstm()
    proj = lstmlayer.output

    proj = (proj * Mask[:, :, None]).sum(axis=0)
    proj = proj / Mask.sum(axis=0)[:, None]

    softmax_layer = SoftmaxLayer(proj,hiddensize,ydim)

    cost = softmax_layer.negative_log_likelihood(Y)

    err = softmax_layer.errors(Y)

    params = emblayer.params + lstmlayer.params + softmax_layer.params

    updates = adam(cost,params)

    train_function = theano.function(inputs=[X,Mask,Y], outputs=[cost, err],updates=updates)
    valid_function = theano.function(inputs=[X,Mask,Y], outputs=[cost, err])
    predict_function = theano.function(inputs=[X,Mask], outputs= softmax_layer.y_pred)

    # see_func = theano.function(inputs=[X,Mask], outputs=softmax_layer.p_y_given_x)
    # hyhy = see_func(data_X,mask_X)
    # print hyhy.shape


    return X,Mask,Y,cost,err, train_function, valid_function, predict_function


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


if __name__ == "__main__":

    vocab_size = 10000
    embsize = 128
    hiddensize = 128
    nstep = 100
    mini_batch_size = 16
    max_epochs = 20
    np.random.seed(4488)


    print('Loading data')
    train, valid, test = imdb.load_data(n_words=10000, valid_portion=0.05,maxlen=100)
    train = test
    ydim = np.max(train[1]) + 1

    print('Build model ')
    X,Mask,Y,\
    cost,err, \
    train_function, valid_function, predict_function = build_model(vocab_size=vocab_size,
                                                                    embsize=embsize,
                                                                    hiddensize=hiddensize)

    print('Training ')
    for eidx in range(max_epochs):
        kf = get_minibatches_idx(len(train[0]), mini_batch_size, shuffle=True)
        costs = []
        errs = []
        for _, train_index in kf:
            # Select the random examples for this minibatch
            y = [train[1][t] for t in train_index]
            x = [train[0][t]for t in train_index]

            # Get the data in numpy.ndarray format
            # This swap the axis!
            # Return something of shape (minibatch maxlen, n samples)
            x, mask, y = imdb.prepare_data(x, y)
            cost, err = train_function(x, mask, y)
            if np.isnan(cost) or np.isnan(err):
                continue
            costs.append(float(cost))
            errs.append(float(err))
        costs = np.array(costs)
        errs = np.array(errs)
        print "Epoch {0}: Cost {1} Err {2}".format(eidx, np.mean(costs), np.mean(errs))




