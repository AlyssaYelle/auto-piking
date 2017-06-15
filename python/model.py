import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import Dense, Dropout, Flatten
from tfsdp.utils import ints_to_multinomials
from tfsdp.models import MultinomialLayer, \
                         DiscreteParametricMixtureLayer, \
                         LocallySmoothedMultiscaleLayer, \
                         DiscreteLogisticMixtureLayer, \
                         ScalableLocallySmoothedMultiscaleLayer

class Model(object):
    def __init__(self, layer, x=None, density=None, labels=None,
                       train_loss=None, test_loss=None,
                       train_dict={}, test_dict={}, fit_dict={}):
        self._layer = layer
        self._x = x
        self._density = density
        self._labels = labels
        self._train_loss = train_loss
        self._test_loss = test_loss
        self._train_dict = train_dict
        self._test_dict = test_dict
        self._fit_dict = fit_dict
        self._bins = [np.arange(c) for c in layer._num_classes]

    @property
    def layer(self):
        return self._layer

    @property
    def x(self):
        return self._x

    @property
    def density(self):
        return self._density

    @property
    def labels(self):
        return self._labels

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def test_loss(self):
        return self._test_loss

    def train_dict(self, x, labels):
        self._train_dict[self._x] = x
        if isinstance(self.layer, MultinomialLayer):
            self._train_dict[self._labels] = labels
        elif isinstance(self.layer, LocallySmoothedMultiscaleLayer) or \
             isinstance(self.layer, ScalableLocallySmoothedMultiscaleLayer):
            self.layer.fill_train_dict(self._train_dict, labels)
        else:
            self._train_dict[self._labels] = labels
        self._train_dict[K.learning_phase()] = 1
        return self._train_dict

    def test_dict(self, x, labels):
        self._test_dict[self._x] = x
        if isinstance(self.layer, MultinomialLayer):
            self._test_dict[self._labels] = labels
        elif isinstance(self.layer, LocallySmoothedMultiscaleLayer) or \
             isinstance(self.layer, ScalableLocallySmoothedMultiscaleLayer):
            self.layer.fill_test_dict(self._test_dict, labels)
        else:
            self._test_dict[self._labels] = labels
        self._test_dict[K.learning_phase()] = 0
        return self._test_dict

    def fit_dict(self, x):
        self._test_dict[self._x] = x
        self._test_dict[K.learning_phase()] = 0
        return self._test_dict

def create_model(model, dataset, variable_scope='simple-1d-conv', **kwargs):
    with tf.variable_scope(variable_scope):
        X = tf.placeholder(tf.float32, [None,dataset.nfeatures,1], name='X')
        # conv1 = Conv1D(32, kernel_size=5, strides=1, activation='relu', kernel_regularizer=l2(0.01),
        #                                 input_shape=(None, dataset.nfeatures,1))(X)
        # pool1 = MaxPooling1D(pool_size=2)(conv1)
        # drop1 = Dropout(0.5)(pool1)
        # conv2 = Conv1D(64, kernel_size=5, strides=1, activation='relu', kernel_regularizer=l2(0.01))(drop1)
        # pool2 = MaxPooling1D(pool_size=2)(conv2)
        # drop2 = Dropout(0.5)(pool2)
        # drop2_flat = tf.reshape(drop2, [-1, 120*64])
        # hidden1 = Dense(128, kernel_regularizer=l2(0.01), activation='relu')(drop2_flat)
        # drop_h1 = Dropout(0.5)(hidden1)
        # hidden2 = Dense(64, kernel_regularizer=l2(0.01), activation='relu')(drop_h1)
        # drop_h2 = Dropout(0.5)(hidden2)
        # hidden3 = Dense(32, kernel_regularizer=l2(0.01), activation='relu')(drop_h2)
        # drop_h3 = Dropout(0.5)(hidden3)
        # input_size = 32
        X_flat = tf.reshape(X, [-1, dataset.nfeatures]) # TEMP
        drop_h3 = X_flat
        input_size = dataset.nfeatures

        if model == 'multinomial':
            dist_model = MultinomialLayer(drop_h3, input_size, dataset.nlabels, one_hot=False, **kwargs)
        elif model == 'gmm':
            dist_model = DiscreteParametricMixtureLayer(drop_h3, input_size, dataset.nlabels, one_hot=False, **kwargs)
        elif model == 'lmm':
            dist_model = DiscreteLogisticMixtureLayer(drop_h3, input_size, dataset.nlabels, one_hot=False, **kwargs)
        elif model == 'sdp':
            dist_model = LocallySmoothedMultiscaleLayer(drop_h3, input_size, dataset.nlabels, one_hot=False, **kwargs)
        elif model == 'fast-sdp':
            dist_model = ScalableLocallySmoothedMultiscaleLayer(drop_h3, input_size, dataset.nlabels, one_hot=False, **kwargs)
        else:
            raise Exception('Unknown model type: {0}'.format(model))

        return Model(dist_model, x=X, density=dist_model.density, labels=dist_model.labels,
                       train_loss=dist_model.train_loss, test_loss=dist_model.test_loss)

