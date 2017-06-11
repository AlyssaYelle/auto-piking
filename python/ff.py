import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import keras.backend as K
import sys
import collections
from model import create_model

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'nfeatures', 'nlabels'])

class Dataset(object):

    def __init__(self,
               features,
               labels,
               batch_size=50,
               seed=42):
        """A basic dataset where each row contains a single feature vector and label
        """
        assert features.shape[0] == labels.shape[0], (
              'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
        self._num_examples = features.shape[0]
        self._features = features
        self._labels = labels
        self._num_features = features.shape[1]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._batch_size = batch_size
        self._rng = np.random.RandomState(seed)
        self.p = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_features(self):
        return self._num_features

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size=None):
        """Return the next `batch_size` examples from this data set."""
        if batch_size is None: batch_size = self._batch_size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]

    def savetxt(self, filename, delimiter=','):
        np.savetxt(filename, np.concatenate((self._features, self._raw_labels), axis=1), delimiter=delimiter)

    def reset(self):
        self.p = 0

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self._batch_size

        # on first iteration permute all data
        if self.p == 0:
            inds = self._rng.permutation(self._features.shape[0])
            self._features = self._features[inds]
            self._labels = self._labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p >= self._features.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        end = min(self.p + n, self._features.shape[0])
        x = self._features[self.p : end]
        y = self._labels[self.p : end]
        self.p += self._batch_size

        return x,y

    def __iter__(self):
        return self

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


def load_dataset(transect_name, val_pct=0.1, test_pct=0.1, batch_size=10):
    # Load all features
    all_features = np.loadtxt('data/{}_x.csv'.format(transect_name), delimiter=',')
    all_labels = np.loadtxt('data/{}_y.csv'.format(transect_name), delimiter=',').astype(int)

    # Filter out missing piks
    all_features = all_features[all_labels > 0]
    all_labels = all_labels[all_labels > 0]

    # Divide into train, validate, and test splits
    indices = np.arange(len(all_labels))
    np.random.shuffle(indices)
    test_size = int(np.round(test_pct * len(all_labels)))
    val_size = int(np.round(val_pct * len(all_labels)))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+val_size]
    train_indices = indices[test_size+val_size:]

    train = Dataset(all_features[train_indices], all_labels[train_indices], batch_size=batch_size)
    validation = Dataset(all_features[val_indices], all_labels[val_indices], batch_size=batch_size)
    test = Dataset(all_features[test_indices], all_labels[test_indices], batch_size=batch_size)

    return Datasets(train=train, validation=validation, test=test,
                    nfeatures=all_features.shape[1],
                    nlabels=all_features.shape[1])


def score_model(sess, model, dataset):
    loss = 0
    for step, (X, y) in enumerate(dataset.validation):
        feed_dict = model.test_dict(X[:,:,np.newaxis], y[:,np.newaxis])
        loss += sess.run(model.test_loss, feed_dict=feed_dict)
    return loss

def explicit_score(sess, model, dataset):
    logprobs = 0
    squared_err = 0
    indices = np.array(list(np.ndindex(model.layer._num_classes)))
    for i in xrange(len(dataset.test.features)):
        feed_dict = model.test_dict(dataset.test.features[i:i+1,:,np.newaxis], dataset.test.labels[i:i+1, np.newaxis])
        if model.density is not None:
            density = sess.run(model.density, feed_dict=feed_dict)[0]
        else:
            density = model.layer.dist(dataset.test.features[i:i+1], sess, feed_dict)[0]
        if np.abs(density.sum() - 1.) > 1e-4:
            raise Exception('Distribution does not add up: {}'.format(density.sum()))
        density /= density.sum()
        if density.min() < 0 or density.max() > 1:
            raise Exception('Distribution outside acceptable bounds: [{}, {}]'.format(density.min(), density.max()))
        logprobs += np.log(density[dataset.test.labels[i]])
        prediction = np.array([density[tuple(idx)] * idx for idx in indices]).sum(axis=0)
        squared_err += np.linalg.norm(dataset.test.labels[i] - prediction)**2
    rmse = np.sqrt(squared_err / float(len(dataset.test.features)))
    print 'Explicit logprobs: {0} RMSE: {1}'.format(logprobs, rmse)
    return logprobs, rmse

if __name__ == '__main__':
    transect_name = sys.argv[1]

    # Parameters for training
    nepochs = 5 # Number of epochs to train for
    batch_size = 10 # Number of observations per batch

    # Load our dataset
    dataset = load_dataset(transect_name, batch_size=batch_size)

    # Create a tensorflow session
    sess = tf.InteractiveSession()

    model = create_model('multinomial', dataset)

    # Setup gradient descent
    learning_rate = tf.placeholder(tf.float32, shape=[])
    cur_learning_rate = 0.9
    learning_decay = 0.995
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = opt.minimize(model.train_loss)

    # Initialize tensorflow graph
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    best_loss = None
    validation_threshold = None
    validation_losses = []
    step = 0
    steps_since_last_test = 0
    outfile = 'models/simple1d'

    # Start learning
    for epoch in xrange(nepochs):
        for step, (X, y) in enumerate(dataset.train):
            feed_dict = model.train_dict(X[:,:,np.newaxis], y[:,np.newaxis])
            feed_dict[learning_rate] = cur_learning_rate
            sess.run(train_step, feed_dict=feed_dict)
            if step % 10 == 0:
                print('\tEpoch {0}, step {1}'.format(epoch, step))
                sys.stdout.flush()
            cur_learning_rate *= learning_decay

        # Test if the model improved on the validation set
        validation_loss = score_model(sess, model, dataset)

        # Check if we are improving
        if best_loss is None or validation_loss < best_loss:
            best_loss = validation_loss
            epochs_since_improvement = 0
            print('Found new best model. Saving to {}'.format(outfile))
            sys.stdout.flush()
            saver.save(sess, 'models/simple1d')
        else:
            epochs_since_improvement += 1

        print('Epoch #{0} Validation loss: {1} Epochs since improvement: {2} (learning rate: {3})'.format(epoch, validation_loss, epochs_since_improvement, cur_learning_rate))
    
    # Reset the model back to the best version
    saver.restore(sess, 'models/simple1d')

    # Save the validation score for this model
    print('Finished training. Scoring model...')
    sys.stdout.flush()
    
    logprobs, rmse = explicit_score(sess, model, dataset)
    np.savetxt(outfile + '_score.csv', [best_loss, logprobs, rmse])











