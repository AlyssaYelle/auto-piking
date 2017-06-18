import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import sys
import collections
from model import create_model


def load_dataset(transect_name):
    # Load all features
    all_features = np.loadtxt('data/{}_u2u_bed_x.csv'.format(transect_name), delimiter=',')
    bed_labels = np.loadtxt('data/{}_f2f_bed_y.csv'.format(transect_name), delimiter=',').astype(int)
    surface_labels = np.loadtxt('data/{}_f2f_srf_y.csv'.format(transect_name), delimiter=',').astype(int)
    return all_features, surface_labels, bed_labels

def mixture_loss_fn(X):
    weights = tf.Variable(np.ones((3,X.shape[0],1)), dtype=tf.float32)
    surface_params = np.zeros((2, X.shape[0]))
    bed_params = np.zeros((2, X.shape[0]))
    surface_params[0,:] = X.shape[1] / 5.
    bed_params[0,:] = X.shape[1] / 2.
    surface_params = tf.Variable(surface_params, dtype=tf.float32)
    bed_params = tf.Variable(bed_params, dtype=tf.float32)
    npdepths = np.tile(np.arange(X.shape[1]), X.shape[0]).reshape(X.shape).T
    depths = tf.constant(npdepths, tf.float32)
    surface_fn = tf.contrib.distributions.NegativeBinomial(tf.nn.softplus(surface_params[0]), logits=surface_params[1])
    bed_fn = tf.contrib.distributions.NegativeBinomial(tf.nn.softplus(bed_params[0]), logits=bed_params[1])
    srf_probs = tf.transpose(surface_fn.prob(depths))
    bed_probs = tf.transpose(bed_fn.prob(depths))
    Xhat = weights[0] + weights[1] * srf_probs + weights[2] * bed_probs
    fit_loss = tf.reduce_mean((X - Xhat)**2)
    return Xhat, weights, surface_params, bed_params, fit_loss

def plot_transect(transect_name):
    from scipy.misc import imread
    if transect_name == 'wsb':
        unpicked_filename = 'screenshots/unfocused data/bed/unpicked/WSB_JKB1a_AVT01a.png'
    elif transect_name == 'tot':
        unpicked_filename = 'screenshots/unfocused data/bed/unpicked/TOT_JKB2d_X19a.png'
    unpicked = imread(unpicked_filename, flatten=True)[140:634, 97:1144]
    unpicked /= 255.
    return unpicked

def plot_results(sess, transect_name, data, srf_labels, bed_labels, Xhat, surface_params, bed_params):
    unpicked = plot_transect(transect_name)
    fig, axarr = plt.subplots(1,2)
    axarr[0].imshow(unpicked, cmap='gray')
    axarr[1].imshow(Xhat.eval().T, cmap='gray')
    print Xhat.eval()

    # predictions = np.zeros_like(unpicked)
    # predictions[:,:] = np.nan
    # plot_split_results(predictions, labels, sess, model, dataset.test)
    # data = np.array([((p * np.arange(len(p))).sum(), label) for p, label in zip(predictions.T, labels) if not np.isnan(label)])
    # axarr[2].scatter(data[:,0], data[:,1], color='darkgray', alpha=0.7)
    # axarr[2].plot(np.arange(predictions.shape[0]), np.arange(predictions.shape[0]), ls='--', color='red')

    plt.show()

if __name__ == '__main__':
    transect_name = sys.argv[1]

    # Parameters for training
    nsteps = 1000 # Number of steps to train for
    
    # Load our dataset
    data, srf_labels, bed_labels = load_dataset(transect_name)

    # Create a tensorflow session
    sess = tf.InteractiveSession()

    Xhat, weights, surface_params, bed_params, loss = mixture_loss_fn(data)

    # Setup gradient descent
    learning_rate = tf.placeholder(tf.float32, shape=[])
    cur_learning_rate = 0.9
    learning_decay = 0.9995
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = opt.minimize(loss)

    # Initialize tensorflow graph
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()

    best_loss = None
    validation_threshold = None
    validation_losses = []
    step = 0
    steps_since_last_test = 0
    outfile = 'models/mixture'

    # Start learning
    for step in xrange(nsteps):
        feed_dict = {learning_rate: cur_learning_rate}
        sess.run(train_step, feed_dict=feed_dict)
        if step % 1 == 0:
            print('\tStep {0}, step {1}'.format(step, step))
            sys.stdout.flush()
        cur_learning_rate *= learning_decay
        cur_loss = loss.eval()

        # Check if we are improving
        if best_loss is None or cur_loss < best_loss:
            best_loss = cur_loss
            steps_since_improvement = 0
            print('Found new best model. Saving to {}'.format(outfile))
            sys.stdout.flush()
            # saver.save(sess, outfile)
        else:
            steps_since_improvement += 1

        print('Step #{0} Validation loss: {1} Steps since improvement: {2} (learning rate: {3})'.format(step, cur_loss, steps_since_improvement, cur_learning_rate))
        # print 'Params1: {} Params2: {}'.format(surface_params[:,500].eval(), bed_params[:,500].eval())
    # Reset the model back to the best version
    # saver.restore(sess, outfile)

    # Save the validation score for this model
    print('Finished training. Scoring model...')
    sys.stdout.flush()
    
    plot_results(sess, transect_name, data, srf_labels, bed_labels, Xhat, surface_params, bed_params)










