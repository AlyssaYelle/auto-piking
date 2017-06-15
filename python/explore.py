import matplotlib.pylab as plt
import numpy as np
import sys


def load_dataset(transect_name):
    # Load all features
    all_features = np.loadtxt('data/{}_u2u_bed_x.csv'.format(transect_name), delimiter=',')
    surface = np.loadtxt('data/{}_f2f_srf_y.csv'.format(transect_name), delimiter=',').astype(int)
    bed = np.loadtxt('data/{}_f2f_bed_y.csv'.format(transect_name), delimiter=',').astype(int)

    # # Filter out missing piks
    # mask = all_labels > 0
    # raw_indices = np.arange(len(all_labels))[mask]
    # all_features = all_features[mask]
    # all_labels = all_labels[mask]

    return all_features, surface, bed


def plot_dist_abs(ax, data, max_val):
    means = np.zeros(max_val)
    lower = np.zeros(max_val)
    upper = np.zeros(max_val)
    for i in xrange(max_val):
        means[i] = np.mean([x[i] for x in data if len(x) > i])
        lower[i] = np.percentile([x[i] for x in data if len(x) > i], 95)
        upper[i] = np.percentile([x[i] for x in data if len(x) > i], 5)
    x = np.arange(max_val)
    ax.plot(x, means, color='blue')
    ax.set_xlim([-5, len(x)+5])
    ax.fill_between(x, lower, upper, color='blue', alpha=0.35)

def plot_dist_rel(ax, data, nbins=20):
    means = np.zeros(nbins)
    lower = np.zeros(nbins)
    upper = np.zeros(nbins)
    for i in xrange(nbins):
        xbin = np.zeros(len(data))
        for j,x in enumerate(data):
            start = int(np.round(len(x) * (1./nbins)*i))
            end = int(np.round(len(x) * (1./nbins)*(i+1)))
            xbin[j] = np.mean(x[start:end])
        means[i] = xbin.mean()
        lower[i] = np.percentile(xbin, 95)
        upper[i] = np.percentile(xbin, 5)
    x = np.arange(nbins)
    ax.plot(1. / nbins * x, means, color='blue')
    ax.fill_between(1. / nbins * x, lower, upper, color='blue', alpha=0.35)


if __name__ == '__main__':
    transect_name = sys.argv[1]

    # Parameters for training
    nepochs = 100 # Number of epochs to train for
    batch_size = 10 # Number of observations per batch

    # Load our dataset
    X, surface, bed = load_dataset(transect_name)
    # plt.imshow(X.T, cmap='plasma')
    # plt.show()
    # plt.clf()
    # plt.close()

    # windows = np.zeros((len(X), 100))
    # for i,j in enumerate(y):
    #     target = X[i,j]
    #     for k in xrange(windows.shape[1]):
    #         windows[i,k] = np.max(np.abs(target - X[i,j-k-1:j+k+1]))
    # plt.plot(windows.mean(axis=0))
    # plt.show()
    # plt.clf()
    # plt.close()

    # diffs = np.zeros_like(X)
    # for i in xrange(len(X)):
    #     diffs[i] = np.arange(diffs.shape[1])**2 *  np.array([np.max(np.abs(X[i,j] - X[i,max(0,j-3):min(j+3,X.shape[1])])) for j in xrange(diffs.shape[1])])
    # plt.imshow(diffs.T, cmap='plasma')
    # plt.colorbar()
    # plt.show()

    # Look at differences
    from scipy.ndimage.filters import gaussian_filter1d
    Xsmooth = gaussian_filter1d(X, sigma=1, axis=1)
    Xgrad = Xsmooth[:,:-1] - Xsmooth[:,1:]
    # plt.imshow(Xgrad.T, cmap='plasma')
    # plt.show()
    # plt.clf()
    # plt.close()

    # plt.hist(X[500,80:350], bins=50)
    # # plt.axvline(y[500], ls='--', c='r')
    # plt.show()
    # plt.clf()
    # plt.close()

    air = []
    ice = []
    ground = []
    max_air, max_ice, max_ground = 0, 0, 0
    for x, s, b in zip(X, surface, bed):
        air.append(list(x[:s]))
        ice.append(list(x[s:b]))
        ground.append(list(x[b:]))
        max_air = max(max_air, s)
        max_ice = max(max_ice, b - s)
        max_ground = max(max_ground, len(x) - b)

    fig, axarr = plt.subplots(1,3)
    plot_dist_abs(axarr[0], air, max_air)
    plot_dist_abs(axarr[1], ice, max_ice)
    plot_dist_abs(axarr[2], ground, max_ground)
    # plot_dist_rel(axarr[0], air)
    # plot_dist_rel(axarr[1], ice)
    # plot_dist_rel(axarr[2], ground)
    plt.show()
    plt.clf()
    plt.close()




