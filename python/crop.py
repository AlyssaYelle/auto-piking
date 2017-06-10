import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
import sys

if __name__ == '__main__':
    unpicked_filename = sys.argv[1]
    picked_filename = sys.argv[2]
    outfile = sys.argv[3]

    unpicked = imread(unpicked_filename, flatten=True)[140:634, 97:1144]
    picked = imread(picked_filename, flatten=True)[140:634, 97:1144]

    diff = picked - unpicked

    labels = np.argmin(diff, axis=0)

    train = (unpicked / 255.).T
    np.savetxt(outfile + '_x.csv', train, delimiter=',')
    np.savetxt(outfile + '_y.csv', labels, delimiter=',')

    plt.plot(labels)
    plt.show()





