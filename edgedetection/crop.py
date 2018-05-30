import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
import sys
import os


if __name__ == '__main__':
    for transect_name in os.listdir('radarfigure/screenshots/'):
        # transect_name = transect_name.replace('.png', '')
        # imread reads an image as an array of pixels
        im = imread('radarfigure/screenshots/{}'.format(transect_name), flatten=True)[67:522, 28:514]


        
        np.savetxt('radarfigure/data/{}.txt'.format(transect_name), im)








