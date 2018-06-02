'''
This is a simple script to crop screenshots of radar images
and convert them to an array of pixel intensities
'''



import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
import sys
import os
import os.path


if __name__ == '__main__':
    path = 'radarfigure/screenshots'
    for transect_name in os.listdir(path):
        transect_renamed = transect_name.replace(':', '-')
        os.rename(os.path.join(path, transect_name), os.path.join(path, transect_renamed))
        # imread reads an image as an array of pixels
        im = imread('radarfigure/screenshots/{}'.format(transect_renamed), flatten=True)[67:522, 43:529]


        
        np.savetxt('radarfigure/data/{}.txt'.format(transect_renamed), im)








