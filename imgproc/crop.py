'''
This is a simple script to crop screenshots of radar images
and convert them to an array of pixel intensities
'''



import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
from PIL import Image
import sys
import os
import os.path

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


if __name__ == '__main__':
    path = 'radarfigure/screenshots'
    for transect_name in mylistdir(path):
        transect_renamed = transect_name.replace(':', '-')
        os.rename(os.path.join(path, transect_name), os.path.join(path, transect_renamed))
        # imread reads an image as an array of pixels
        im = imread('radarfigure/screenshots/{}'.format(transect_renamed), flatten=True)[67:522, 43:529]


        
        np.savetxt('radarfigure/data/srf_picked/{}.txt'.format(transect_renamed), im)








