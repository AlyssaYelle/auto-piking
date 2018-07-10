'''
just viewing an image to see what it looks like
'''



import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.misc import imread
from skimage import feature
import sys
import os

im = np.load('LSE_GCX0f_X61a_pik1_chan2.npy')
im = im.T
print im.shape



plt.figure(figsize = (13,6))
plt.imshow(im, cmap=plt.cm.gray)
plt.show()
plt.close()








