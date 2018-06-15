import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy import ndimage as ndi
from skimage import feature
from skimage import restoration
from skimage import img_as_float


# reads in transect of interest
def read_im(transect_name):
	return np.loadtxt('radarfigure/data/raw_imgs/{}'.format(transect_name))


# take in array of pixel intensities
# create array of same shape, fill with 0s
# find max pixel intensity of each col and replace with 1
def max_pixel_map(im):
	shape = im.shape
	srf_map = np.zeros((shape[0],shape[1]), dtype=int)
	
	max_pix = np.argmax(im, axis = 0)

	for i in range(len(max_pix)):
		srf_map[max_pix[i]][i] = 1
	
	return srf_map
	




if __name__ == '__main__':
	#transect_name = 'LSE-GCX0f-Y03a.png.txt'
	transect_name = 'LSE-GCX0f-Y153a.png.txt'
	im = read_im(transect_name)

	srf_map = max_pixel_map(im)
	
	plt.imshow(im, cmap='gray')
	plt.contour(srf_map)
	plt.show()
	plt.close()















