import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy import ndimage as ndi
from skimage import feature
from skimage import restoration
from skimage import img_as_float
from scipy.stats import norm


# reads in transect of interest
def read_picked_im(transect_name):
	return np.loadtxt('radarfigure/data/srf_picked/{}'.format(transect_name))

def read_unpicked_im(transect_name):
	return np.loadtxt('radarfigure/data/raw_imgs/{}'.format(transect_name))


# take in array of pixel intensities
# representing img "picked" to find srf
# return srf position
def srf_pos(picked_im, unpicked_im):
	ar = np.not_equal(picked_im, unpicked_im)
	ar_t = ar.transpose()


	shape = ar_t.shape


	pos = []
	for row in range(shape[0]):
		for col in range(shape[1]):
			if ar_t[row][col] == True and len(pos) <= row:
				pos.append(col)


	return pos




	
def srf_jumps(picked_im, unpicked_im):
	y = srf_pos(picked_im,unpicked_im)
	jumps = []
	for i in range(1,len(y)):
		jump = y[i] - y[i-1]
		jumps.append(jump)
	return jumps



if __name__ == '__main__':
	#transect_name = 'LSE-GCX0f-Y27a.png.txt'
	transect_name = 'LSE-GCX0f-Y165a.png.txt'
	a = read_picked_im(transect_name)
	b = read_unpicked_im(transect_name)
	pos = srf_pos(a,b)


	jumps = srf_jumps(a,b)
	#print jumps

	plt.hist(jumps)
	plt.show()
	plt.close()









