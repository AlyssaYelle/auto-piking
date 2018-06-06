import sys
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
	return np.loadtxt('radarfigure/data/{}'.format(transect_name))

# denoises image, if necessary
def denoise_im(im):
	im_float = img_as_float(im)
	im_denoised = restoration.nl_means_denoising(im_float, h=0.05)
	return im_denoised

# applies canny edge detection algorithm
def edge_detection(im, sig, thresh1, thresh2):
	edges = feature.canny(im, sigma = sig, low_threshold = thresh1, high_threshold = thresh2)
	return edges


if __name__ == '__main__':
	transect_name = 'LSE-GCX0f-Y03a.png.txt'
	im = read_im(transect_name)

	thresh1 = 0.07
	thresh2 = 0.8

	edges1 = edge_detection(im, 1, thresh1, thresh2)
	edges2 = edge_detection(im, 5, thresh1, thresh2)
	edges3 = edge_detection(im, 10, thresh1, thresh2)
	edges4 = edge_detection(im, 15, thresh1, thresh2)
	edges5 = edge_detection(im, 20, thresh1, thresh2)


	# best result (for this transect, at least)
	plt.imshow(im, cmap='gray')
	plt.contour(edges3)
	plt.show()
	plt.close()

	# results for varying sigmas
	fig, ((ax11, ax21, ax31), (ax12, ax22, ax32)) = plt.subplots(nrows=2, ncols=3, figsize=(13, 7),
                                    sharex=True, sharey=True)

	ax11.imshow(im, cmap=plt.cm.gray)
	ax11.axis('off')
	ax11.set_title('Original image', fontsize=10)

	ax21.imshow(im, cmap=plt.cm.gray)
	ax21.contour(edges1)
	ax21.axis('off')
	ax21.set_title('Canny filter, $\sigma=1$', fontsize=10)

	ax31.imshow(im, cmap=plt.cm.gray)
	ax31.contour(edges2)
	ax31.axis('off')
	ax31.set_title('Canny filter, $\sigma=5$', fontsize=10)

	ax12.imshow(im, cmap=plt.cm.gray)
	ax12.contour(edges3)
	ax12.axis('off')
	ax12.set_title('Canny filter, $\sigma=10$', fontsize=10)

	ax22.imshow(im, cmap=plt.cm.gray)
	ax22.contour(edges4)
	ax22.axis('off')
	ax22.set_title('Canny filter, $\sigma=15$', fontsize=10)

	ax32.imshow(im, cmap=plt.cm.gray)
	ax32.contour(edges5)
	ax32.axis('off')
	ax32.set_title('Canny filter, $\sigma=20$', fontsize=10)

	fig.tight_layout()

	plt.show()
	plt.close()






