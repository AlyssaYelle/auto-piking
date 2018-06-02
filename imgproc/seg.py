import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io 
from skimage import feature
from skimage import restoration
from skimage import img_as_float
from scipy import ndimage
from skimage import morphology

transect_name = 'LSE-GCX0f-Y03a.png.txt'
im = np.loadtxt('radarfigure/data/{}'.format(transect_name))

plt.imshow(im, cmap='gray')
plt.show()
plt.close()

im_float = img_as_float(im)
im_denoised = restoration.nl_means_denoising(im_float, h=0.05)
plt.imshow(im_denoised, cmap='gray')
ax = plt.axis('off')
plt.show()
plt.close()
'''
plt.imshow(im_denoised, cmap='gray')
plt.contour(im_denoised, [0.5], colors='yellow')
plt.contour(im_denoised, [0.45], colors='blue')
ax = plt.axis('off')
plt.show()
plt.close()
'''
# Try to detect edges with Canny filter

edges = feature.canny(im_denoised, sigma=10.2, low_threshold=0.07, \
                      high_threshold=0.8)
plt.imshow(im_denoised, cmap='gray')
plt.contour(edges)
plt.show()
plt.close()









