import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.misc import imread
from skimage import feature
import sys
import os



# Test radar image
transect_name = 'LSE-GCX0f-Y03a.png.txt'
im = np.loadtxt('radarfigure/data/{}'.format(transect_name))

# Compute the Canny filter for several values of sigma
edges1 = feature.canny(im, sigma=1)
edges2 = feature.canny(im, sigma=5)
edges3 = feature.canny(im, sigma=10)
edges4 = feature.canny(im, sigma=15)
edges5 = feature.canny(im, sigma=20)

# display results
fig, ((ax11, ax21, ax31), (ax12, ax22, ax32)) = plt.subplots(nrows=2, ncols=3, figsize=(13, 7),
                                    sharex=True, sharey=True)

ax11.imshow(im, cmap=plt.cm.gray)
ax11.axis('off')
ax11.set_title('noisy image', fontsize=10)

ax21.imshow(edges1, cmap=plt.cm.gray)
ax21.axis('off')
ax21.set_title('Canny filter, $\sigma=1$', fontsize=10)

ax31.imshow(edges2, cmap=plt.cm.gray)
ax31.axis('off')
ax31.set_title('Canny filter, $\sigma=5$', fontsize=10)

ax12.imshow(edges3, cmap=plt.cm.gray)
ax12.axis('off')
ax12.set_title('Canny filter, $\sigma=10$', fontsize=10)

ax22.imshow(edges4, cmap=plt.cm.gray)
ax22.axis('off')
ax22.set_title('Canny filter, $\sigma=15$', fontsize=10)

ax32.imshow(edges5, cmap=plt.cm.gray)
ax32.axis('off')
ax32.set_title('Canny filter, $\sigma=20$', fontsize=10)

fig.tight_layout()

plt.show()