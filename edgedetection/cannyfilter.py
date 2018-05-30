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

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma=10)
edges2 = feature.canny(im, sigma=20)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=10$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=20$', fontsize=20)

fig.tight_layout()

plt.show()