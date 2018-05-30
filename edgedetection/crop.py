import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imread
import sys
import os

def get_labels(unpicked, picked):
    diff = picked - unpicked
    labels = np.argmin(diff, axis=0)
    return labels

if __name__ == '__main__':
    for transect_name in os.listdir('screenshots/focused/bed/picked'):
        transect_name = transect_name.replace('.png', '')
        srf_unfoc_unpick = imread('screenshots/unfocused/srf/unpicked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        srf_unfoc_pick = imread('screenshots/unfocused/srf/picked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        srf_foc_unpick = imread('screenshots/focused/srf/unpicked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        srf_foc_pick = imread('screenshots/focused/srf/picked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        bed_unfoc_unpick = imread('screenshots/unfocused/bed/unpicked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        bed_unfoc_pick = imread('screenshots/unfocused/bed/picked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        bed_foc_unpick = imread('screenshots/focused/bed/unpicked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]
        bed_foc_pick = imread('screenshots/focused/bed/picked/{}.png'.format(transect_name), flatten=True)[140:634, 97:1144]

        srf_unfoc_x = 1. - (srf_unfoc_unpick / 255.).T
        srf_foc_x = 1. - (srf_foc_unpick / 255.).T
        bed_unfoc_x = 1. - (bed_unfoc_unpick / 255.).T
        bed_foc_x = 1. - (bed_foc_unpick / 255.).T

        srf_unfoc_y = get_labels(srf_unfoc_unpick, srf_unfoc_pick)
        srf_foc_y = get_labels(srf_foc_unpick, srf_foc_pick)
        bed_unfoc_y = get_labels(bed_unfoc_unpick, bed_unfoc_pick)
        bed_foc_y = get_labels(bed_foc_unpick, bed_foc_pick)
        
        np.savetxt('data/{}_{}_x.txt'.format(transect_name, 'srf_unfoc'), srf_unfoc_x)
        np.savetxt('data/{}_{}_x.txt'.format(transect_name, 'srf_foc'), srf_foc_x)
        np.savetxt('data/{}_{}_x.txt'.format(transect_name, 'bed_unfoc'), bed_unfoc_x)
        np.savetxt('data/{}_{}_x.txt'.format(transect_name, 'bed_foc'), bed_foc_x)

        np.savetxt('data/{}_{}_y.txt'.format(transect_name, 'srf_unfoc'), srf_unfoc_y)
        np.savetxt('data/{}_{}_y.txt'.format(transect_name, 'srf_foc'), srf_foc_y)
        np.savetxt('data/{}_{}_y.txt'.format(transect_name, 'bed_unfoc'), bed_unfoc_y)
        np.savetxt('data/{}_{}_y.txt'.format(transect_name, 'bed_foc'), bed_foc_y)

        # plt.imshow(diff, cmap='gray')
        # plt.show()

        # plt.plot(bed_foc_y)
        # plt.gca().invert_yaxis()
        # plt.show()





