'''
simulates radar image (srf only)
'''



import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import colors as mcolors


# create empty radargram
def rg_init():

	rg = np.zeros((486,455), dtype=int)
	return rg

# simulate srf
# smaller vertical jumps than bed
# srf cannot dip below bed
def sim_srf():
	start = np.random.randint(400,425)
	jumps = [-1,0,0,0,0,0,0,0,0,1]
	return_pix = 0

	rwalk = []
	rwalk.append(start)

	upper_thresh = 440
	lower_thresh = 375

	for i in range(1,486):
		jump = np.random.randint(0,10)
		if rwalk[i-1] != 0:
			next_pix = rwalk[i-1] + jumps[jump]
		else:
			next_pix = return_pix + jumps[jump]
		if next_pix > upper_thresh or next_pix < lower_thresh:
			rwalk.append(0)
			if rwalk[i-1] != 0:
				return_pix = rwalk[i-1]

			
		else:
			rwalk.append(next_pix)
	print rwalk
	return rwalk



def contour_map_assist(srf):
	# takes in bed, srf, converts to 1000x1000 contour array
	rg_srf = rg_init()

	for i in range(455):

		if srf[i] != 0:
			val = srf[i]
			rg_srf[val][i] = 1

	return rg_srf






if __name__ == '__main__':


	srf = sim_srf()
	map = contour_map_assist(srf)

	plt.contour(map, colors = 'blue')

	plt.show()