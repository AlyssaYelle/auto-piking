'''
simulates radar image
'''



import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import colors as mcolors


# create empty radargram
def rg_init():

	rg = np.zeros((1000,1000), dtype=int)
	return rg

# simulate srf
# smaller vertical jumps than bed
# srf cannot dip below bed
def sim_srf(bed):
	start = np.random.randint(800,900)
	jumps = [-3,-2,-1,-1,0,0,1,1,2,3]
	return_pix = 0

	rwalk = []
	rwalk.append(start)

	upper_thresh = 950
	lower_thresh = 700

	for i in range(1,1000):
		jump = np.random.randint(0,10)
		if rwalk[i-1] != 0:
			next_pix = rwalk[i-1] + jumps[jump]
		else:
			next_pix = return_pix + jumps[jump]
		if next_pix > upper_thresh or next_pix < lower_thresh:
			rwalk.append(0)
			if rwalk[i-1] != 0:
				return_pix = rwalk[i-1]
		elif next_pix < bed[i]:
			rwalk.append(bed[i])
			
		else:
			rwalk.append(next_pix)
	return rwalk



# simulate bed 
# assume some distribution of vertical jumps from col to col
# radar can't penetrate ice deeper than some thresh
def sim_bed():
	start = np.random.randint(100,900)
	jumps = [-15,-10,-5,-2,-1,1,2,5,10,15]
	return_pix = 0

	rwalk = []
	rwalk.append(start)

	upper_thresh = 950
	lower_thresh = 50

	for i in range(1,1000):
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
	return rwalk


def contour_map_assist(bed, srf):
	# takes in bed, srf, converts to 1000x1000 contour array
	rg_srf = rg_init()
	rg_bed = rg_init()
	for i in range(1000):
		if bed[i] != 0:
			rg_bed[bed[i]][i] = 1
		if srf[i] != 0:
			rg_srf[srf[i]][i] = 1

	return (rg_srf,rg_bed)






if __name__ == '__main__':

	bed = sim_bed()
	srf = sim_srf(bed)
	map = contour_map_assist(bed, srf)

	plt.contour(map[0], colors = 'blue')
	plt.contour(map[1])
	plt.show()
















