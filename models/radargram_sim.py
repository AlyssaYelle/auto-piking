'''
random processes to simulate a radargram
work in progress
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
def sim_srf(p, q, r, start):
	ul_stay = 1-p
	ll_stay = r
	srf = rg_init()
	srf[start][0] = 1
	depth = start

	for i in range(1,1000):

		state = np.random.uniform(0,1)
		if state < ll_stay:
			depth -= 1
		elif ll_stay < state < ul_stay:
			depth = depth
		elif state > ul_stay:
			depth += 1
		if depth < 975:
			srf[depth][i] = 1
	return srf



# markov chain
def markov_chain(mtx):
	pass


# simulate bed with markov chain + random walk in state(no lake)
def sim_bed(p,q,r,start,rg):
	ul_stay = 1-p
	ll_stay = r
	rg[start][0] = 1
	depth = start

	for i in range(1,1000):

		state = np.random.uniform(0,1)
		if state < ll_stay:
			depth -= 7
		elif ll_stay < state < ul_stay:
			depth = depth
		elif state > ul_stay:
			depth += 7
		rg[depth][i] = 1
	return rg




if __name__ == '__main__':
	rg = rg_init()

	srf = sim_srf(.1,.8,.1,900)
	bed_start = np.random.randint(100,900)
	map = sim_bed(.5,.01,.49,bed_start,rg)

	plt.contour(map)
	plt.contour(srf, colors = 'blue')
	plt.show()
















