'''
take in list of transects
get data for transects
save
'''


import os
WAIS = os.getenv('WAIS')

import numpy as np
import matplotlib.pyplot as plt 

import sys
sys.path.append('%s/syst/linux/py' % (WAIS))
import radutils




if __name__ == '__main__':

	filepath = 'transects.txt'
	
	f = open(filepath, 'r')
	transects = []
	line = f.readline().rstrip('\n')
	
	while line:
		transects.append(line)
		line = f.readline().rstrip('\n')
	f.close()

	#print transects


	#data = radutils.radutils.load_radar_data('LSE/GCX0f/X61a', 'pik1', 2, None)
	for transect in transects:
		transect_renamed = transect.replace('/', '_')
		print transect_renamed
		data = radutils.radutils.load_radar_data(transect, 'pik1', 2, None)
		np.save('/disk/kea/WAIS/home/wais/alyssa/{}_pik1_chan2.npy'.format(transect_renamed), data)










