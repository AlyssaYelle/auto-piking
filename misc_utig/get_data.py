# use this in WAIS
# gets a memap of whatever transect i want
# can then load it as image or whatever

import os
import numpy as np 
import sys
import radutils
import matplotlib.pyplot as plt 





WAIS = os.getenv('WAIS')
sys.path.append('%s/syst/linux/py' % (WAIS))

data = radutils.radutils.load_radar_data('ASB/JKB1a/R10Wa', 'pik1', 2, None)









