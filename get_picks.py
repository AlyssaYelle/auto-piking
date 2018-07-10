import os
WAIS = os.getenv('WAIS')

import numpy as np
import matplotlib.pyplot as plt 

import sys
sys.path.append('%s/syst/linux/py' % (WAIS))
import radutils


pick_dict = radutils.pickutils.load_all_picks('LSE/GCX0f/Y03a')

'''
pick_dict returns
{'pik1.chan2.test_adj': <radutils.pickutils.Picks object at 0x7fa23d172a10>, 
'pik1.chan1.srf_adj': <radutils.pickutils.Picks object at 0x7fa23d172a90>, 
'pik1.chan2.bed_adj': <radutils.pickutils.Picks object at 0x7fa23d172990>, 
'pik1.chan2.test2_adj': <radutils.pickutils.Picks object at 0x7fa23d1729d0>}

dir(pick_dict['pik1.chan2.test_adj'])
returns
['__class__', '__delattr__', '__dict__', '__doc__', '__format__', 
'__getattribute__', '__hash__', '__init__', '__module__', '__new__', 
'__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', 
'__str__', '__subclasshook__', '__weakref__', 'add_pair', 'add_pick', 
'autopick', 'bottom', 'channel', 'close_pick', 'delete_pick', 'delete_picks', 
'filepath', 'get_sweeps', 'label', 'max_vals', 'product', 'pst', 'top']
'''

srf = pick_dict['pik1.chan2.test_adj'].max_vals
'''
the second column of this is what the autopicker determines is the bed/srf
'''

top_bound = pick_dict['pik1.chan2.test_adj'].top
lower_bound = pick_dict['pik1.chan2.test_adj'].bottom
'''
return array of [x_loc, y_loc]
'''




# autopick
'''
    def autopick(self):
        # type: () -> None
        if self.top is None or self.bottom is None:
            return

        num_samples = radutils.params.get_radar_samples(self.pst)
        datafile = radutils.get_radar_filename(self.pst, self.product,
                                                        self.channel)

        tmp_bounds_filename = '/tmp/deva.bounds.%r' % (os.getpid())
        tmp_max_filename = '/tmp/deva.max.%r' % (os.getpid())
        with open(tmp_bounds_filename, 'w') as fp:
            if self.top is not None:
                for pick in self.top[self.top[:,0].argsort()]:
                    outline = 'L\t%d\t%0.6f\n' % (int(round(pick[0])), pick[1])
                    fp.write(outline)
            if self.bottom is not None:
                for pick in self.bottom[self.bottom[:,0].argsort()]:
                    outline = 'U\t%d\t%0.6f\n' % (int(round(pick[0])), pick[1])
                    fp.write(outline)

        pk_cmd = get_autopicker(self.pst)
        with open(tmp_bounds_filename, 'r') as infile, open(tmp_max_filename, 'w') as outfile:
            try:
                subprocess.check_call([pk_cmd, str(num_samples), str(0),
                                       str(num_samples), datafile],
                                      stdin=infile, stdout=outfile)
            except subprocess.CalledProcessError as ex:
                print "autopick failed!", str(ex)
                # TODO: Do we want this to propagate the error up?
                return

        pick_max = []
        with open(tmp_max_filename, 'r') as fp:
            for line in fp:
                tokens = line.rstrip().split()
                if tokens[0] != 'P':
                    raise Exception('Unexpected format for generated datafile! %r' % (line))
                else:
                    floats = [[np.NAN if val == 'x' else float(val) for val in tokens[1:]]]
                    pick_max.extend(floats)

        self.max_vals = np.array(pick_max)

'''









