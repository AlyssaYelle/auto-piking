#! /usr/bin/env python2.7

import subprocess
import sys

import radarUtilities

if __name__=="__main__":
    #all_seasons = ['2001', 'ASE1', 'CHA1', 'ICP1', 'ICP2',
    #               'ICP3', 'ICP4', 'ICP5', 'ICP6', 'ICP7']
    all_seasons = ['ICP3']
    for season in all_seasons:
        tmp = subprocess.check_output(['get_psts', season])
        psts = tmp.split('\n')
        for pst in psts:
            if pst == '':
                continue
            try:
                rtc = radarUtilities.RadarTimeConverter(pst)
                str_posix = 'o' if rtc.posix is None else 'x'
                str_1m = 'o' if rtc.traces_1m is None else 'x'
                outstr = '\t'.join([pst, str(rtc.num_traces), str_posix, str_1m])
                print outstr
                sys.stdout.flush()
            except radarUtilities.TimeConversionError as ex:
                print ex
