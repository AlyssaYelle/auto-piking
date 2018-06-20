import numpy as np

import os
WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set')
import sys
sys.path.append('%s/syst/linux/py' % (WAIS))

from radutils import pickutils

#try:
#    import typing
#    from typing import List, Tuple
#except:
#    pass

# values from $WAIS/orig/pcor/$season/params, for ICP3-ICP6
# I couldn't find it for all seasons ... since this is relativel
# anyways, exact numbers probably don't matter.
# Indexed by channel.
# AUUUGH. The comments and the values don't match in the params file!!!
# channel_offsets = [None, -191.9, -237.9] # <=== what the comments say should be used
# channel_offsets = [None, -182.9, -229.9] # <=== What's actually encoded
channel_offsets = [None, -182.9, -229.9, 0, 0, 0, 0, 0, 0] # <=== modified for field MARFA
# DAY says that this should be power at the receiver.
# There's a 60dB difference between pik1 and focused.
# DAY says should be subtracting -182.9, -229.9 for ICP3
# (reading the file, it's a matter of HiCARS1/HiCARS2)

# TODO: These really should come from the params file used elsewhere...
#main_bang = 155 # From LEL's visual inspection of multiples in TOT/JKB2d/X16a
# TODO: make this a parameter!!
main_bang = 155.6 # From $WAIS/orig/tpro/ICP3/params
main_bang = 100 # from LEL's visual inspection of KRT1 data.

c_air = 299.792458 # speed of light in air, m/us
n_ice = 1.78
c_ice = c_air/n_ice

lambda_air = 5.0 # wavelength in air
# TODO: this should be a parameter, as it varied between inco/coh radar.
kk = 2.e-2  # for 50MHz sampling frequency, convert to us/sample

# TODO: This needs to be combined with calculate_rcoeff, since it shares
# tons of calculations. The trick will be to avoid having each one
# reload the pickfile, which is slow.
def calculate_thickness(pst, # type: str
                        product, # type: str
                        srf_pickfile, # type: str
                        bed_pickfile # type: str
                        ):
    # type: (...) -> Tuple[List[float], List[float]]
    '''
    Calculates the reflection coefficient for input pst, using input pickfile
    names, applied to the specified product.
    '''
    # get picks for srf and bed (srf, low_bed, high_bed), then reprocess
    # with desired channels. (have to do so manually - the max vals that were
    # loaded will be wrong)
    # TODO: This only supports using pik1 for reflection coefficients.
    srf_lowgain_picks = pickutils.load_picks(pst, product, 1, srf_pickfile)
    srf_lowgain_picks.autopick()
    bed_lowgain_picks = pickutils.load_picks(pst, product, 1, bed_pickfile)
    bed_lowgain_picks.autopick()
    bed_highgain_picks = pickutils.load_picks(pst, product, 2, bed_pickfile)
    bed_highgain_picks.autopick()

    # pick file format is fun: (times are in sweeps, values are counts)
    # [vmax_time max_time maxval vmaxval]
    # We just want [time, val]
    srf_filtered = np.array([[row[0], row[3]] if not np.isnan(row[0]) else row[1:3]
                             for row in srf_lowgain_picks.max_vals])
    srf_filtered[:,1] = srf_filtered[:,1] / 1000. + channel_offsets[1]

    bed_low_filtered = np.array([[row[0], row[3]] if not np.isnan(row[0]) else row[1:3]
                                 for row in bed_lowgain_picks.max_vals])
    bed_low_filtered[:,1] = bed_low_filtered[:,1] / 1000. + channel_offsets[1]

    bed_high_filtered = np.array([[row[0], row[3]] if not np.isnan(row[0])
                                  else row[1:3]
                                  for row in bed_highgain_picks.max_vals])
    bed_high_filtered[:,1] = bed_high_filtered[:,1] / 1000. + channel_offsets[2]
    # combine high and low gain to get full dynamic range of bed's rcoeff
    if product == 'lel':
        # NB: This is temporary ... since all params are for 'pik1', and 'lel' has a much higher noise floor / different gains ...
        cutoff = -20
    else:
        cutoff = -60
    bed_filtered = np.array([low if high[1] > cutoff else high for (low, high)
                             in zip(bed_low_filtered, bed_high_filtered)])

    hh = 0.5 * c_air * (srf_filtered[:,0] - main_bang) * kk
    zz = 0.5 * c_ice * (bed_filtered[:,0] - srf_filtered[:,0]) * kk

    good_idxs = np.where(np.isfinite(zz))
    tt = np.arange(0, len(zz))
    return tt[good_idxs], zz[good_idxs]

def calculate_rcoeff(pst, # type: str
                     product, # type: str
                     srf_pickfile, # type: str
                     bed_pickfile, # type: str
                     iceloss_rate # type: float
                    ):
    # type: (...) -> Tuple[List[float], List[float]]
    '''
    Calculates the reflection coefficient for input pst, using input pickfile
    names and iceloss (dB/km), applied to the specified product.

    Has been checked against the values in
    $WAIS/targ/tpro/TOT/JKB2d/X16a/p_echo/ztim_llehr_bedeco.bin
    (using iceloss = 0)
    '''
    # TODO: This needs to gracefully handle being commanded to load picks that aren't there ...
    # 1) modify pickutils.load_picks to test for file existence
    # 2) catch error here and return None?

    # get picks for srf and bed (srf, low_bed, high_bed), then reprocess
    # with desired channels. (have to do so manually - the max vals that were
    # loaded will be wrong)
    # TODO: This only supports using pik1 for reflection coefficients.
    print pst, product, srf_pickfile
    srf_lowgain_picks = pickutils.load_picks(pst, product, 1, srf_pickfile)
    if srf_lowgain_picks is None:
        print "could not load picks for %s, %s, %s" % (pst, product, srf_pickfile)
        return None, None
    srf_lowgain_picks.autopick()
    bed_lowgain_picks = pickutils.load_picks(pst, product, 1, bed_pickfile)
    if bed_lowgain_picks is None:
        print "could not load picks for %s, %s, %s" % (pst, product, bed_pickfile)
        return None, None
    bed_lowgain_picks.autopick()
    bed_highgain_picks = pickutils.load_picks(pst, product, 2, bed_pickfile)
    if bed_highgain_picks is None:
        print "could not load picks for %s, %s, %s" % (pst, product, bed_pickfile)
        return None, None
    bed_highgain_picks.autopick()

    # pick file format is fun: (times are in sweeps, values are counts)
    # [vmax_time max_time maxval vmaxval]
    # We just want [time, val]
    srf_filtered = np.array([[row[0], row[3]] if not np.isnan(row[0]) else row[1:3]
                             for row in srf_lowgain_picks.max_vals])
    srf_filtered[:,1] = srf_filtered[:,1] / 1000. + channel_offsets[1]

    bed_low_filtered = np.array([[row[0], row[3]] if not np.isnan(row[0]) else row[1:3]
                                 for row in bed_lowgain_picks.max_vals])
    bed_low_filtered[:,1] = bed_low_filtered[:,1] / 1000. + channel_offsets[1]

    bed_high_filtered = np.array([[row[0], row[3]] if not np.isnan(row[0])
                                  else row[1:3]
                                  for row in bed_highgain_picks.max_vals])
    bed_high_filtered[:,1] = bed_high_filtered[:,1] / 1000. + channel_offsets[2]
    # combine high and low gain to get full dynamic range of bed's rcoeff
    if product == 'lel':
        # NB: This is temporary ... since all params are for 'pik1', and 'lel' has a much higher noise floor / different gains ...
        cutoff = -20
    else:
        cutoff = -60
    bed_filtered = np.array([low if high[1] > cutoff else high for (low, high)
                             in zip(bed_low_filtered, bed_high_filtered)])

    # Finally, we have raw power received from the bed.
    P_r_dB = bed_filtered[:,1]
    #P_r_dB = bed_high_filtered[:,1] # ONLY FOR DEBUGGING
    #P_r_dB = bed_low_filtered[:,1] # ONLY FOR DEBUGGING

    hh = 0.5 * c_air * (srf_filtered[:,0] - main_bang) * kk
    zz = 0.5 * c_ice * (bed_filtered[:,0] - srf_filtered[:,0]) * kk

    # (The signs of these terms are such that they should be _added_ to P_r)
    # TODO: This is giving ridiculous answers ... even if I ignore hh
    spreading_loss = 20 * np.log10(2*(hh + zz/n_ice))
    # TODO: Is the geometric loss already accounted for in the adjustments DAY gave me?
    geometric_loss = 20 * np.log10(lambda_air / (4*np.pi))
    geometric_loss = -8
    ice_loss = -2 * zz * iceloss_rate / 1000.0
    transmission_loss = 2 * -0.5 # in dB, from Peters2007
    P_t_dB = 10*np.log10(1000*8000.) # P_t = 8000W, from Peters2007, but we need mW for dBm
    P_t_dB = 67 # from $WAIS/code/tpro/ICP7/ALL/p_icethk/params
    misc_loss = -4. # cable, splitter/combiner, chirp power rolloff; not included in DAY's, so I'm leaving it out.
    # 9.0 matches $WAIS/code/tpro/ICP7/ALL/p_icethk/components/step1.make_pik1
    # G_a = 9.4 from Peters2007
    antenna_gain = 9.0

    # transmission_loss => not included in DAY's
    rcoeff = P_r_dB - P_t_dB + spreading_loss - geometric_loss - 2*antenna_gain - ice_loss

    good_idxs = np.where(np.isfinite(rcoeff))
    tt = np.arange(0, len(rcoeff))

    return tt[good_idxs], rcoeff[good_idxs]

def calculate_multiple(pst, # type: str
                       product, # type: str
                       num_srf, # type: int
                       srf_pickfile, # type: str
                       num_bed, # type: int
                       bed_pickfile # type: str
                      ):
    # type: (...) -> Tuple[List[float], List[float]]
    '''
    Calculates the location of the multiple given srf and bed.

    * num_srf - additional bounces between airplane and surface
    * num_bed - additional bounces between srf/bed
    '''
    # get picks for srf and bed (srf, low_bed, high_bed), then reprocess
    # with desired channels.
    # TODO: This only supports using pik1 for reflection coefficients.

    srf_picks = pickutils.load_picks(pst, product, 1, srf_pickfile)
    srf_picks.autopick()

    srf_filtered = np.array([row[0] if not np.isnan(row[0]) else row[1]
                             for row in srf_picks.max_vals])

    multiple = srf_filtered + num_srf * (srf_filtered - main_bang)

    if num_bed > 0:
        bed_picks = pickutils.load_picks(pst, product, 2, bed_pickfile)
        bed_picks.autopick()
        bed_filtered = np.array([row[0] if not np.isnan(row[0]) else row[1]
                                 for row in bed_picks.max_vals])
        multiple += (num_bed + 1) * (bed_filtered - srf_filtered)

    good_idxs = np.where(np.isfinite(multiple))
    tt = np.arange(0, len(multiple))
    return tt[good_idxs], multiple[good_idxs]

def calculate_multiple_error(pst, # type: str
                             product, # type: str
                             num_srf, # type: int
                             srf_pickfile, # type: str
                             num_bed, # type: int
                             bed_pickfile, # type: str
                             mult_pickfile # type: str
                             ):
    # type: (...) -> Tuple[List[float], List[float]]
    '''
    Given input surface / bed / multiple files, returns the difference between
    the predicted multiple position and the actual.

    * num_srf - additional bounces between airplane and surface
    * num_bed - additional bounces between srf/bed
    '''
    # get picks for srf and bed (srf, low_bed, high_bed), then reprocess
    # with desired channels.
    # TODO: This only supports using pik1 for reflection coefficients.

    srf_picks = pickutils.load_picks(pst, product, 1, srf_pickfile)
    srf_picks.autopick()

    srf_filtered = np.array([row[0] if not np.isnan(row[0]) else row[1]
                             for row in srf_picks.max_vals])

    theoretical_multiple = srf_filtered + num_srf * (srf_filtered - main_bang)

    mult_picks = pickutils.load_picks(pst, product, 2, mult_pickfile)
    mult_picks.autopick()
    mult_filtered = np.array([row[0] if not np.isnan(row[0]) else row[1]
                              for row in mult_picks.max_vals])

    if num_bed > 0:
        bed_picks = pickutils.load_picks(pst, product, 2, bed_pickfile)
        bed_picks.autopick()
        bed_filtered = np.array([row[0] if not np.isnan(row[0]) else row[1]
                                 for row in bed_picks.max_vals])
        theoretical_multiple += (num_bed + 1) * (bed_filtered - srf_filtered)

    mult_error = theoretical_multiple - mult_filtered

    good_idxs = np.where(np.isfinite(mult_error))
    tt = np.arange(0, len(mult_error))
    return tt[good_idxs], mult_error[good_idxs]
