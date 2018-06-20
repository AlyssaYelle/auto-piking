import gdal
import numpy as np
import os
import pickle
import sys
import time

import matplotlib.collections as mpc
import matplotlib.patches as mpp

import devaParameters as dp
import plotUtilities

WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set')
sys.path.append('%s/syst/linux/py' % (WAIS))
import waisutils
import waisutils.season
import plotutils
import plotutils.basemap

try:
    import typing
    from typing import Any, Dict, Optional
    if typing.TYPE_CHECKING:
        import matplotlib
except:
    pass


###########################
# Loading in PSTs and GLAS lines.

def load_transects(antarctic=True):
    # type: (bool) -> Dict[str, np.ndarray]
    # Nx4 array of txyz
    '''
    Loads transects found in the hard-coded directory.
    Filters which projects to load based on whether they're in the arctic
    or the antarctic.

    The pickle'd data is created manually, in ipython:
    import os
    WAIS = os.getenv('WAIS')
    import devaUtilities
    import pickle
    transects = devaUtilities.load_transects()
    pickle.dump(transects, open('%s/targ/xtra/ALL/deva/psts.pickle' % WAIS, 'w'))
    '''
    transects = {} # type: Dict[str, np.ndarray]
    pst_dir = WAIS+'/targ/xtra/ALL/deva/psts'
    pst_cache = WAIS+'/targ/xtra/ALL/deva/psts.pickle'

    # If we have a cached set of psts, that's faster!
    if os.path.isfile(pst_cache):
        with open(pst_cache, 'r') as fp:
            all_transects = pickle.load(fp)
        for pst in all_transects.keys():
            season, _ = waisutils.season.SEASON_LOOKUP.get_season(pst)
            if season is None:
                # print "No season found for PST: %r (filename: %r)" % (pst, filename)
                continue
            if antarctic and waisutils.season.season_is_northern(season):
                continue
            elif not antarctic and not waisutils.season.season_is_northern(season):
                continue
            transects[pst] = all_transects[pst]
    else: # have to load 'em all now...
        print "Could not find psts.pickle; loading from text files."
        for filename in os.listdir(pst_dir):
            if not os.path.isfile(pst_dir + '/' + filename):
                continue
            pst = filename.replace('.txyz','').replace('.','/')
            # Ignore Greenland data for now.
            # TODO: go back and load greenland/antarctica/canada data separately?
            season, _ = waisutils.season.SEASON_LOOKUP.get_season(pst)
            if season is None:
                print "No season found for PST: %r (filename: %r)" % (pst, filename)
                continue
            if antarctic and waisutils.season.season_is_northern(season):
                continue
            elif not antarctic and not waisutils.season.season_is_northern(season):
                continue

            data = np.genfromtxt(pst_dir + '/' + filename)
            # Want at least 4 points ... each point has t,x,y,z in data
            if len(data) <= 16:
                # print "%r was empty!" % (filename)
                continue
            transects[pst] = data

    return transects

def load_glas_lines():
    # type: () -> Dict[int, np.ndarray]
    # Nx2 array
    # TODO: This argument type is inconsistent with the flights/transects keyed with a string
    '''
    Loads _all_ the glas lines found in the hard coded directory:
    '''
    goodlines = np.genfromtxt(WAIS + '/targ/xtra/ALL/deva/good_glas_lines')

    glas_lines = {} # type: Dict[int, np.ndarray]
    glas_dir = WAIS + '/targ/xtra/ALL/deva/glas'
    for filename in os.listdir(glas_dir):
        fullname = glas_dir + '/' + filename
        if not os.path.isfile(fullname):
            continue
        line_number = int(filename.rstrip('.xy'))
        if line_number in goodlines:
            data = np.genfromtxt(fullname)
            glas_lines[line_number] = data
    return glas_lines

# TODO: Extend this to handle other seasons
# TODO: I ALSO want to load and plot the PCL data quality info. zorder should be:
#  * flight, in thin black
#  * tof0 in crosses
#  * tof1 in dots
#  * camera cursor, in larger maroon ???
def load_flight_lines():
    # type: () -> Dict[str, np.ndarray]
    '''
    Loads _all_ the flights found in the old flight visualizer dir
    '''

    flight_lines = {} # type: Dict[str, np.ndarray]
    flight_dir = WAIS + '/home/wais/lindzey/data/flight_visualizer/plane_traces'
    for filename in os.listdir(flight_dir):
        # Don't handle non-ICECAP data yet
        if 'ICP' not in filename:
            continue
        fullname = flight_dir + '/' + filename
        if not os.path.isfile(fullname):
            continue
        season = filename.split('.')[0]
        flight = filename.split('.')[1]
        data = np.genfromtxt(fullname)
        flight_lines['%s/%s' % (season, flight)] = data
    return flight_lines

def load_pcl_quality():
    # type: () -> Dict[str, np.ndarray]
    '''
    Loads all the PCL info found in old flight_
    '''
    pcl_lines = {} # type: Dict[str, np.ndarray]
    qc_dir = '%s/home/wais/lindzey/data/QC_lidar' % (WAIS)
    for season in os.listdir(qc_dir):
        qc_season_dir = '%s/%s' % (qc_dir, season)
        if os.path.isdir(qc_season_dir):
            for flight in os.listdir(qc_season_dir):
                qc_flight_dir = '%s/%s' % (qc_season_dir, flight)
                if os.path.isdir(qc_flight_dir):
                    for filename in os.listdir(qc_flight_dir):
                        qc_card_file = '%s/%s' % (qc_flight_dir, filename)
                        if (os.path.isfile(qc_card_file)
                            and 'georef.txt' in filename
                            and os.stat(qc_card_file).st_size > 0):
                            card = filename.split('.')[0]
                            label = '%s/%s/%s' % (season, flight, card)
                            data = np.loadtxt(qc_card_file)
                            if data.ndim == 2 and data.shape[1] == 7:
                                pcl_lines[label] = data
    return pcl_lines


##############################
# GL-specific stuff.

def make_grounding_line_dict():
    # type: () -> Dict[str, Optional[Any]]
    # TODO: This is tricky, because the grounding lines can be anything that
    # responds to setVisible.
    gl = {'modis':None, # From the simplified modis outline in Quantarctica
          'rignot':None, # Downloaded from NSIDC
          'tot_insar':None, # From file JSG provided
          'asaid':None, # ...I can't remember where I got this data ...
          } # type: Dict[str, Optional[Any]]
    return gl

# TODO: Equivalent GL utility to basemap utilities?
def set_grounding_line(ax, grounding_lines, label):
    # type: (matplotlib.axes.Axes, Dict[str, Any], str) -> None
    '''
    Sets desired grounding line to be visible (and creates the plot
    object if required). Sets all other grounding lines to not visible.
    '''
    for key in grounding_lines.keys():
        if key == label:
            if grounding_lines[key] is None:
                # have to create that grounding line!
                if key == 'modis':
                    patches = plotutils.basemap.load_moa_simple_basemap(gl=True)
                    grounding_lines[key] = ax.add_collection(patches)
                    grounding_lines[key].set_zorder(dp.gl_zorder)
                elif key == 'asaid':
                    # data = np.loadtxt('/media/psf/Home/Documents/WAIS/targ/supl/deva/gl/gl_asaid.xy')
                    # pickle.dump(data, open('/media/psf/Home/Documents/WAIS/targ/xtra/ALL/deva/gl_)asaid.pkl', 'wb'))
                    filename = WAIS + '/targ/xtra/ALL/deva/gl_asaid.pkl'
                    try:
                        data = pickle.load(open(filename, 'rb'))
                    except IOError as ex:
                        # If file doesn't exist, handle it hopefully-gracefully
                        print str(ex)
                        return
                    grounding_lines[key], = ax.plot(data[:,0], data[:,1], 'k.', markersize=2)
                    grounding_lines[key].set_zorder(dp.gl_zorder)

                elif key == 'rignot':
                    # TODO: move this to plotutils, and add a makefile that
                    #       performs the conversion. It would also be nice
                    #       to have a config file pointing to these things,
                    #       rather than hardcoded paths to our hierarchy...
                    # Downloaded from NSIDC, then reformatted:
                    # cat InSAR_GL_Antarctica.txt | awk '{print $2, $1}' | peony -2xy -pps71s > rignot2.txt
                    # data2 = np.loadtxt('/media/psf/Home/Documents/WAIS/targ/supl/xtra-rignot/nsidc0498_MEASURES_gl_antarc_V01/rignot2.txt')
                    # pickle.dump(data2, open('/media/psf/Home/Documents/WAIS/targ/xtra/ALL/deva/rignot_gl.pkl', 'wb'))
                    # I don't know what's going on here, b/c data2.nbytes gives
                    # 20M, but the resulting pickle file is 53M. I'm not digging
                    # into it now, because it loads quickly enough ...
                    filename = WAIS + '/targ/xtra/ALL/deva/rignot_gl.pkl'
                    try:
                        data = pickle.load(open(filename, 'rb'))
                    except IOError as ex:
                        # If file doesn't exist, handle it hopefully-gracefully
                        print str(ex)
                        return
                    grounding_lines[key], = ax.plot(data[:,0], data[:,1], 'k.', markersize=2)
                    grounding_lines[key].set_zorder(dp.gl_zorder)
                elif key == 'tot_insar':
                    filename = WAIS + '/targ/supl/deva/gl/gl_insar.XYZ'
                    try:
                        data = np.loadtxt(filename)
                    except IOError as ex:
                        # If file doesn't exist, handle it hopefully-gracefully
                        print str(ex)
                        return

                    grounding_lines[key], = ax.plot(data[:,0], data[:,1], '.', markersize=2)
                    grounding_lines[key].set_zorder(dp.gl_zorder)
                else:
                    plotUtilities.show_error_message_box("%r GL NYI" % (key))
            else:
                grounding_lines[key].set_visible(True)
        elif grounding_lines[key] is not None:
            grounding_lines[key].set_visible(False)
