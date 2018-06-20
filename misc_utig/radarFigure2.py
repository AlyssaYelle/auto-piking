#!/usr/bin/env python2.7

"""
Standalone viewer for UTIG's radar data; heavily inspired by
(x)eva(s)'s UI. However, in addition to keeping the same picking controls and
picker, enables switching between data products.

Also used as part of deva, for displaying radar data in context with the map.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np
import sys
import time

# # SDK attempted to install typing on melt, but it failed.
# # This is fully optional, and LEL can run it on her machine.
# try:
#     import typing
#     from typing import Any, Dict, List, Optional, Tuple
# except ImportError:
#     pass

import matplotlib as mpl
mpl.use('Qt4Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter
import matplotlib.widgets as mpw

# Hmmm. Adding the FigureCanvas means that pyside doesn't work anymore.
# Arrgh. It seems that pyqt isn't python3 compatible
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
#import PySide.QtCore as QtCore
#import PySide.QtGui as QtGui
#mpl.rcParams['backend.qt4']='PySide'
#from mpl.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# TODO: These need to be renamed. it's currently really confusing ...
# * mplUtilites - things that depend only on matplotlib
# * qtWidgets - currently are really pyqt widgets
# * plotUtilites - other pyqt stuff that doesn't count as widgets

import mplUtilities # for unzoomable, xevasHorizSelector, XevasVertSelector
import plotUtilities # for HLine, VLine
import qtWidgets #for RadioCheckInterface
import radarAnalysis # for channel offsets, rcoeff calculations, etc

import os
WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set')

sys.path.append('%s/syst/linux/py' % (WAIS))

import radutils

import plotutils
import plotutils.colormaps
mpl.cm.register_cmap(name='viridis', cmap=plotutils.colormaps.viridis)
mpl.cm.register_cmap(name='inferno', cmap=plotutils.colormaps.inferno)
import plotutils.scalebar
import plotutils.sparkline


import waisutils
import waisutils.season

# Notes on organization:
# * There's a hierarchy of functions that actually update the screen:
#   1) full_redraw() - this is for when the radar background has changed.
#      This will be cmap/clim changes, or xlim/ylim, or chan/product.
#   2) data_blit() - whenever any of the usually static data has changed.
#      This includes picks, pick maxima, and the various analysis/ELSA
#      plots.
#   3) cursor_blit() - for anything that follows the mouse around.
#      needs to be fast =)
#      This is currently just the trace, crosshairs & camera.
#
#   I'm not entirely sold that it's better to do it this way, rather than
#   re-capture the background/blit for each individual element, but it's
#   enough of a speed improvement, and seemed simpler. Maybe later I can break
#   it down (was going to be complicated since I couldn't figure out how to
#   use the sparkline as an artist)
#
# * The only functions that call draw_artist are data_blit, cursor_blit, and
#   the plot_*() functions that are called by data_blit.
# * Only *_set_visible should call artist.set_visible() ... any other calls
#   are superfluous, since blitting requires setting invisible/visible, and
#   if it has been modified, the correct blitting function has to be called
#   anyway to actually draw it.


# Constants identifying status of the pick buttons.
PICK_OFF = 0
PICK_ON = 1
PICK_ACTIVE = 2

# And, identifying which index is correct for the pick array
PICK_SAMPLE = {'vmax': 0, 'max': 1}
PICK_VAL = {'max': 2, 'vmax': 3}


class PlotConfig(object):
    '''
    Various constants about the plot configuration, that won't be
    changed at runtime.
    '''
    def __init__(self, pst):
        # type: (str) -> None

        # TODO: Only display the ones that are valid for the given pst ...
        season, _ = waisutils.SEASON_LOOKUP.get_season(pst)
        if waisutils.season.season_is_utig(season):
            if radutils.radutils.season_is_coherent(season):
                self.all_products = ['pik1', 'foc1', 'foc2', '1m', '1mfoc1',
                                     'lel', 'sdk']
            else:
                self.all_products = ['der', 'under']
        elif waisutils.season.season_is_cresis(season):
            self.all_products = ['qlook', 'mvdr', 'standard']
        elif waisutils.season.season_is_bas(season):
            self.all_products = ['cL_log', 'cL', 'pL_log', 'pL']
        else: # For now, this is just INKA1
            self.all_products = ['raw']

        self.available_channels = [] # type: List[int]

        self.all_cmaps = ['gray', 'jet', 'viridis', 'inferno', 'seismic',
                          'Greys']
        # Depending on the cmap, we may want to show the spark lines in
        # different colors ...
        self.cmap_major_colors = {} # type: Dict[str, str]
        self.cmap_minor_colors = {} # type: Dict[str, str]
        for cmap in ['gray', 'Greys']:
            # I had the rcoeff sparkline as 'c'/'b'
            self.cmap_major_colors[cmap] = 'r'
            self.cmap_minor_colors[cmap] = 'orange'
        for cmap in ['jet', 'viridis', 'inferno', 'seismic']:
            self.cmap_major_colors[cmap] = 'k'
            self.cmap_minor_colors[cmap] = 'grey'

# Hacky way of creating structs to hold all the various data...
# TODO: Should these wind up non-expandable?
class PlotParams(object):
    '''
    All the user-supplied parameters required for regenerating the plot
    (Includes some that depend on the data, if the user has final say.)

    Some are initialized here; I'm not sure if that's a good idea.
    '''
    def __init__(self):
        # type: () -> None
        self.curr_xlim = None # type: Optional[Tuple[int, int]]
        self.curr_ylim = None # type: Optional[Tuple[int, int]]
        # whether we're using traces_pik1 or traces_1m
        self.xunits = None # type: Optional[str]
        # how many traces are skipped between the displayed traces
        self.radar_skip = None # type: Optional[int]

        # which trace the camera cursor should currently be on.
        self.camera_trace = None # type: Optional[int]
        self.displayed_trace_num = None # type: Optional[int]

        # Whether these positions should be frozen or updated as the mouse moves
        self.crosshair_frozen = False
        self.camera_frozen = False
        self.trace_frozen = False

        # Whether these should be visible ..
        self.crosshair_visible = False
        self.camera_visible = False
        self.trace_visible = False

        self.simple_rcoeff_visible = False
        self.simple_rcoeff_needs_recalc = True

        self.rcoeff_visible = False
        self.rcoeff_needs_recalc = True
        self.rcoeff_srf_label = None # type: Optional[str]
        self.rcoeff_bed_label = None # type: Optional[str]
        self.rcoeff_iceloss = None # type: Optional[float]

        self.multiple_visible = False
        self.multiple_needs_recalc = True
        self.multiple_srf_label = None # type: Optional[str]
        self.multiple_bed_label = None # type: Optional[str]
        self.multiple_srf_count = None # type: Optional[int]
        self.multiple_bed_count = None # type: Optional[int]

        self.hydrostatic_visible = False

        self.vert_scale_visible = False
        # Units of m.
        self.vert_scale_length = 500 # type: float
        # Units of axis-fraction
        self.vert_scale_x0 = 0.05
        self.vert_scale_y0 = 0.1
        self.horiz_scale_visible = False
        # this is  in units of km.
        self.horiz_scale_length = 10 # type: float
        # units of axis-fraction
        self.horiz_scale_x0 = 0.1
        self.horiz_scale_y0 = 0.05

        #UGH. This makes me uncomfortable, since it's touched by adding
        # new picks ... also, the naming sucks.
        self.pick_visible = {} # type: Dict[str, bool]
        self.pick_colors = {} # type: Dict[str, QtGui.QColor]
        # NB - when we initialize the display from the parameters,
        # this is not taken into account.
        self.top_pick_status = PICK_OFF
        self.bottom_pick_status = PICK_OFF

        self.product = None # type: Optional[str]
        self.channel = None # type: Optional[int]

        self.cmap = 'gray'
        self.clim = (0, 1) #what's currently displayed
        self.cmin = 0 # min val from radar
        self.cmax = 1 # max val from radar

        self.active_horiz = None # type: Optional[str]
        self.max_type = 'max' #alternative would be 'imax'

        self.mouse_mode = 'zoom'

    def initialize_from_radar(self, radar_data):
        # type: (RadarData) -> None
        '''
        Called to initialize the plotting parameters to match the
        limits implied by the radargram. Only called at the start - we don't
        want the plots to change as a function of reloading data.
        # TODO: Maybe have clim change, but not xlim, for reloading? ... the
        # pik1/1m change is a bigger, more annoying, problem.
        '''
        self.curr_xlim = (0, radar_data.num_traces-1)
        self.curr_ylim = (radar_data.num_samples-1, 0)

        self.update_clim_from_radar(radar_data)

    def update_clim_from_radar(self, radar_data):
        # type: (RadarData) -> None
        '''
        Called to update the plotting parameters without changing the bounds.
        '''
        self.cmin = radar_data.min_val
        self.cmax = radar_data.max_val

        if self.product == 'der':
            init_clim = (-5000, 5000)
        elif self.product == 'under':
            init_clim = (0, 1000000)
        else:
            init_clim = (self.cmin, self.cmax)
        self.clim = init_clim


class TransectData(object):
    '''
    This is all the per-transect data that's loaded in from the hierarchy.
    Doesn't change after initialization.
    '''
    def __init__(self, pst):
        # type: (str) -> None
        self.pst = pst

        self.available_products = radutils.radutils.get_available_products(pst)

        # TODO: initialize this with start/end from deva?
        # TODO: need to use a different one for Cresis and BAS, since
        # their timing stuff is very very different from UTIG's
        # (Simpler! We just trust the data in $WAIS/targ/xtra/ALL/deva/psts!)
        self.rtc = radutils.conversions.RadarTimeConverter(self.pst)
        self.rpc = radutils.conversions.RadarPositionConverter(self.pst,
                                                               self.rtc)

        t0 = time.time()
        self.pick_dict = radutils.pickutils.load_all_picks(self.pst)
        print("%0.2f secs to load picks" % (time.time() - t0))

        # # Compute along-track distance for all traces.
        # self.dists = self.pcor_position_converter.along_track_dists(
        #     np.arange(self.num_traces), 'traces')


class RadarData(object):
    '''
    This is all the radar-specific data, for a given product that has
    been loaded. Includes parameters derived from the data.
    '''
    def __init__(self, pst, product, channel, filename=None):
        # type: (str, str, int, Optional[str]) -> None
        if waisutils.season.pst_is_cresis(pst):
            self.data = radutils.radutils.load_cresis_data(pst, product,
                                                           channel)
            print('cressis')
            print(pst)
            print(product)
            print(channel)
        elif waisutils.season.pst_is_bas(pst):
            # no diff channels...
            self.data = radutils.radutils.load_bas_data(pst, product)
            print('bas')
            print(pst)
            print(product)
            print(channel)
        else:
            self.data = radutils.radutils.load_radar_data(pst, product, channel,
                                                          filename)
            print('not cressis or bas')
            print(pst)
            print(product)
            print(channel)
            print(filename)

        self.num_traces, self.num_samples = self.data.shape
        self.min_val = np.amin(self.data)
        self.max_val = np.amax(self.data)
        print('num traces:', self.num_traces)
        print('num samples:', self.num_samples)
        print('data shape:', self.data.shape)
        print('min value', self.min_val)
        print('max value:', self.max_val)


class PlotObjects(object):
    '''
    Various handles that result from plotting that we want to keep in scope.
    Used when updating the plots ...
    This has slots for ALL of them, but some of the RadarWindow classes don't
    use all.
    '''
    def __init__(self):
        # type: () -> None
        # TODO: I'm not sure how to handle this in mypy. I want to use this
        # class like a struct to just pass around the various variables, but
        # all of 'em need to be instantiated at setup time.
        # I wasn't sure how to do a dynamic container class, so I'm just
        # declaring all of 'em here.
        # I'm also not sure how many of 'em need to be kept around vs.
        # just creating them...


        # All of these are initialized by create_layout
        self.main_frame = None # type: Optional[QtGui.QWidget]
        self.cursor = None # type: Optional[QtGui.QCursor]

        self.fig = None # type: Optional[Figure]
        self.canvas = None # type: Optional[FigureCanvas]
        self.mpl_toolbar = None # type: Optional[mplUtilities.SaveToolbar]
        self.radar_plot = None # type: Optional[mpl.image.AxesImage]

        self.dpi = None # type: Optional[int]

        self.full_ax = None # type: Optional[mpl.axes.Axes]
        self.pick_ax = None # type: Optional[mpl.axes.Axes]
        self.radar_ax = None # type: Optional[mpl.axes.Axes]
        self.xevas_horiz_ax = None # type: Optional[mpl.axes.Axes]
        self.xevas_vert_ax = None # type: Optional[mpl.axes.Axes]

        self.xevas_horiz = None
        # type: Optional[mplUtilities.XevasHorizSelector]
        self.xevas_vert = None # type: Optional[mplUtilities.XevasVertSelector]

        self.top_picks = None # type: Optional[mpl.lines.Line2D]
        self.bottom_picks = None # type: Optional[mpl.lines.Line2D]

        self.camera_x = None # type: Optional[mpl.lines.Line2D]
        self.crosshair_x = None # type: Optional[mpl.lines.Line2D]
        self.crosshair_y = None # type: Optional[mpl.lines.Line2D]

        self.trace_sparkline = None
        # type: Optional[plotutils.sparkline.Sparkline]
        self.trace_base = None # type: Optional[mpl.lines.Line2D]
        self.simple_rcoeff_sparkline = None
        # type: Optional[plotutils.sparkline.Sparkline]
        self.rcoeff_sparkline = None
        # type: Optional[plotutils.sparkline.Sparkline]
        self.multiple_line = None # type: Optional[mpl.lines.Line2D]

        self.left_click_rs = {} # type: Dict[str, mpw.RectangleSelector]
        self.right_click_rs = {} # type: Dict[str, mpw.RectangleSelector]

        self.mouse_mode_buttons = {} # type: Dict[str, QtGui.QRadioButton]
        self.mouse_mode_group = None # type: Optional[QtGui.QButtonGroup]

        self.prev_button = None # type: Optional[QtGui.QPushButton]
        self.full_button = None # type: Optional[QtGui.QPushButton]
        self.next_button = None # type: Optional[QtGui.QPushButton]

        self.colormap_buttons = {} # type: Dict[str, QtGui.QRadioButton]
        self.colormap_group = None # type: Optional[QtGui.QButtonGroup]

        self.channel_buttons = {} # type: Dict[int, QtGui.QRadioButton]
        self.channel_group = None # type: Optional[QtGui.QButtonGroup]

        self.product_buttons = {} # type: Dict[str, QtGui.QRadioButton]
        self.product_group = None # type: Optional[QtGui.QButtonGroup]

        self.max_type_buttons = {} # type: Dict[str, QtGui.QRadioButton]
        self.max_type_group = None # type: Optional[QtGui.QButtonGroup]

        self.trace_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.crosshair_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.crossover_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.camera_checkbox = None # type: Optional[QtGui.QCheckBox]

        self.clim_slider = None # type: Optional[qtWidgets.DoubleSlider]

        self.pick_button1 = None # type: Optional[QtGui.QPushButton]
        self.pick_button2 = None # type: Optional[QtGui.QPushButton]
        self.auto_pick_button = None # type: Optional[QtGui.QPushButton]
        self.save_picks_button = None # type: Optional[QtGui.QPushButton]
        self.copy_picks_button = None # type: Optional[QtGui.QPushButton]
        self.new_picks_textbox = None # type: Optional[QtGui.QLineEdit]
        self.new_picks_button = None # type: Optional[QtGui.QPushButton]

        self.simple_rcoeff_checkbox = None # type: Optional[QtGui.QCheckBox]

        self.rcoeff_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.rcoeff_recalc_button = None # type: Optional[QtGui.QPushButton]
        self.rcoeff_srf_label = None # type: Optional[QtGui.QLabel]
        self.rcoeff_srf_combo = None # type: Optional[QtGui.QComboBox]
        self.rcoeff_bed_label = None # type: Optional[QtGui.QLabel]
        self.rcoeff_bed_combo = None # type: Optional[QtGui.QComboBox]
        self.rcoeff_iceloss_label = None # type: Optional[QtGui.QLabel]
        self.rcoeff_iceloss_textbox = None # type: Optional[QtGui.QLineEdit]

        self.hydrostatic_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.hydrostatic_bed_label = None # type: Optional[QtGui.QLabel]
        self.hydrostatic_bed_combo = None # type: Optional[QtGui.QComboBox]

        self.multiple_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.multiple_recalc_button = None # type: Optional[QtGui.QPushButton]
        self.multiple_srf_label = None # type: Optional[QtGui.QLabel]
        self.multiple_srf_combo = None # type: Optional[QtGui.QComboBox]
        self.multiple_bed_label = None # type: Optional[QtGui.QLabel]
        self.multiple_bed_combo = None # type: Optional[QtGui.QComboBox]
        self.multiple_srf_count_label = None # type: Optional[QtGui.QLabel]
        self.multiple_srf_count_textbox = None # type: Optional[QtGui.QLineEdit]
        self.multiple_bed_count_label = None # type: Optional[QtGui.QLabel]
        self.multiple_bed_count_textbox = None # type: Optional[QtGui.QLineEdit]

        self.vert_scale_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.vert_scale_length_label = None # type: Optional[QtGui.QLabel]
        self.vert_scale_length_textbox = None # type: Optional[QtGui.QLineEdit]
        self.vert_scale_origin_label = None # type: Optional[QtGui.QLabel]
        self.vert_scale_x0_textbox = None # type: Optional[QtGui.QLineEdit]
        self.vert_scale_y0_textbox = None # type: Optional[QtGui.QLineEdit]

        self.horiz_scale_checkbox = None # type: Optional[QtGui.QCheckBox]
        self.horiz_scale_length_label = None # type: Optional[QtGui.QLabel]
        self.horiz_scale_length_textbox = None # type: Optional[QtGui.QLineEdit]
        self.horiz_scale_origin_label = None # type: Optional[QtGui.QLabel]
        self.horiz_scale_x0_textbox = None # type: Optional[QtGui.QLineEdit]
        self.horiz_scale_y0_textbox = None # type: Optional[QtGui.QLineEdit]

        self.vert_scale = None # type: Optional[plotutils.scalebar.Scalebar]
        self.horiz_scale = None # type: Optional[plotutils.scalebar.Scalebar]

        self.rci = None # type: Optional[qtWidgets.RadioCheckInterface]

        self.pick_scroll_area = None # type: Optional[QtGui.QScrollArea]

        self.quit_button = None # type: Optional[QtGui.QPushButton]
        self.new_elsa_data_button = None # type: Optional[QtGui.QPushButton]

        self.elsa_data_combo = None # type: Optional[QtGui.QComboBox]
        self.elsa_widget = None # type: Optional[qtWidgets.TextColorInterface]

        # This is filled in in initialize_gui_from_params_data
        self.pick_lines = {} # type: Dict[str, mpl.lines.Line2D]

        self.wiggles_hbox = None # type: Optional[QtGui.QHBoxLayout]
        self.tabs = None # type: Optional[QtGui.QTabWidget]


def calc_radar_skip(fig, ax, xlim):
    # type: (Figure, mpl.axes.Axes, Tuple[int, int]) -> int
    '''
    calculates how many traces can be dropped and still have as many
    traces as available pixels.

    Uses curr_xlim so we can re-calculate skip before plotting.
    '''
    if xlim is None:
        print("WARNING: called calc_radar_skip w/ xlim==None")
        return 1 # I added this to make mypy happy.
    ax_width, _ = mplUtilities.get_ax_shape(fig, ax)
    num_fig_traces = xlim[1] - xlim[0]
    radar_skip = max(1, int(np.ceil(num_fig_traces / ax_width)))
    return radar_skip


class EvaRadarWindow(object):
    """
    Eventually, DAY wants to make a version of RadarWindow that's a near-clone
    to xevas, in order to replicate the undergrad picking experience.
    I'm less invested in this project, though it would get rid of all the
    trickiest bits of radarFigure (trying to figure out where data is), since
    it just passes in the radargram filename and the pickdir.
    """
    pass

class BasicRadarWindow(QtGui.QMainWindow):
    # This is the one that I think people should be using to pick.
    def __init__(self,
                 pst, # type: str
                 filename=None, # type: Optional[str]
                 parent=None, # type: Optional[Any]
                 parent_xlim_changed_cb=None,
                 # type: Optional[Callable[List[float]]]
                 parent_cursor_cb=None, # type: Optional[Callable[float]]
                 close_cb=None # type: Optional[Callable[None]]
                 ):
        # type: (...) -> None
        '''
        params:
        * pst - which PST to plot.
        * filename - direct path to the file to load
        * parent_xlim_changed_cb - callback (into main Deva window) that keeps
          the highlighted segment of the PST updated. expects a tuple of posix
          times.
        * parent_cursor_cb - callback (into main Deva window) that puts a mark
          on the PST corresponding to where the cursor is in the radarFigure.
        * gps_{start, end} - posix timestamps for when the corresponding line
          in deva should be updated.
        * close_cb() - callback for when radar figure is being closed, used so
          the main deva window can clear the highlighted regions.
        '''
        # This is for the QtGui stuff
        super(BasicRadarWindow, self).__init__(parent)

        self.pst = pst
        self.parent_xlim_changed_cb = parent_xlim_changed_cb
        self.parent_cursor_cb = parent_cursor_cb
        self.close_cb = close_cb

        self.setWindowTitle('keva: ' + pst)

        # These parameters should be independent of the plotting tool we use
        self.plot_params = PlotParams()
        self.plot_config = PlotConfig(pst)

        self.transect_data = TransectData(self.pst)

        if 'pik1' in self.transect_data.available_products:
            self.plot_params.product = 'pik1'
        else:
            self.plot_params.product = self.transect_data.available_products[0]
        print('printing stuff')
        print(self.pst, self.plot_params.product)
        self.plot_config.available_channels = radutils.radutils.get_available_channels(self.pst, self.plot_params.product)
        if 2 in self.plot_config.available_channels:
            self.plot_params.channel = 2
        else:
            self.plot_params.channel = self.plot_config.available_channels[-1]

        if radutils.radutils.product_is_1m(self.plot_params.product):
            self.plot_params.xunits = 'traces_1m'
        else:
            self.plot_params.xunits = 'traces_pik1'

        # Set up the visual display, and hook up all the callbacks.
        # TODO: get rid of dependence on plot_params.available_products?
        self.plot_objects = self.create_layout(self.plot_params,
                                               self.plot_config)

        # # TODO: At some point, will have to add in the pcor loader stuff...
        # sorted_data = self.pcor_loader.available_data.keys()
        # sorted_data.sort()
        # for data_name in sorted_data:
        #     self.pcor_data_combo.addItem(data_name)

        self.radar_data = RadarData(self.pst, self.plot_params.product,
                                    self.plot_params.channel, filename)

        self.plot_params.initialize_from_radar(self.radar_data)

        # This needs to come after initialize_from_radar, b/c it depends on xlim
        self.initialize_gui_from_params_data(self.plot_params,
                                             self.transect_data)

        # This is annoying, because it depends on and modifies plot_params
        # However, I think that all that matters is that the fig and ax exist,
        # not their state.
        self.plot_params.radar_skip = calc_radar_skip(
            self.plot_objects.fig, self.plot_objects.radar_ax,
            self.plot_params.curr_xlim)

        self.plot_objects.radar_plot = self.plot_objects.radar_ax.imshow(
            self.radar_data.data[:, ::self.plot_params.radar_skip].T,
            aspect='auto', interpolation='nearest', zorder=0)

        # a simple canvas.draw() doesn't work here for some reason...
        # plot
        self.full_redraw()

    # This is hooked up automagically!
    # However, it only works if the focus is on the frame, not the canvas.
    # So, I made the canvas unfocusable...
    def keyPressEvent(self, event):
        # type: (QtGui.QKeyEvent) -> None
        if type(event) == QtGui.QKeyEvent:
            self._on_qt_key_press(event)
            # By doing this here, we don't let anybody downstream of this
            # catch 'em. If I wanted to allow that, move event.accept()
            # into the callback so we only accept keypresses that it handles.
            event.accept()
        else:
            event.ignore()


    def maybe_update_trace(self, trace_num):
        # type: (int) -> bool
        '''
        Called if we want to check for frozen before moving the trace.
        '''
        if self.plot_params.trace_visible and not self.plot_params.trace_frozen:
            self.update_trace(trace_num)
            return True
        else:
            return False

    def initialize_gui_from_params_data(self, plot_params, transect_data):
        # type: (PlotParams, TransectData) -> None
        '''
        This just sets the current state of various GUI widgets based on:
        * plot params - initial state of buttons
        * transect_data - used for pickfile names, available radar products
        (Yeah, I could just access self.plot_params, but I want the call
        signature to be explicit what it depends on.)
        '''
        for product in transect_data.available_products:
            self.plot_objects.product_buttons[product].setEnabled(True)
        sorted_pick_names = sorted(transect_data.pick_dict.keys())
        for pick_name in sorted_pick_names:
            self.add_pick_label(pick_name)
            color = self.plot_objects.rci.get_color(pick_name)
            self.plot_objects.pick_lines[pick_name], = self.plot_objects.radar_ax.plot(0, 0, color=color)
            # NB: This makes me unhappy. I'm also modifying the params!
            self.plot_params.pick_visible[pick_name] = False
            self.plot_params.pick_colors[pick_name] = color

        self.plot_objects.channel_buttons[plot_params.channel].setChecked(True)
        self.plot_objects.product_buttons[plot_params.product].setChecked(True)
        self.plot_objects.colormap_buttons[plot_params.cmap].setChecked(True)
        self.plot_objects.max_type_buttons[plot_params.max_type].setChecked(True)
        self.plot_objects.mouse_mode_buttons[plot_params.mouse_mode].setChecked(True)
        mouse_mode = self.plot_params.mouse_mode
        self.plot_objects.left_click_rs[mouse_mode].set_active(True)
        self.plot_objects.right_click_rs[mouse_mode].set_active(True)

        self.plot_objects.radar_ax.set_xlim(plot_params.curr_xlim)
        self.plot_objects.radar_ax.set_ylim(plot_params.curr_ylim)

        self.plot_objects.clim_slider.set_range((plot_params.cmin,
                                                 plot_params.cmax))
        self.plot_objects.clim_slider.set_value(plot_params.clim)

    def add_pick_label(self, label):
        # type: (str) -> None
        self.plot_objects.rci.add_row(label)

    def update_trace(self, trace_num):
        # type: (int) -> None
        '''
        Center trace on median, scaled to take up 1/16th of display..
        Raw values are reported in dBm, with a season-dependent offset.
        '''
        self.plot_params.displayed_trace_num = trace_num

        offset = radarAnalysis.channel_offsets[self.plot_params.channel]
        trace_dB = self.radar_data.data[trace_num, :]/1000. + offset
        yy = np.arange(0, self.radar_data.num_samples)

        self.plot_objects.trace_sparkline.set_data(trace_dB, yy,
                                                   trace_num + 0.5)
        self.plot_objects.trace_base.set_data([trace_num + 0.5,
                                               trace_num + 0.5],
                                              [0, self.radar_data.num_samples])

    def maybe_update_crosshair(self, trace, sample):
        # type: (int, int) -> bool
        '''
        Called if we want to check for frozen before moving the trace.
        '''
        if self.plot_params.crosshair_visible and not self.plot_params.crosshair_frozen:
            self.update_crosshair(trace, sample)
            return True
        else:
            return False

    def update_crosshair(self, trace, sample):
        # type: (int, int) -> None

        if self.parent_cursor_cb is not None:
            trace_units = 'traces_pik1'
            if radutils.radutils.product_is_1m(self.plot_params.product):
                trace_units = 'traces_1m'
            #if radutils.radutils.pst_is_cresis(self.pst):
            #    trace_units = 'linear'

            tt, = self.transect_data.rtc.convert([trace], trace_units, 'posix')
            self.parent_cursor_cb(tt)

        self.plot_objects.crosshair_y.set_data([0, self.radar_data.num_traces],
                                               [sample, sample])
        self.plot_objects.crosshair_x.set_data([trace, trace],
                                               [0, self.radar_data.num_samples])

    def update_xlim(self, new_xlim):
        # type: (Tuple[int, int]) -> None
        assert len(new_xlim) == 2
        # Sometimes calls will have it be a float; it seems easisest
        # to round & cast here, rather than force all callers to remember.
        new_xlim = (int(np.round(new_xlim[0])), int(np.round(new_xlim[1])))
        self.plot_params.curr_xlim = new_xlim
        self.plot_params.radar_skip = calc_radar_skip(
            self.plot_objects.fig, self.plot_objects.radar_ax, new_xlim)
        self.plot_objects.radar_ax.set_xlim(new_xlim)
        # It's OK, this isn't infinitely circular ...
        # update_selection doesn't trigger any cbs.
        num_traces = self.radar_data.num_traces
        self.plot_objects.xevas_horiz.update_selection(
            (1.0*new_xlim[0]/(num_traces-1), 1.0*new_xlim[1]/(num_traces-1)))

    def update_ylim(self, new_ylim):
        # type: (Tuple[int, int]) -> None
        assert len(new_ylim) == 2
        # Sometimes calls will have it be a float; it seems easisest
        # to round & cast here, rather than force all callers to remember.
        new_ylim = (int(np.round(new_ylim[0])), int(np.round(new_ylim[1])))

        self.plot_params.curr_ylim = new_ylim
        self.plot_objects.radar_ax.set_ylim(new_ylim)
        # It's OK, this isn't infinitely circular ...
        # update_selection doesn't trigger any cbs.
        num_samples = self.radar_data.num_samples
        self.plot_objects.xevas_vert.update_selection(
            (1-1.0*new_ylim[0]/(num_samples-1),
             1-1.0*new_ylim[1]/(num_samples-1)))

    def full_redraw(self):
        # type: () -> None
        '''
        Does a full redraw of everything; radar_data, transect_data and
        plot_params should have all the information to update plot_objects.
        I expect this to only be necessary if axes, cmap, clim,
        or plot_size changes.

        NB - clim/cmap still requires the full one, b/c I want to capture
          the state w/o any artists.
        * if they're animated=True, then they won't show up after the draw
          w/o a blit, so might as well do a full one anyways.
        * if they're animated=False, then I'll need to clear 'em off
          before recording, which also requires a draw.
        '''
        t0 = time.time()
        data = self.radar_data.data
        xlim = self.plot_params.curr_xlim
        ylim = self.plot_params.curr_ylim
        radar_skip = self.plot_params.radar_skip
        try:
            self.plot_objects.radar_plot.set_data(data[xlim[0]:xlim[1]:radar_skip,
                                                       ylim[1]:ylim[0]].T)
        except TypeError as ex:
            print(ex)
            print("xlim: %r, ylim:%r, radar_skip: %r" % (xlim, ylim, radar_skip))
            raise ex
        extent = np.append(xlim, ylim)
        self.plot_objects.radar_plot.set_extent(extent)
        self.plot_objects.radar_plot.set_cmap(self.plot_params.cmap)
        self.plot_objects.radar_plot.set_clim(self.plot_params.clim)

        self.cursor_set_invisible(self.plot_objects)
        self.data_set_invisible(self.plot_objects)

        # Needs to happen before calls to blitting, but after updating
        # other axes. We use full_ax in order to also capture the
        # axis labels and xevas bars
        self.plot_objects.canvas.draw()
        self.radar_restore = self.plot_objects.canvas.copy_from_bbox(
            self.plot_objects.full_ax.bbox)

        # QUESTION: Surely these set_visible calls are redundant, since
        # the blitting sets them invisible/visible?
        self.cursor_set_visible(self.plot_objects, self.plot_params)
        self.data_set_visible(self.plot_objects, self.plot_params)

        self.data_blit() # also calls cursor_blit

        # TODO: I expect this to be slow until deva blits its updates ...
        if self.parent_xlim_changed_cb is not None:
            trace_units = 'traces_pik1'
            if radutils.radutils.product_is_1m(self.plot_params.product):
                trace_units = 'traces_1m'
            #if radutils.radutils.pst_is_cresis(self.pst):
            #    trace_units = 'linear'
            #print self.pst, trace_units, xlim
            #print self.transect_data.rtc.num_traces
            #print self.transect_data.rtc.posix

            t0, = self.transect_data.rtc.convert([xlim[0]], trace_units,
                                                 'posix')
            t1, = self.transect_data.rtc.convert([xlim[1]], trace_units,
                                                 'posix')
            self.parent_xlim_changed_cb((t0, t1))

        #print("time for full redraw:", time.time() - t0)

    def data_blit(self):
        # type: () -> None
        '''
        This redraws all the various rcoeff/pick/etc plots, but not the
        radar background.
        # TODO: I haven't tested  whether it would be faster to do it like
        # this or do a per-artist blit when it changes. However, this seems
        # easier/cleaner.
        '''
        t0 = time.time()

        self.plot_objects.canvas.restore_region(self.radar_restore)

        self.data_set_visible(self.plot_objects, self.plot_params)
        # TODO: If this series starts getting too slow, move the "set_data"
        # logic back to the callbacks that change it. However, there are
        # enough things that change the picks/max values that it's a lot
        # simpler to put all of that right here.
        self.plot_curr_picks()
        self.plot_computed_horizons()

        self.plot_objects.canvas.update()

        self.data_restore = self.plot_objects.canvas.copy_from_bbox(
            self.plot_objects.full_ax.bbox)

        t1 = time.time()
        #print("time for data blit:  ", t1 - t0)
        self.cursor_blit()
        #print("time for cursor blit:  ", time.time() - t1)

    def data_set_invisible(self, plot_objects):
        # type: (PlotObjects) -> None
        plot_objects.top_picks.set_visible(False)
        plot_objects.bottom_picks.set_visible(False)
        for _, pick_line in plot_objects.pick_lines.iteritems():
            pick_line.set_visible(False)

    def data_set_visible(self, plot_objects, plot_params):
        # type: (PlotObjects, PlotParams) -> None
        '''
        Set data elements visible if they should be
        '''
        if plot_params.active_horiz is not None:
            if plot_params.top_pick_status != PICK_OFF:
                plot_objects.top_picks.set_visible(True)
            else:
                plot_objects.top_picks.set_visible(False)
            if plot_params.bottom_pick_status != PICK_OFF:
                plot_objects.bottom_picks.set_visible(True)
            else:
                plot_objects.bottom_picks.set_visible(False)

        for pick_name, pick_line in plot_objects.pick_lines.iteritems():
            visible = plot_params.pick_visible[pick_name]
            pick_line.set_visible(visible)

    def cursor_blit(self):
        # type: () -> None
        '''
        Restores JUST the background, not any of the mouse-following
        artists, then redraws all of 'em.

        I'm a little worried about how long it'll take to redraw
        ALL of 'em, but trying to recapture only the bits that
        changed seems combinatorially annoying.
        Plus, the reflection coeff/elsa plots don't need to be blitted.
        '''
        t0 = time.time()
        # TODO: Make this more general, rather than hardcoding artists in?
        # I couldn't figure out how to do it alongside the special cases
        # for the sparkline (and making the sparkline an artist ...
        # was a no-go)
        self.plot_objects.canvas.restore_region(self.data_restore)

        self.cursor_set_visible(self.plot_objects, self.plot_params)

        # I tried moving all plotting to a different function and always
        # calling it, but set_data is slow. Much better to only set it when
        # something has changed, and it's easier to do so in a callback.

        # Draw the artists that need to be blitted ...this needs to happen
        # after the restore_region and set_visible and before canvas.update()
        self.plot_objects.radar_ax.draw_artist(self.plot_objects.crosshair_x)
        self.plot_objects.radar_ax.draw_artist(self.plot_objects.crosshair_y)

        self.plot_objects.radar_ax.draw_artist(self.plot_objects.trace_base)
        for element in self.plot_objects.trace_sparkline.elements.values():
            self.plot_objects.radar_ax.draw_artist(element)

        # This is oft-recommended but reputed to leak memory:
        # (http://bastibe.de/2013-05-30-speeding-up-matplotlib.html)
        #self.plot_objects.canvas.blit(self.plot_objects.radar_ax.bbox)
        # This seems to be just about as fast ...
        self.plot_objects.canvas.update()
        #print("time for cursor blit:", time.time() - t0)

    def cursor_set_invisible(self, plot_objects):
        # type: (PlotObjects) -> None
        plot_objects.crosshair_x.set_visible(False)
        plot_objects.crosshair_y.set_visible(False)
        plot_objects.trace_sparkline.set_visible(False)
        plot_objects.trace_base.set_visible(False)

    def cursor_set_visible(self, plot_objects, plot_params):
        # type: (PlotObjects, PlotParams) -> None
        '''
        Restores any elements that follow the mouse around to be visible
        if they were supposed to be ...
        '''
        plot_objects.crosshair_x.set_visible(plot_params.crosshair_visible)
        plot_objects.crosshair_y.set_visible(plot_params.crosshair_visible)
        plot_objects.trace_sparkline.set_visible(plot_params.trace_visible)
        plot_objects.trace_base.set_visible(plot_params.trace_visible)


    def plot_computed_horizons(self):
        # type: () -> None
        '''
        This sets the data and draws artists for the horizon
        # NB - used to be update_pick_lines
        '''
        for pick_name, picks in self.transect_data.pick_dict.iteritems():
            if picks.max_vals is not None:
                sweeps = picks.get_sweeps()

                if radutils.radutils.product_is_1m(picks.product):
                    pick_units = 'traces_1m'
                else:
                    pick_units = 'traces_pik1'
                # mypy expects convert to take and return floats
                disp_sweeps = self.transect_data.rtc.convert(
                    [float(sw) for sw in sweeps], pick_units,
                    self.plot_params.xunits)

                yy = picks.max_vals[:, PICK_SAMPLE[self.plot_params.max_type]]
                self.plot_objects.pick_lines[pick_name].set_data(disp_sweeps,
                                                                 yy)
                color = self.plot_params.pick_colors[pick_name]
                self.plot_objects.pick_lines[pick_name].set_color(color)
            self.plot_objects.radar_ax.draw_artist(
                self.plot_objects.pick_lines[pick_name])

    def plot_curr_picks(self):
        # type: () -> None
        '''
        This sets the data and draws artists for picks
        '''
        if self.plot_params.active_horiz is None:
            self.plot_objects.top_picks.set_data(0, 0)
            self.plot_objects.bottom_picks.set_data(0, 0)
        else:
            picks = self.transect_data.pick_dict[self.plot_params.active_horiz]
            if picks.top is not None:
                # TODO: Convert these to the correct x axis scale!
                #plot_picks = picks.convert_picks(1, self.plot_params.product)
                plot_picks = picks.top
                self.plot_objects.top_picks.set_data(plot_picks[:, 0],
                                                     plot_picks[:, 1])
            else:
                self.plot_objects.top_picks.set_data(0, 0)

            if picks.bottom is not None:
                # TODO: Convert these to the correct x axis scale!
                #plot_picks = picks.convert_picks(2, self.plot_params.product)
                plot_picks = picks.bottom
                self.plot_objects.bottom_picks.set_data(plot_picks[:, 0],
                                                        plot_picks[:, 1])
            else:
                self.plot_objects.bottom_picks.set_data(0, 0)

        self.plot_objects.radar_ax.draw_artist(self.plot_objects.top_picks)
        self.plot_objects.radar_ax.draw_artist(self.plot_objects.bottom_picks)

    def radar_from_pick_coords(self, pick):
        # type: (Tuple[float, float]) -> Tuple[int, int]
        '''
        Converts point in display coords (from the pick axis) into data
        coords in the radar_ax. This thresholds to the shape of the radar
        axis, which means that picks just slightly off the side will be
        interpreted as labeling the last trace.
        '''
        # I tried precalculating this, but it was awkward to make sure it got
        # initialized correctly. It takes < 0.2ms per call, so I'm OK with
        # that penalty. Putting it in initialize_gui_from_params_data just
        # after set_{xlim,ylim} didn't do it.
        inv = self.plot_objects.radar_ax.transData.inverted()
        p0 = inv.transform(pick)
        xlim = self.plot_params.curr_xlim
        ylim = self.plot_params.curr_ylim
        xx = min(xlim[1], max(xlim[0], int(round(p0[0]))))
        # Tricksy .. axis reversed!
        yy = max(ylim[1], min(ylim[0], int(round(p0[1]))))
        return xx, yy

    def check_multiple_params(self, disp=False):
        # type: (bool) -> bool
        '''
        Inputs:
        * disp - if True, pops up error boxes regarding the invalid params.

        Returns:
        * False if there is not a complete valid set of parameters. Single
          pick names may be '' if corresponding count is 0.
          (plane-srf count, srf_pickname, srf-bed count, bed_pickname)
        '''
        srf_count = self.plot_params.multiple_srf_count
        if srf_count is None:
            if disp:
                msg = "please enter surface count"
                plotUtilities.show_error_message_box(msg)
            return False
        bed_count = self.plot_params.multiple_bed_count
        if bed_count is None:
            if disp:
                msg = "please enter bed count"
                plotUtilities.show_error_message_box(msg)
            return False

        if bed_count == 0 and srf_count == 0:
            if disp:
                msg = ("Cannot calcuate multiple if both plane-srf and "
                       "srf-bed count is 0!")
                plotUtilities.show_error_message_box(msg)
            return False

        srf_label = self.plot_params.multiple_srf_label
        if srf_label is None or srf_label == '':
            if disp:
                msg = "Please select a multiple surface pick."
                plotUtilities.show_error_message_box(msg)
            return False

        bed_label = ''
        if bed_count > 0:
            bed_label = self.plot_params.multiple_bed_label
            if bed_label is None or bed_label == '':
                if disp:
                    msg = "Please select a multiple bed pick."
                    plotUtilities.show_error_message_box(msg)
                return False

        return True

    def check_rcoeff_params(self, disp=False):
        # type: (bool) -> bool
        '''
        Inputs:
        * disp - if True, pops up error boxes regarding the invalid params.

        Returns:
        * False if there is not a complete valid set of parameters
          (srf_label, bed_label, iceloss)
        '''
        srf_label = self.plot_params.rcoeff_srf_label
        if srf_label is None or srf_label == '':
            if disp:
                msg = "Please select a rcoeff surface pick."
                plotUtilities.show_error_message_box(msg)
            return False

        bed_label = self.plot_params.rcoeff_bed_label
        if bed_label is None or bed_label == '':
            if disp:
                msg = "Please select a rcoeff bed pick."
                plotUtilities.show_error_message_box(msg)
            return False

        iceloss = self.plot_params.rcoeff_iceloss
        if iceloss is None:
            if disp:
                msg = "please select iceloss"
                plotUtilities.show_error_message_box(msg)
            return False
        return True

    def _on_left_rect_click_pick(self,
                                 eclick, # type: mpl.backend_bases.MouseEvent,
                                 erelease, # type: mpl.backend_bases.MouseEvent
                                ):
        # type: (...) -> None
        '''
        Adds a pick to the current list at the location the cursor was clicked.
        This requires converting the pick from pick_ax to radar_ax...
        (The erelease parameter is unused; it's there because this function
        is called by the rectangle selector.)
        '''
        top_pick_active = (self.plot_params.top_pick_status == PICK_ACTIVE)
        bottom_pick_active = (self.plot_params.bottom_pick_status == PICK_ACTIVE)
        if not top_pick_active and not bottom_pick_active:
            print("can't add pick if no horizons are active!")
            return
        # Check that we're allowed to modify the currently-active pick file,
        # based on it matching the data product that's currently displayed.
        active_horiz = self.plot_params.active_horiz
        pick_channel = self.transect_data.pick_dict[active_horiz].channel
        pick_product = self.transect_data.pick_dict[active_horiz].product
        if (pick_channel != self.plot_params.channel
            or pick_product != self.plot_params.product):
            msg = ("Cannot modify picks - data products do not match.\n\n"
                   "Selected picks are from: chan: %s / product: %s. \n\n"
                   % (pick_channel, pick_product) +
                   "Current plot is: chan: %s / product: %s."
                   % (self.plot_params.channel, self.plot_params.product))
            plotUtilities.show_error_message_box(msg)
            return
        # Actually insert the pick.
        trace, sample = self.radar_from_pick_coords((eclick.x, eclick.y))
        self.transect_data.pick_dict[active_horiz].add_pick(
            trace, sample, top_pick_active)
        radutils.pickutils.save_picks(
            self.transect_data.pick_dict[active_horiz], backup=True)

        self.data_blit()

    def _on_right_rect_click_pick(self,
                                  eclick, # type: mpl.backend_bases.MouseEvent
                                  erelease # type: mpl.backend_bases.MouseEvent
                                 ):
        # type: (...) -> None
        ''' right click-and-drag erases picked points.'''
        top_pick_active = (self.plot_params.top_pick_status == PICK_ACTIVE)
        bottom_pick_active = (self.plot_params.bottom_pick_status == PICK_ACTIVE)
        if not top_pick_active and not bottom_pick_active:
            print("can't delete pick if no horizons are active!")
            return
        # Check that we're allowed to modify the currently-active pick file,
        # based on it matching the data product that's currently displayed.
        active_horiz = self.plot_params.active_horiz
        if active_horiz is None:
            return
        pick_channel = self.transect_data.pick_dict[active_horiz].channel
        pick_product = self.transect_data.pick_dict[active_horiz].product
        if (pick_channel != self.plot_params.channel
            or pick_product != self.plot_params.product):
            msg = ("Cannot modify picks - data products do not match.\n\n"
                   "Selected picks are from: chan: %s / product: %s. \n\n"
                   % (pick_channel, pick_product) +
                   "Current plot is: chan: %s / product: %s."
                   % (self.plot_params.channel, self.plot_params.product))
            plotUtilities.show_error_message_box(msg)
            return

        click = self.radar_from_pick_coords((eclick.x, eclick.y))
        release = self.radar_from_pick_coords((erelease.x, erelease.y))
        minx = min([click[0], release[0]])
        maxx = max([click[0], release[0]])

        self.transect_data.pick_dict[active_horiz].delete_picks(
            minx, maxx, top_pick_active)

        radutils.pickutils.save_picks(
            self.transect_data.pick_dict[active_horiz], backup=True)


        self.data_blit()

    def _on_left_rect_click_zoom(self,
                                 eclick, # type: mpl.backend_bases.MouseEvent
                                 erelease # type: mpl.backend_bases.MouseEvent
                                ):
        # type: (...) -> None
        ''' left click-and-drag zooms in. '''
        num_traces = self.radar_data.num_traces
        num_samples = self.radar_data.num_samples
        click = self.radar_from_pick_coords((eclick.x, eclick.y))
        release = self.radar_from_pick_coords((erelease.x, erelease.y))
        x0 = min(max(0, click[0]), max(0, release[0]))
        x1 = max(min(num_traces-1, click[0]),
                 min(num_traces-1, release[0]))
        new_xlim = (x0, x1)
        y0 = min(max(0, click[1]), max(0, release[1]))
        y1 = max(min(num_samples-1, click[1]),
                 min(num_samples-1, release[1]))
        new_ylim = (y1, y0) # y-axis reversed ...

        if x1 == x0 or y1 == y0:
            #msg = "can't zoom in; selected region too small!"
            #plotUtilities.show_error_message_box(msg)
            return

        self.update_xlim(new_xlim)
        self.update_ylim(new_ylim)
        self.full_redraw()

    def _on_right_rect_click_zoom(self,
                                  eclick, # type: mpl.backend_bases.MouseEvent
                                  erelease # type: mpl.backend_bases.MouseEvent
                                 ):
        # type: (...) -> None
        '''
        Right click-and-drag zooms out s.t. the region presently displayed
        is shrunk to fit into the box drawn.
        Will not zoom out past image coordinates.
        '''
        curr_xlim = self.plot_params.curr_xlim
        curr_ylim = self.plot_params.curr_ylim
        num_traces = self.radar_data.num_traces
        num_samples = self.radar_data.num_samples

        click = self.radar_from_pick_coords((eclick.x, eclick.y))
        release = self.radar_from_pick_coords((erelease.x, erelease.y))
        curr_dx = curr_xlim[1] - curr_xlim[0]
        curr_dy = curr_ylim[0] - curr_ylim[1] # y-axis reversed ...

        selection_dx = abs(release[0] - click[0])
        selection_dy = abs(release[1] - click[1])
        if selection_dx == 0 or selection_dy == 0:
            #msg = "can't zoom out; selected region too small!"
            #plotUtilities.show_error_message_box(msg)
            return

        scalex = 1.0*curr_dx/selection_dx
        scaley = 1.0*curr_dy/selection_dy

        xbar = 0.5*(curr_xlim[0] + curr_xlim[1])
        ybar = 0.5*(curr_ylim[0] + curr_ylim[1])
        x0 = max(0, int(xbar - 0.5*curr_dx*scalex))
        x1 = min(num_traces-1, int(xbar + 0.5*curr_dx*scalex))
        y0 = max(0, int(ybar - 0.5*curr_dy*scaley))
        y1 = min(num_samples-1, int(ybar + 0.5*curr_dy*scaley))

        new_xlim = (x0, x1)
        new_ylim = (y1, y0)

        self.update_xlim(new_xlim)
        self.update_ylim(new_ylim)
        self.full_redraw()

    def _on_rect_click_pan(self,
                           eclick, # type: mpl.backend_bases.MouseEvent
                           erelease # type: mpl.backend_bases.MouseEvent
                          ):
        # type: (...) -> None
        ''' left and right clicks both pan identically.'''
        xmin, xmax = self.plot_params.curr_xlim
        ymax, ymin = self.plot_params.curr_ylim
        num_traces = self.radar_data.num_traces
        num_samples = self.radar_data.num_samples

        click = self.radar_from_pick_coords((eclick.x, eclick.y))
        release = self.radar_from_pick_coords((erelease.x, erelease.y))
        click_dx = release[0] - click[0]
        click_dy = release[1] - click[1]

        dx = min(xmin, max(click_dx, xmax + 1 - num_traces))
        dy = min(ymin, max(click_dy, ymax + 1 - num_samples))

        self.update_xlim((xmin-dx, xmax-dx))
        self.update_ylim((ymax-dy, ymin-dy))
        self.full_redraw()

    def _on_resize_event(self, event):
        # type: (mpl.backend_bases.ResizeEvent) -> None
        """
        TODO
        """
        self.plot_params.radar_skip = calc_radar_skip(
            self.plot_objects.fig, self.plot_objects.radar_ax,
            self.plot_params.curr_xlim)
        self.full_redraw()

    def _on_pick_key_press(self, key):
        # type: (QtCore.Key) -> None
        '''
        Handles all of the keyboard interaction for my new picking interface.
        NB - this interface works in parallel with the existing one.
        NB - new picks will overwrite old ones at same sweep.

        * key_A - autopick!
        * key_S - save!
        * key_D - delete nearest pick (or top & bottom if both on same trace)
        * key_C - close picks! Inserts inverted top/bottom picks at mouses's
                  current trace/sample.
        * 1-9   - add top/bottom bounds at the mouse's trace, with sample offset
                  given by numeric value of key.
        '''
        # Need to reject key press if there's no active pickfile
        if self.plot_params.active_horiz is None:
            return
        # Don't allow modifying the picks unless they're at least visible!
        if (self.plot_params.top_pick_status == PICK_OFF
            or self.plot_params.bottom_pick_status == PICK_OFF):
            return
        else:
            # This is confusing - even though the matplotlib transformations
            # tutorial seemed to claim to be in screen coordinates, you still
            # have to transform the result of the cursor from actual screen
            # coords to canvas coords
            global_pos = self.plot_objects.cursor.pos()
            canvas_pos = self.plot_objects.fig.canvas.mapFromGlobal(global_pos)
            inv = self.plot_objects.pick_ax.transAxes.inverted()
            xpick, ypick = inv.transform([canvas_pos.x(), canvas_pos.y()])
            # Ugh. This is the only way I could figure out how to flip the
            # yaxis and get the sign right on the resulting radar coords ...
            p0 = self.plot_objects.pick_ax.transAxes.transform([xpick,
                                                                1.0-ypick])
            if 0 <= xpick <= 1 and 0 <= ypick <= 1:
                xx, yy = self.radar_from_pick_coords(p0)
                trace = max(0, min(self.radar_data.num_traces-1,
                                   int(round(xx))))
                sample = max(0, min(self.radar_data.num_samples-1,
                                    int(round(yy))))
                #print("pressed key at:", xpick, ypick, trace, sample)
                picks = self.transect_data.pick_dict[self.plot_params.active_horiz]
                if key == QtCore.Qt.Key_D:
                    # TODO: Magic numbers! These just indicate how close the
                    # user has to click to a point in order to delete it.
                    picks.delete_pick(trace, sample, dtrace=100, dsample=20)
                    self.data_blit()
                elif key == QtCore.Qt.Key_C:
                    picks.close_pick(trace, sample)
                    self.data_blit()
                else:
                    radius = int(key) - 48
                    picks.add_pair(trace, sample, radius)
                    self.data_blit()

    # TODO: Maybe have this be partly/entirely configurable via plot_params?
    def _on_qt_key_press(self, event):
        # type: (QtGui.QKeyEvent) -> None
        """
        TODO
        """
        if event.key() == QtCore.Qt.Key_F and self.plot_params.trace_visible:
            self.plot_params.trace_frozen = not self.plot_params.trace_frozen
        elif (event.key() == QtCore.Qt.Key_G
              and self.plot_params.crosshair_visible):
            self.plot_params.crosshair_frozen = not self.plot_params.crosshair_frozen
        # And, adding support for enhanced picking =)
        elif event.key() == QtCore.Qt.Key_A:
            self._on_auto_pick_button_clicked()
        elif event.key() == QtCore.Qt.Key_S:
            self._on_save_picks_button_clicked()
        elif event.key() == QtCore.Qt.Key_E:
            self._on_prev_button_clicked()
        elif event.key() == QtCore.Qt.Key_R:
            self._on_next_button_clicked()
        elif event.key() == QtCore.Qt.Key_Y:
            self._on_full_button_clicked()
        elif event.key() in [QtCore.Qt.Key_C, QtCore.Qt.Key_D,
                             QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3,
                             QtCore.Qt.Key_4, QtCore.Qt.Key_5, QtCore.Qt.Key_6,
                             QtCore.Qt.Key_7, QtCore.Qt.Key_8, QtCore.Qt.Key_9]:
            # These all require figuring out the mouse postion in the radar ax,
            # so I'm sending them through the same handler.
            self._on_pick_key_press(event.key())

        # Using the key press events from the gui, the gui elements get first
        # dibbs on handling 'em. So, can't use left/right... replacing with ,.
        elif (event.key() == QtCore.Qt.Key_Comma
              and self.plot_params.trace_visible
              and self.plot_params.trace_frozen):
            self.plot_params.displayed_trace_num -= self.plot_params.radar_skip
            self.update_trace(self.plot_params.displayed_trace_num)
            self.cursor_blit()

        elif (event.key() == QtCore.Qt.Key_Period
              and self.plot_params.trace_visible
              and self.plot_params.trace_frozen):
            self.plot_params.displayed_trace_num += self.plot_params.radar_skip
            self.update_trace(self.plot_params.displayed_trace_num)
            self.cursor_blit()

    # TODO: connect to other canvas events ...
    # http://matplotlib.org/users/event_handling.html

    def _on_mouse_mode_group_pressed(self):
        # type: () -> None
        """
        TODO
        """
        for mode, button in self.plot_objects.mouse_mode_buttons.iteritems():
            if button.isDown():
                self.plot_params.mouse_mode = mode
                self.plot_objects.left_click_rs[mode].set_active(True)
                self.plot_objects.right_click_rs[mode].set_active(True)
            else:
                self.plot_objects.left_click_rs[mode].set_active(False)
                self.plot_objects.right_click_rs[mode].set_active(False)

    def _on_prev_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        xlim = self.plot_params.curr_xlim
        width = xlim[1] - xlim[0]
        shift = np.min([0.8 * width, xlim[0]])
        self.update_xlim((xlim[0] - shift, xlim[1] - shift))
        self.full_redraw()
    def _on_full_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        self.update_xlim((0, self.radar_data.num_traces-1))
        self.update_ylim((self.radar_data.num_samples-1, 0))
        self.full_redraw()
    def _on_next_button_clicked(self):
        # type: () -> None
        xlim = self.plot_params.curr_xlim
        width = xlim[1] - xlim[0]
        shift = np.min([0.8 * width, self.radar_data.num_traces - 1 - xlim[1]])
        self.update_xlim((xlim[0] + shift, xlim[1] + shift))
        self.full_redraw()

    def _on_colormap_group_pressed(self):
        # type: () -> None
        """
        TODO
        """
        for cmap, button in self.plot_objects.colormap_buttons.iteritems():
            if button.isDown():
                if self.plot_params.cmap != cmap:
                    self.plot_params.cmap = cmap
                    major = self.plot_config.cmap_major_colors[cmap]
                    minor = self.plot_config.cmap_minor_colors[cmap]
                    self.plot_objects.trace_sparkline.set_major_color(major)
                    self.plot_objects.trace_sparkline.set_minor_color(minor)
                    self.plot_objects.trace_base.set_color(major)
                    self.plot_objects.crosshair_x.set_color(major)
                    self.plot_objects.crosshair_y.set_color(major)
                    #self.plot_objects.camera_x.set_color(major)
                    self.full_redraw()

    def _on_channel_group_pressed(self):
        # type: () -> None
        '''
        Does NOT cause simple_rcoeff to be updated, because that's tied to
        a given set of picks, which belong to a channel.
        '''
        for channel, button in self.plot_objects.channel_buttons.iteritems():
            if button.isDown():
                if self.plot_params.channel != channel:
                    self.plot_params.channel = channel
                    # switch radar files!!
                    self.radar_data = RadarData(
                        self.pst, self.plot_params.product, channel)
                    self.plot_params.update_clim_from_radar(self.radar_data)
                    self.full_redraw()

    def _on_product_group_pressed(self):
        # type: () -> None
        """
        TODO
        """
        old_product = self.plot_params.product
        for new_product, button in self.plot_objects.product_buttons.iteritems():
            if button.isDown():
                if self.plot_params.product != new_product:
                    # Need to reason about updating the limits between 1m and pik1 data
                    prev_1m = radutils.radutils.product_is_1m(old_product)
                    new_1m = radutils.radutils.product_is_1m(new_product)
                    if new_1m:
                        self.plot_params.xunits = 'traces_1m'
                    else:
                        self.plot_params.xunits = 'traces_pik1'

                    prev_xlim = self.plot_params.curr_xlim

                    prev_num_traces = self.radar_data.num_traces
                    self.plot_params.product = new_product
                    self.radar_data = RadarData(
                        self.pst, new_product, self.plot_params.channel)
                    new_num_traces = self.radar_data.num_traces

                    # Converting bounds is trickier than you might think because
                    # it has to result in integers, which leads to propagating
                    # rounding errors.
                    if prev_1m and not new_1m:
                        # Have to be careful here to be consistent ... w/o the
                        # ceil, repeately converting between the two caused the
                        # boundaries to drift slightly.
                        new_xcoords = self.transect_data.rtc.convert(
                            list(prev_xlim), 'traces_1m', 'traces_pik1')
                        new_xlim = (np.ceil(new_xcoords[0]),
                                    np.ceil(new_xcoords[1]))
                    elif new_1m and not prev_1m:
                        # Have to be careful here - each pik1 sweep points to
                        # the middle of the range of 1m sweeps used to generate
                        # it, so we want to make the boundary the midpoint.
                        xcoords = [prev_xlim[0]-1, prev_xlim[0],
                                   prev_xlim[1]-1, prev_xlim[1]]
                        # This awkwardness is for mypy type checking - convert takes floats!
                        new_xcoords = self.transect_data.rtc.convert(
                            [float(xc) for xc in xcoords],
                            'traces_pik1', 'traces_1m')
                        new_xlim = (int(np.round(np.mean(new_xcoords[0:2]))),
                                    int(np.round(np.mean(new_xcoords[2:4]))))
                    else:
                        new_xlim = prev_xlim
                        # If we're at the start/end of the PST, want new display
                        # to also include all data ....
                    if prev_xlim[0] == 0:
                        new_xlim = (0, new_xlim[1])
                    if prev_xlim[-1] >= prev_num_traces - 1:
                        new_xlim = (new_xlim[0], new_num_traces - 1)

                    self.update_xlim(new_xlim)
                    self.plot_params.update_clim_from_radar(self.radar_data)
                    self.plot_params.rcoeff_needs_recalc = True

                    # This recalculates skip and sets data based on curr_xlim
                    self.full_redraw()

    def _on_max_type_group_pressed(self):
        # type: () -> None
        """
        TODO
        """
        for max_type, button in self.plot_objects.max_type_buttons.iteritems():
            if button.isDown():
                if max_type != self.plot_params.max_type:
                    self.plot_params.max_type = max_type
                    self.plot_params.simple_rcoeff_needs_recalc = True
                    self.data_blit()

    def _on_trace_checkbox_changed(self, val):
        # type: (int) -> None
        '''
        Registers / unregisters the trace callback.
        '''
        self.plot_params.trace_visible = self.plot_objects.trace_checkbox.isChecked()
        # Should be responsive when turned back on...
        if self.plot_params.trace_visible:
            self.plot_params.trace_frozen = False
        self.cursor_blit()

    def _on_crosshair_checkbox_changed(self, val):
        # type: (int) -> None
        """
        TODO
        """
        self.plot_params.crosshair_visible = self.plot_objects.crosshair_checkbox.isChecked()
        # Should be responsive when turned back on...
        if self.plot_params.crosshair_visible:
            self.plot_params.crosshair_frozen = False # want it responsive by default.
        self.cursor_blit()

    def _on_crossover_checkbox_changed(self, val):
        # type: (int) -> None
        """
        TODO
        """
        print("crossover checkbox changed")

    def _on_camera_checkbox_changed(self, val):
        # type: (int) -> None
        """
        TODO
        """
        print("camera checkbox changed")

    def _on_motion_notify_event(self, event):
        # type: (mpl.backend_bases.MotionNotifyEvent) -> None
        """
        TODO
        """
        #if event.inaxes is not self.plot_objects.pick_ax:
        if event.inaxes is not self.plot_objects.pick_ax:
            return

        trace, sample = self.radar_from_pick_coords((event.x, event.y))
        trace = int(np.round(np.min([self.radar_data.num_traces-1,
                                     np.max([0, trace])])))
        sample = int(np.round(np.min([self.radar_data.num_samples-1,
                                     np.max([0, sample])])))

        # blitting when neither crosshairs nor trace are active takes ~0.0005.
        # crosshairs is ~0.001, and trace is ~0.005, regardless of whether
        # they changed. So, checking if we need to blit saves up to 5ms.
        trace_changed = self.maybe_update_trace(trace)
        crosshair_changed = self.maybe_update_crosshair(trace, sample)
        if trace_changed or crosshair_changed:
            self.cursor_blit()

    def _on_clim_slider_changed(self, clim):
        # type: (Tuple[int, int]) -> None
        """
        TODO
        """
        self.plot_params.clim = clim
        self.full_redraw()

    def _on_pick_button1_clicked(self):
        # type: () -> None
        """
        TODO
        """
        if self.plot_params.active_horiz is None:
            return

        if self.plot_params.top_pick_status == PICK_OFF:
            self.plot_params.top_pick_status = PICK_ON
            self.plot_objects.pick_button1.setStyleSheet(
                'QPushButton {background-color: cyan}')
        elif self.plot_params.top_pick_status == PICK_ON:
            self.plot_params.top_pick_status = PICK_ACTIVE
            self.plot_objects.pick_button1.setStyleSheet(
                'QPushButton {background-color: darkCyan}')
            if self.plot_params.bottom_pick_status == PICK_ACTIVE:
                self.plot_params.bottom_pick_status = PICK_ON
                self.plot_objects.pick_button2.setStyleSheet(
                    'QPushButton {background-color: cyan}')
        elif self.plot_params.top_pick_status == PICK_ACTIVE:
            self.plot_params.top_pick_status = PICK_OFF
            self.plot_objects.pick_button1.setStyleSheet(
                'QPushButton {background-color: white}')

        self.data_blit()

    def _on_pick_button2_clicked(self):
        # type: () -> None
        """
        TODO
        """
        if self.plot_params.active_horiz is None:
            return

        if self.plot_params.bottom_pick_status == PICK_OFF:
            self.plot_params.bottom_pick_status = PICK_ON
            self.plot_objects.pick_button2.setStyleSheet(
                'QPushButton {background-color: cyan}')
        elif self.plot_params.bottom_pick_status == PICK_ON:
            self.plot_params.bottom_pick_status = PICK_ACTIVE
            self.plot_objects.pick_button2.setStyleSheet(
                'QPushButton {background-color: darkCyan}')
            if self.plot_params.top_pick_status == PICK_ACTIVE:
                self.plot_params.top_pick_status = PICK_ON
                self.plot_objects.pick_button1.setStyleSheet(
                    'QPushButton {background-color: cyan}')
        elif self.plot_params.bottom_pick_status == PICK_ACTIVE:
            self.plot_params.bottom_pick_status = PICK_OFF
            self.plot_objects.pick_button2.setStyleSheet(
                'QPushButton {background-color: white}')

        self.data_blit()

    def _on_auto_pick_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        if self.plot_params.active_horiz is None:
            return
        self.transect_data.pick_dict[self.plot_params.active_horiz].autopick()
        self.plot_objects.rci.activate_radio_checkbox()
        self.data_blit()

    def _on_save_picks_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        if self.plot_params.active_horiz is None:
            return
        picks = self.transect_data.pick_dict[self.plot_params.active_horiz]
        try:
            picks.autopick()
        except: # TODO: What exception does it throw??
            # We want to save even if the autopicker fails!
            pass
        print("saving picks! %s" % self.plot_params.active_horiz)
        radutils.pickutils.save_picks(picks)

    def _on_copy_picks_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        product = self.plot_params.product
        channel = self.plot_params.channel
        label = self.plot_objects.new_picks_textbox.text()

        if product == '1mfoc1':
            msg = "1mfoc1 doesn't support picking"
            plotUtilities.show_error_message_box(msg)
            return
        if self.plot_params.active_horiz is None:
            msg = "Please select picks to copy"
            plotUtilities.show_error_message_box(msg)
            return
        if label == '':
            msg = "Please enter name for new picks!"
            plotUtilities.show_error_message_box(msg)
            return
        pick_name = radutils.pickutils.make_display_name(product, channel,
                                                         label)

        if pick_name in self.transect_data.pick_dict.keys():
            msg = "Duplicate pickfile name - try again!"
            plotUtilities.show_error_message_box(msg)
            return
        self.plot_objects.new_picks_textbox.setText('')

        self.add_pick_label(pick_name)

        old_picks = self.transect_data.pick_dict[self.plot_params.active_horiz]

        new_top_pick, new_bottom_pick = self.convert_picks(old_picks, product)

        self.transect_data.pick_dict[pick_name] = radutils.pickutils.Picks(
            new_top_pick, new_bottom_pick, None, self.pst,
            self.plot_params.product, self.plot_params.channel, label)

        color = self.plot_objects.rci.get_color(pick_name)
        self.plot_objects.pick_lines[pick_name], = self.plot_objects.radar_ax.plot(0, 0, color=color)
        self.plot_params.pick_colors[pick_name] = color
        self.plot_params.pick_visible[pick_name] = False

    def convert_picks(self, old_picks, new_product):
        # type: (radutils.pickutils.Picks, str) -> Tuple[np.array, np.array]
        '''
        Returns the manually-labeled picks from old_picks that have been
        converted to the proper time scale to match new_product.
        '''
        # convert the manually-chosen picks
        if radutils.radutils.product_is_1m(old_picks.product):
            old_time_units = 'traces_1m'
        else:
            old_time_units = 'traces_pik1'
        if radutils.radutils.product_is_1m(new_product):
            new_time_units = 'traces_1m'
        else:
            new_time_units = 'traces_pik1'

        if old_picks.top is not None:
            top_sweeps = self.transect_data.rtc.convert(
                old_picks.top[:, 0], old_time_units, new_time_units)
            new_top_pick = np.array(zip(top_sweeps, old_picks.top[:, 1]))
        else:
            new_top_pick = None
        if old_picks.bottom is not None:
            bottom_sweeps = self.transect_data.rtc.convert(
                old_picks.bottom[:, 0], old_time_units, new_time_units)
            new_bottom_pick = np.array(zip(bottom_sweeps, old_picks.bottom[:, 1]))
        else:
            new_bottom_pick = None

        return new_top_pick, new_bottom_pick

    def _on_new_picks_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        product = self.plot_params.product
        channel = self.plot_params.channel
        label = self.plot_objects.new_picks_textbox.text()

        if product == '1mfoc1':
            msg = "1mfoc1 doesn't support picking"
            plotUtilities.show_error_message_box(msg)
            return
        if label == '':
            msg = "Please enter name for new picks!"
            plotUtilities.show_error_message_box(msg)
            return
        pick_name = radutils.pickutils.make_display_name(product, channel,
                                                         label)

        if pick_name in self.transect_data.pick_dict.keys():
            msg = "Duplicate pickfile name - try again!"
            plotUtilities.show_error_message_box(msg)
            return
        self.plot_objects.new_picks_textbox.setText('')

        self.add_pick_label(pick_name)
        self.transect_data.pick_dict[pick_name] = radutils.pickutils.Picks(
            None, None, None, self.pst, product, channel, label)

        color = self.plot_objects.rci.get_color(pick_name)
        self.plot_params.pick_colors[pick_name] = color
        self.plot_params.pick_visible[pick_name] = False
        self.plot_objects.pick_lines[pick_name], = self.plot_objects.radar_ax.plot(0, 0, color=color)

    def _on_active_horiz_changed(self, label):
        # type: (str) -> None
        """
        TODO
        """
        if self.plot_params.active_horiz != label:
            self.plot_params.active_horiz = label
            self.plot_params.simple_rcoeff_needs_recalc = True
            self.data_blit()

    def _on_pick_color_changed(self, label, color):
        # type: (str, QtGui.QColor) -> None
        """
        TODO
        """
        if self.plot_params.pick_colors[label] != color:
            self.plot_params.pick_colors[label] = color
            self.data_blit()

    def _on_pick_visible_changed(self, label, checked):
        # type: (str, bool) -> None
        """
        TODO
        """
        if self.plot_params.pick_visible[label] != checked:
            self.plot_params.pick_visible[label] = checked
            self.data_blit()

    def _on_quit_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        if self.close_cb is not None:
            self.close_cb()
        self.close()

    def _on_xevas_update_x(self, xmin, xmax):
        # type: (float, float) -> None
        ''' callback passed to the Xevas selector bars for updating xrange '''
        # xevas selector is [0,1], and we want [0, num_traces).
        # So, subtract 1 before multiplying!
        xlim = ((self.radar_data.num_traces-1)*xmin,
                (self.radar_data.num_traces-1)*xmax)
        self.update_xlim(xlim)
        self.full_redraw()

    def _on_xevas_update_y(self, ymin, ymax):
        # type: (float, float) -> None
        ''' callback passed to the Xevas selector bars for updating yrange '''
        # reversed y-axis ...
        # Also, xevas selector is [0,1], and we want [0, num_samples).
        # So, subtract 1 before multiplying!
        ylim = ((self.radar_data.num_samples-1)*(1-ymin),
                (self.radar_data.num_samples-1)*(1-ymax))
        self.update_ylim(ylim)
        self.full_redraw()

    # TODO: This code is sloppy about when to keep references or not.
    # I know that QObjects are deleted when they fall out of scope ... but
    # this seems to be working with my sporadic, inconsistent assignments to
    # plot_objects vs local vars ...
    def create_layout(self, plot_params, plot_config):
        # type: (PlotParams, PlotConfig) -> PlotObjects
        '''
        Only uses self for connecting callbacks, calling ._on* callbacks, and
        one QtGui call. Does not modify any variables.

        Parameters:
        * plot_params - includes min_val, max_val

        Returns:
        * plot_objects - has all of the objects that we created for the plot.
        '''

        # Set up figure & canvas
        plot_objects = PlotObjects()

        plot_objects.main_frame = QtGui.QWidget()
        # TODO: I'm not at all sure that this is the right way to handle it ...
        # Should it belong to the canvas instead?
        plot_objects.cursor = QtGui.QCursor()
        plot_objects.main_frame.setCursor(plot_objects.cursor)

        plot_objects.dpi = 100
        # Huh. This seems to only affect the vertical scale of the figure ...
        plot_objects.fig = Figure((18.0, 12.0), dpi=plot_objects.dpi)
        plot_objects.canvas = FigureCanvas(plot_objects.fig)
        # This can't go any earlier, or else I'll get errors about fig not having canvas
        plot_objects.fig.canvas.mpl_connect('resize_event',
                                            self._on_resize_event)
        # Used for save button + info about trace/sample of mouse position
        plot_objects.mpl_toolbar = mplUtilities.SaveToolbar(
            plot_objects.canvas, plot_objects.main_frame)

        # For now, I never want the canvas to have focus, so I can handle all
        # the keypresses through Qt. Some of the widgets can respond anyways,
        # (RectangleSelector, and motion_notify_event), and I use those.
        # https://srinikom.github.io/pyside-docs/PySide/QtGui/QWidget.html#PySide.QtGui.PySide.QtGui.QWidget.focusPolicy
        #plot_objects.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        plot_objects.canvas.setFocusPolicy(QtCore.Qt.NoFocus)

        plot_objects.canvas.setParent(plot_objects.main_frame)

        # Variables controlling the layout of the figure where the
        # radargram and xevas bars will be displayed.
        # space between fig edge and xevas bar, and xevas bar and radargram
        margin = 0.01
        # width of xevas bars
        zoom_width = 0.02

        radar_x0 = zoom_width + 2*margin
        radar_y0 = radar_x0
        radar_dx = 1.0 - radar_x0 - 6*margin
        radar_dy = 1.0 - radar_y0 - 6*margin

        # Used purely for blitting, so that the whole region gets restored
        # (sometimes, the sparkline text was going out-of-bounds)
        plot_objects.full_ax = plot_objects.fig.add_axes([0, 0, 1, 1])
        plot_objects.full_ax.axis('off')
        # Don't want to show anything when we're outside the pick_ax
        plot_objects.full_ax.format_coord = lambda x, y: ''

        plot_objects.radar_ax = plot_objects.fig.add_axes(
            [radar_x0, radar_y0, radar_dx, radar_dy], zorder=1, label='radar')
        plot_objects.radar_ax.xaxis.set_major_formatter(FuncFormatter(self.format_xlabel))
        plot_objects.radar_ax.yaxis.set_major_formatter(FuncFormatter(self.format_ylabel))
        plot_objects.radar_ax.minorticks_on()
        plot_objects.radar_ax.tick_params(which='both', direction='out',
                                          labelbottom=False, labeltop=True,
                                          labelleft=False, labelright=True,
                                          labelsize=8,
                                          bottom=False, top=True,
                                          left=False, right=True)
        #plot_objects.radar_ax.tick_params(which='major', length=10, width=1.5)
        #plot_objects.radar_ax.tick_params(which='minor', length=7, width=1)

        # These put it in the wrong spot. If I decide that I want 'em, look at:
        # stackoverflow.com/questions/9290938/how-to-set-my-xlabel-at-the-end-of-xaxis
        #plot_objects.radar_ax.set_xlabel('seconds')
        #plot_objects.radar_ax.set_ylabel('one-way distance in ice (m)')

        # In order to implement picking accepting values just outside the
        # actual axes (required to pick last trace), this axes extends past
        # the radargram, taking up the margin.
        plot_objects.pick_ax = plot_objects.fig.add_axes(
            [radar_x0 - margin, radar_y0, radar_dx + 2*margin, radar_dy],
            zorder=3, label='pick')
        plot_objects.pick_ax.axis('off')
        # This is the one that shows ... and it's called with units of 0-1
        plot_objects.pick_ax.format_coord = self.format_coord

        xmargin_frac = margin / radar_dx
        ymargin_frac = abs(margin / radar_dy)

        xevas_horiz_bounds = [radar_x0-margin, margin,
                              radar_dx+2*margin, zoom_width]
        plot_objects.xevas_horiz_ax = plot_objects.fig.add_axes(
            xevas_horiz_bounds, projection='unzoomable')
        plot_objects.xevas_horiz_ax.format_coord = lambda x, y: ''

        xevas_vert_bounds = [margin, radar_y0-margin,
                             zoom_width, radar_dy+2*margin]
        plot_objects.xevas_vert_ax = plot_objects.fig.add_axes(
            xevas_vert_bounds, projection='unzoomable')
        plot_objects.xevas_vert_ax.format_coord = lambda x, y: ''

        # Have to give 0-1 and 1-0 for these to be agnostic to changing x units.
        plot_objects.xevas_horiz = mplUtilities.XevasHorizSelector(
            plot_objects.xevas_horiz_ax, 0, 1.0, self._on_xevas_update_x,
            margin_frac=xmargin_frac)
        plot_objects.xevas_vert = mplUtilities.XevasVertSelector(
            plot_objects.xevas_vert_ax, 0, 1.0, self._on_xevas_update_y,
            margin_frac=ymargin_frac)

        plot_objects.top_picks, = plot_objects.radar_ax.plot(
            0, 0, 'sk', linestyle='-', markerfacecolor='none')
        plot_objects.bottom_picks, = plot_objects.radar_ax.plot(
            0, 0, 'sk', linestyle='-', markerfacecolor='none')

        # Crosshairs for showing where mouse is on radargram, linked with
        # display on the main deva plot.
        plot_objects.crosshair_x, = plot_objects.radar_ax.plot(
            0, 0, 'r', linestyle=':', linewidth=2)
        plot_objects.crosshair_y, = plot_objects.radar_ax.plot(
            0, 0, 'r', linestyle=':', linewidth=2)

        major_color = self.plot_config.cmap_major_colors[self.plot_params.cmap]
        minor_color = self.plot_config.cmap_minor_colors[self.plot_params.cmap]
        plot_objects.trace_sparkline = plotutils.sparkline.Sparkline(
            plot_objects.radar_ax, units='dB',
            major_color=major_color, minor_color=minor_color,
            scalebar_pos=[0.85, 0.95], scalebar_len=20,
            plot_width=.0625, plot_offset=0, data_axis='y')
        plot_objects.trace_base, = plot_objects.radar_ax.plot(
            0, 0, 'r', linestyle='--')

        # TODO: These wil have to be connected to pick_ax, which will
        #  be on top of the various pcor axes.
        # (they only select if it's the top axis ... and the radar one
        # can't be top if we're creating new axes for hte pcor stuff...)
        # (I'm wondering if I'll need to put them on the full_ax, then convert
        # coordinates, in order to support clicking just beyond the axis to get
        # the end of the line...or at least, the pick one...)

        # I'm doing it like this because I want all of 'em to have different
        # line styles....
        plot_objects.left_click_rs['pick'] = mpw.RectangleSelector(
            plot_objects.pick_ax, self._on_left_rect_click_pick,
            drawtype='none', button=[1])
        plot_objects.right_click_rs['pick'] = mpw.RectangleSelector(
            plot_objects.pick_ax, self._on_right_rect_click_pick,
            drawtype='line', button=[3])
        # TODO: It'd be cool to use rectprops to change linestyle between
        # zoom in/out, but I didn't immediately see how to do so.
        plot_objects.left_click_rs['zoom'] = mpw.RectangleSelector(
            plot_objects.pick_ax, self._on_left_rect_click_zoom,
            drawtype='box', button=[1])
        plot_objects.right_click_rs['zoom'] = mpw.RectangleSelector(
            plot_objects.pick_ax, self._on_right_rect_click_zoom,
            drawtype='box', button=[3])
        # Pan is the same for both of 'em (it's easier this way)
        plot_objects.left_click_rs['pan'] = mpw.RectangleSelector(
            plot_objects.pick_ax, self._on_rect_click_pan,
            drawtype='line', button=[1])
        plot_objects.right_click_rs['pan'] = mpw.RectangleSelector(
            plot_objects.pick_ax, self._on_rect_click_pan,
            drawtype='line', button=[3])
        for artist in plot_objects.left_click_rs.values():
            artist.set_active(False)
        for artist in plot_objects.right_click_rs.values():
            artist.set_active(False)

        # This used to be connected/disconnected as the various lines were
        # activated/deactivated, but now that a single one is controlling all
        # of 'em, it's simpler to just leave it connected. Only change that if
        # it turns into a bottleneck...
        plot_objects.canvas.mpl_connect('motion_notify_event',
                                        self._on_motion_notify_event)

        # Radio buttons for controlling what mouse clicks mean!
        # (This used to be done w/ their toolbar, but I wanted it to be
        # more explicit, and to have more control on when axes were redrawn.)
        # (Hopefully this also gets rid of the weirdness that ensued when
        # trying to zoom on the xevas bars ... we'll see ...)
        plot_objects.mouse_mode_buttons = {}

        plot_objects.mouse_mode_group = QtGui.QButtonGroup()
        mouse_mode_hbox = QtGui.QHBoxLayout()
        for mode in ['pick', 'zoom', 'pan']:
            button = QtGui.QRadioButton(mode)
            plot_objects.mouse_mode_buttons[mode] = button
            plot_objects.mouse_mode_group.addButton(button)
            mouse_mode_hbox.addWidget(button)
        self.connect(plot_objects.mouse_mode_group,
                     QtCore.SIGNAL('buttonPressed(int)'),
                     self._on_mouse_mode_group_pressed)

        # Create buttons!
        plot_objects.prev_button = QtGui.QPushButton('Prev (e)')
        self.connect(plot_objects.prev_button, QtCore.SIGNAL('clicked()'),
                     self._on_prev_button_clicked)

        plot_objects.full_button = QtGui.QPushButton('Full (y)')
        self.connect(plot_objects.full_button, QtCore.SIGNAL('clicked()'),
                     self._on_full_button_clicked)

        plot_objects.next_button = QtGui.QPushButton('Next (r)')
        self.connect(plot_objects.next_button, QtCore.SIGNAL('clicked()'),
                     self._on_next_button_clicked)

        controls_hbox = QtGui.QHBoxLayout()
        controls_hbox.addWidget(plot_objects.mpl_toolbar)
        controls_hbox.addStretch(1)
        controls_hbox.addLayout(mouse_mode_hbox)
        controls_hbox.addWidget(plot_objects.prev_button)
        controls_hbox.addWidget(plot_objects.full_button)
        controls_hbox.addWidget(plot_objects.next_button)

        data_vbox = QtGui.QVBoxLayout()
        data_vbox.addWidget(plot_objects.canvas)
        data_vbox.addLayout(controls_hbox)


        ####
        # All of the control on the right half of the window
        # This includes sub-boxes for:
        # * appearance (colorscale, data product, channel)
        # * data range sliders
        # * pick mode (pick1/pick2/save)
        # * loading new pick file
        # * which maxima to show
        # * which picks are active

        # switching colormaps

        appearance_grid = QtGui.QGridLayout()

        colormap_col = 0
        channel_col = 1
        product_col = 2
        max_type_col = 3

        colormap_row = 0
        channel_row = 0
        product_row = 0
        max_type_row = 0

        for colormap in plot_config.all_cmaps:
            plot_objects.colormap_buttons[colormap] = QtGui.QRadioButton(colormap)

        plot_objects.colormap_group = QtGui.QButtonGroup()
        for cmap, button in plot_objects.colormap_buttons.iteritems():
            plot_objects.colormap_group.addButton(button)
            appearance_grid.addWidget(button, colormap_row, colormap_col)
            colormap_row += 1
        self.connect(plot_objects.colormap_group,
                     QtCore.SIGNAL('buttonPressed(int)'),
                     self._on_colormap_group_pressed)

        # switching which channel to display
        for chan in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            plot_objects.channel_buttons[chan] = QtGui.QRadioButton('chan %d' % chan)
            if chan not in plot_config.available_channels:
                plot_objects.channel_buttons[chan].setEnabled(False)
        plot_objects.channel_group = QtGui.QButtonGroup()
        for chan, button in plot_objects.channel_buttons.iteritems():
            plot_objects.channel_group.addButton(button)
            appearance_grid.addWidget(button, channel_row, channel_col)
            channel_row += 1
        self.connect(plot_objects.channel_group,
                     QtCore.SIGNAL('buttonPressed(int)'),
                     self._on_channel_group_pressed)

        # switching which product to display; all options are displayed,
        # but only ones with available data will wind up active.
        for product in plot_config.all_products:
            plot_objects.product_buttons[product] = QtGui.QRadioButton(product)
            plot_objects.product_buttons[product].setEnabled(False)
        plot_objects.product_group = QtGui.QButtonGroup()
        for product, button in plot_objects.product_buttons.iteritems():
            plot_objects.product_group.addButton(button)
            appearance_grid.addWidget(button, product_row, product_col)
            product_row += 1
        self.connect(plot_objects.product_group,
                     QtCore.SIGNAL('buttonPressed(int)'),
                     self._on_product_group_pressed)

        # Whether to display max or vmax picks
        for pick in ['vmax', 'max']:
            plot_objects.max_type_buttons[pick] = QtGui.QRadioButton(pick)

        plot_objects.max_type_group = QtGui.QButtonGroup()
        for pick, button in plot_objects.max_type_buttons.iteritems():
            plot_objects.max_type_group.addButton(button)
            appearance_grid.addWidget(button, max_type_row, max_type_col)
            max_type_row += 1
        self.connect(plot_objects.max_type_group,
                     QtCore.SIGNAL('buttonPressed(int)'),
                     self._on_max_type_group_pressed)


        # enable/disable the various wriggles plotted on the radargram (traces/rcoeff)
        plot_objects.wiggles_hbox = QtGui.QHBoxLayout()
        plot_objects.trace_checkbox = QtGui.QCheckBox('Traces')
        self.connect(plot_objects.trace_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_trace_checkbox_changed)
        plot_objects.crosshair_checkbox = QtGui.QCheckBox('Crosshairs')
        self.connect(plot_objects.crosshair_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_crosshair_checkbox_changed)
        plot_objects.wiggles_hbox.addWidget(plot_objects.trace_checkbox)
        plot_objects.wiggles_hbox.addWidget(plot_objects.crosshair_checkbox)
        plot_objects.wiggles_hbox.addStretch(1) # left-justify this rowx

        ############

        plot_objects.clim_slider = qtWidgets.DoubleSlider(
            new_lim_cb=self._on_clim_slider_changed)

        ############
        # Tab for buttons controlling picking
        plot_objects.pick_button1 = QtGui.QPushButton('top')
        self.connect(plot_objects.pick_button1, QtCore.SIGNAL('clicked()'),
                     self._on_pick_button1_clicked)
        plot_objects.pick_button2 = QtGui.QPushButton('bottom')
        self.connect(plot_objects.pick_button2, QtCore.SIGNAL('clicked()'),
                     self._on_pick_button2_clicked)
        plot_objects.auto_pick_button = QtGui.QPushButton('Autopick')
        self.connect(plot_objects.auto_pick_button, QtCore.SIGNAL('clicked()'),
                     self._on_auto_pick_button_clicked)
        plot_objects.save_picks_button = QtGui.QPushButton('Save Picks')
        self.connect(plot_objects.save_picks_button, QtCore.SIGNAL('clicked()'),
                     self._on_save_picks_button_clicked)

        edit_pick_hbox = QtGui.QHBoxLayout()
        edit_pick_hbox.addStretch(1)
        edit_pick_hbox.addWidget(plot_objects.pick_button1)
        edit_pick_hbox.addWidget(plot_objects.pick_button2)
        edit_pick_hbox.addWidget(plot_objects.auto_pick_button)
        edit_pick_hbox.addWidget(plot_objects.save_picks_button)

        # Sub-section for starting a new pick file.
        plot_objects.copy_picks_button = QtGui.QPushButton('Copy Pick')
        self.connect(plot_objects.copy_picks_button,
                     QtCore.SIGNAL('clicked()'),
                     self._on_copy_picks_button_clicked)
        plot_objects.new_picks_textbox = QtGui.QLineEdit()
        plot_objects.new_picks_textbox.setMinimumWidth(100)
        plot_objects.new_picks_button = QtGui.QPushButton('New Pick')
        self.connect(plot_objects.new_picks_button,
                     QtCore.SIGNAL('clicked()'),
                     self._on_new_picks_button_clicked)
        new_pick_hbox = QtGui.QHBoxLayout()
        new_pick_hbox.addWidget(plot_objects.copy_picks_button)
        new_pick_hbox.addStretch(1)
        new_pick_hbox.addWidget(plot_objects.new_picks_textbox)
        new_pick_hbox.addWidget(plot_objects.new_picks_button)


        # Sub-section for controlling which pick file is active,
        # and which ones have maxima displayed
        show_picks_hbox = QtGui.QHBoxLayout()
        plot_objects.rci = qtWidgets.RadioCheckInterface(
            parent=self,
            radio_cb=self._on_active_horiz_changed,
            check_cb=self._on_pick_visible_changed,
            color_cb=self._on_pick_color_changed)
        # An attempt to de-select any picks!
        # TODO: This would be cleaner if built in to the widget.
        plot_objects.rci.add_row(None)
        plot_objects.rci.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                       QtGui.QSizePolicy.Expanding)

        plot_objects.pick_scroll_area = QtGui.QScrollArea()
        plot_objects.pick_scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn)
        plot_objects.pick_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        plot_objects.pick_scroll_area.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                                    QtGui.QSizePolicy.Expanding)
        plot_objects.pick_scroll_area.setWidgetResizable(True)
        plot_objects.pick_scroll_area.setWidget(plot_objects.rci)

        #show_picks_hbox.addWidget(plot_objects.rci)
        show_picks_hbox.addWidget(plot_objects.pick_scroll_area)

        # Button to exit (the little one in the corner is a PITA.
        quit_hbox = QtGui.QHBoxLayout()
        plot_objects.quit_button = QtGui.QPushButton('Quit')
        self.connect(plot_objects.quit_button,
                     QtCore.SIGNAL('clicked()'),
                     self._on_quit_button_clicked)
        quit_hbox.addStretch(1)
        quit_hbox.addWidget(plot_objects.quit_button)


        # Making the tabs...
        picking_vbox = QtGui.QVBoxLayout()
        picking_vbox.addLayout(edit_pick_hbox)
        picking_vbox.addLayout(new_pick_hbox)
        picking_vbox.addWidget(plot_objects.pick_scroll_area)

        picking_widget = QtGui.QWidget()
        picking_widget.setLayout(picking_vbox)


        plot_objects.tabs = QtGui.QTabWidget()
        plot_objects.tabs.addTab(picking_widget, "Picking")


        # Assembling the right vbox ...
        control_vbox = QtGui.QVBoxLayout()
        control_vbox.addLayout(appearance_grid)
        control_vbox.addWidget(plotUtilities.HLine())
        control_vbox.addLayout(plot_objects.wiggles_hbox)
        control_vbox.addWidget(plotUtilities.HLine())
        control_vbox.addWidget(plot_objects.clim_slider)
        control_vbox.addWidget(plotUtilities.HLine())
        control_vbox.addWidget(plot_objects.tabs)
        control_vbox.addLayout(quit_hbox)

        ####
        # Put it all together.
        hbox = QtGui.QHBoxLayout()
        hbox.addLayout(data_vbox, 1) # I want this one to stretch
        hbox.addLayout(control_vbox)
        plot_objects.main_frame.setLayout(hbox)

        self.setCentralWidget(plot_objects.main_frame)

        #############

        return plot_objects

    def format_xlabel(self, xx, pos):
        # type: (float, int) -> str
        '''
        This maps traces to time since start of transect.
        It would be cool to also do distance, but that requires pulling more
        data sources into radarFigure than I'm comfortable with ... it's meant
        to be as lightweight as possible for looking at radar data.
        '''
        # TODO: Why do we have the pos argument that's not used?
        # TODO: This needs to obey whatever x-axis is currently in use
        # TODO: Why doesn't this just use self.plot_params.xunits????
        trace_units = 'traces_pik1'
        if radutils.radutils.product_is_1m(self.plot_params.product):
            trace_units = 'traces_1m'
        #if radutils.radutils.pst_is_cresis(self.pst):
        #    trace_units = 'linear'

        try:
            t0, = self.transect_data.rtc.convert([0], trace_units, 'posix')
            t1, = self.transect_data.rtc.convert([xx], trace_units, 'posix')
            dist = self.transect_data.rpc.along_track_dist([0, xx], trace_units)
            minutes, seconds = divmod(t1-t0, 60)
            #return '%d\n%0.1f km\n%02d:%02.2f' % (xx, dist[0]/1000., minutes, seconds)
            return '%0.1f km\n%02d:%02.2f\n%d' % (dist[0]/1000., minutes,
                                                  seconds, xx)
        except radutils.conversions.TimeConversionError:
            return str(xx)

    # TODO: This is wrong for cresis data - should actually get the sampling
    # rate from the params.
    def format_ylabel(self, yy, pos):
        # type: (float, float) -> str
        '''
        We sample at 50MHz, so each sample is 20ns, or 0.02us
        In ice, this translates to a one-way distance of 0.02*169/2
        '''
        dt = yy * 0.02
        dz = dt*169*0.5
        #return '%d\n%0.2f us\n%d m' % (yy, dt, dz)
        return '%0.2f us\n%d m' % (dt, dz)

    def format_coord(self, xx, yy):
        # type: (float, float) -> str
        coord = self.plot_objects.pick_ax.transData.transform([xx, yy])
        trace, sample = self.radar_from_pick_coords(coord)
        try:
            counts = self.radar_data.data[trace, sample]
        except IndexError as ex:
            counts = None
            print(ex)
            print("trace: %r, %r" % (trace, type(trace)))
            print("sample: %r, %r" % (sample, type(sample)))
            print("radar data type: %r" % type(radar_data))
        return 'trace=%d sample=%d (%d counts)' % (trace, sample, counts)

class ExperimentalRadarWindow(BasicRadarWindow):
    def __init__(self,
                 pst, # type: str
                 filename=None, # type: Optional[str]
                 parent=None, # type: Optional[Any]
                 parent_xlim_changed_cb=None, # type: Optional[Callable[List[float]]]
                 parent_cursor_cb=None, # type: Optional[Callable[float]]
                 close_cb=None # type: Optional[Callable[None]]
                 ):
        # type: (...) -> None
        """
        TODO
        """
        super(ExperimentalRadarWindow, self).__init__(pst, filename, parent,
                                                      parent_xlim_changed_cb,
                                                      parent_cursor_cb,
                                                      close_cb)
        # TODO: do we care about gps_start, gps_end?
        # They would be used to allow interpolating from traces to linear time,
        # where the time bounds were from the data in targ/xtra/ALL/deva/psts.
        # Usage would be transect_data.rtc.set_linear_bounds(gps_start, gps_end),
        # and then if posix fails, return linear when either of the parent
        # positioning callbacks is called.
        # NB - for now, a hack in conversions.py means that it loads
        # time/positions from deva/psts, so there's no need to pass 'em in here.

    def create_layout(self, plot_params, plot_config):
        # type: (PlotParams, PlotConfig) -> PlotObjects
        """
        TODO
        """
        plot_objects = super(ExperimentalRadarWindow, self).create_layout(plot_params, plot_config)

        print("called add_experimental_layout")
        # Line for showing where camera image is.
        plot_objects.camera_x, = plot_objects.radar_ax.plot(
            0, 0, 'r', linestyle='-.', linewidth=2)

        major_color = self.plot_config.cmap_major_colors[self.plot_params.cmap]
        minor_color = self.plot_config.cmap_minor_colors[self.plot_params.cmap]

        plot_objects.simple_rcoeff_sparkline = plotutils.sparkline.Sparkline(
            plot_objects.radar_ax, units='dB',
            major_color=major_color, minor_color=minor_color,
            scalebar_pos=[0.05, 0.95], scalebar_len=20,
            plot_width=0.125, plot_offset=0.825)

        plot_objects.rcoeff_sparkline = plotutils.sparkline.Sparkline(
            plot_objects.radar_ax, units='dB',
            major_color=major_color, minor_color=minor_color,
            scalebar_pos=[0.05, 0.95], scalebar_len=20,
            plot_width=0.125, plot_offset=0.825, data_axis='x')

        plot_objects.multiple_line, = plot_objects.radar_ax.plot(
            0, 0, 'r', linestyle='--')

        plot_objects.vert_scale = plotutils.scalebar.Scalebar(
            plot_objects.radar_ax, 0, 0, 0, 0.01, fontsize=24, majorcolor='r',
            barstyle='simple', coords='frac', orientation='vert', linewidth=4,
            unit_label='m', autoupdate=False)

        plot_objects.horiz_scale = plotutils.scalebar.Scalebar(
            plot_objects.radar_ax, 0, 0, 0, 0.01, fontsize=24, majorcolor='r',
            barstyle='simple', coords='frac', orientation='horiz', linewidth=4,
            unit_label='km', autoupdate=False)

        plot_objects.crossover_checkbox = QtGui.QCheckBox('Crossovers')
        self.connect(plot_objects.crossover_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_crossover_checkbox_changed)
        plot_objects.crossover_checkbox.setEnabled(False)
        plot_objects.camera_checkbox = QtGui.QCheckBox('Camera')
        #TODO: Enable it once I'm ready to show camera data!!
        plot_objects.camera_checkbox.setEnabled(False)
        self.connect(plot_objects.camera_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_camera_checkbox_changed)

        plot_objects.wiggles_hbox.addWidget(plot_objects.crossover_checkbox)


        ######
        ##########
        # Tab for various analysis experiments

        plot_objects.simple_rcoeff_checkbox = QtGui.QCheckBox('Simple Rcoeff')
        self.connect(plot_objects.simple_rcoeff_checkbox,
                     #QtCore.SIGNAL('stateChanged(int)'),
                     QtCore.SIGNAL('clicked()'),
                     self._on_simple_rcoeff_checkbox_changed)

        # sub-panel for reflection coefficient.
        # (Unlike the one above, this one computes spreading and ice losses)
        # two rows - first is just enable/disable, 2nd has boxes.
        rcoeff_vbox = QtGui.QVBoxLayout()
        rcoeff_hbox1 = QtGui.QHBoxLayout()
        rcoeff_hbox2 = QtGui.QHBoxLayout()
        rcoeff_vbox.addLayout(rcoeff_hbox1)
        rcoeff_vbox.addLayout(rcoeff_hbox2)

        # checkbox for enable/disable
        plot_objects.rcoeff_checkbox = QtGui.QCheckBox('RCoeff')
        # Augh. stateChanged didn't toggle for two clicks after
        # setChecked(False); clicked() fires on setChecked()
        self.connect(plot_objects.rcoeff_checkbox,
                     #QtCore.SIGNAL('stateChanged(int)'),
                     QtCore.SIGNAL('clicked()'),
                     self._on_rcoeff_checkbox_changed)

        # Way to force it to recalculate
        plot_objects.rcoeff_recalc_button = QtGui.QPushButton('recalc')
        self.connect(plot_objects.rcoeff_recalc_button,
                     QtCore.SIGNAL('clicked()'),
                     self._on_rcoeff_recalc_button_clicked)

        # Drop-down menus for actually controlling the thing.
        plot_objects.rcoeff_srf_label = QtGui.QLabel('srf')
        plot_objects.rcoeff_srf_combo = QtGui.QComboBox()
        plot_objects.rcoeff_srf_combo.addItem('')
        plot_objects.rcoeff_srf_combo.activated[str].connect(
            self._on_rcoeff_srf_combo_activated)

        plot_objects.rcoeff_bed_label = QtGui.QLabel('bed')
        plot_objects.rcoeff_bed_combo = QtGui.QComboBox()
        plot_objects.rcoeff_bed_combo.addItem('')
        plot_objects.rcoeff_bed_combo.activated[str].connect(
            self._on_rcoeff_bed_combo_activated)

        plot_objects.rcoeff_iceloss_label = QtGui.QLabel('L_ice (dB/km, one-way)')
        plot_objects.rcoeff_iceloss_textbox = QtGui.QLineEdit()
        plot_objects.rcoeff_iceloss_textbox.setMinimumWidth(40)
        plot_objects.rcoeff_iceloss_textbox.setMaximumWidth(60)
        self.connect(plot_objects.rcoeff_iceloss_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_rcoeff_iceloss_textbox_edited)

        rcoeff_hbox2.addStretch(1)
        rcoeff_hbox2.addWidget(plot_objects.rcoeff_srf_label)
        rcoeff_hbox2.addWidget(plot_objects.rcoeff_srf_combo)
        rcoeff_hbox2.addStretch(1)
        rcoeff_hbox2.addWidget(plot_objects.rcoeff_bed_label)
        rcoeff_hbox2.addWidget(plot_objects.rcoeff_bed_combo)

        rcoeff_hbox1.addWidget(plot_objects.rcoeff_checkbox)
        rcoeff_hbox1.addStretch(1)
        rcoeff_hbox1.addWidget(plot_objects.rcoeff_recalc_button)
        rcoeff_hbox1.addStretch(1)
        rcoeff_hbox1.addWidget(plot_objects.rcoeff_iceloss_label)
        rcoeff_hbox1.addWidget(plot_objects.rcoeff_iceloss_textbox)

        # Sub-section for computing and displaying the hydrostatic line. Will
        # use the laser data, if it exists, the selected bed pick, the known
        # geoid correction, and ___ firn model.
        hydrostatic_hbox = QtGui.QHBoxLayout()
        plot_objects.hydrostatic_checkbox = QtGui.QCheckBox('Hydrostatic')
        plot_objects.hydrostatic_checkbox.setEnabled(False)
        self.connect(plot_objects.hydrostatic_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_hydrostatic_checkbox_changed)
        plot_objects.hydrostatic_bed_label = QtGui.QLabel('bed')
        plot_objects.hydrostatic_bed_combo = QtGui.QComboBox()
        plot_objects.hydrostatic_bed_combo.addItem('')
        plot_objects.hydrostatic_bed_combo.activated[str].connect(
            self._on_hydrostatic_bed_combo_activated)

        hydrostatic_hbox.addWidget(plot_objects.hydrostatic_checkbox)
        hydrostatic_hbox.addStretch(1)
        hydrostatic_hbox.addWidget(plot_objects.hydrostatic_bed_label)
        hydrostatic_hbox.addWidget(plot_objects.hydrostatic_bed_combo)

        # Sub-section for computing and displaying the expected multiple.
        # Will use the selected picks to compute surface/bed range.
        multiple_vbox = QtGui.QVBoxLayout()
        multiple1_hbox = QtGui.QHBoxLayout()
        multiple2_hbox = QtGui.QHBoxLayout()
        multiple_vbox.addLayout(multiple1_hbox)
        multiple_vbox.addLayout(multiple2_hbox)
        # checkbox for enable/disable
        plot_objects.multiple_checkbox = QtGui.QCheckBox('Multiple')
        self.connect(plot_objects.multiple_checkbox,
                     #QtCore.SIGNAL('stateChanged(int)'),
                     QtCore.SIGNAL('clicked()'),
                     self._on_multiple_checkbox_changed)

        # Way to force it to recalculate
        plot_objects.multiple_recalc_button = QtGui.QPushButton('recalc')
        self.connect(plot_objects.multiple_recalc_button,
                     QtCore.SIGNAL('clicked()'),
                     self._on_multiple_recalc_button_clicked)

        # Drop-down menus for actually controlling the thing.
        plot_objects.multiple_srf_label = QtGui.QLabel('srf')
        plot_objects.multiple_srf_combo = QtGui.QComboBox()
        plot_objects.multiple_srf_combo.addItem('')
        plot_objects.multiple_srf_combo.activated[str].connect(
            self._on_multiple_srf_combo_activated)

        plot_objects.multiple_bed_label = QtGui.QLabel('bed')
        plot_objects.multiple_bed_combo = QtGui.QComboBox()
        plot_objects.multiple_bed_combo.addItem('')
        plot_objects.multiple_bed_combo.activated[str].connect(
            self._on_multiple_bed_combo_activated)

        plot_objects.multiple_srf_count_label = QtGui.QLabel('plane-srf')
        plot_objects.multiple_srf_count_textbox = QtGui.QLineEdit()
        plot_objects.multiple_srf_count_textbox.setMinimumWidth(40)
        plot_objects.multiple_srf_count_textbox.setMaximumWidth(60)
        # TODO: If I really want to initialize these, that needs to come
        # from plot_params
        #plot_objects.multiple_srf_count_textbox.setText('1')
        self.connect(plot_objects.multiple_srf_count_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_multiple_srf_count_textbox_edited)

        plot_objects.multiple_bed_count_label = QtGui.QLabel('srf-bed')
        plot_objects.multiple_bed_count_textbox = QtGui.QLineEdit()
        plot_objects.multiple_bed_count_textbox.setMinimumWidth(40)
        plot_objects.multiple_bed_count_textbox.setMaximumWidth(60)
        #plot_objects.multiple_bed_count_textbox.setText('0')
        self.connect(plot_objects.multiple_bed_count_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_multiple_bed_count_textbox_edited)


        multiple1_hbox.addWidget(plot_objects.multiple_checkbox)
        multiple1_hbox.addStretch(1)
        multiple1_hbox.addWidget(plot_objects.multiple_recalc_button)
        multiple1_hbox.addStretch(1)
        multiple1_hbox.addWidget(plot_objects.multiple_srf_count_label)
        multiple1_hbox.addWidget(plot_objects.multiple_srf_count_textbox)
        multiple1_hbox.addStretch(1)
        multiple1_hbox.addWidget(plot_objects.multiple_bed_count_label)
        multiple1_hbox.addWidget(plot_objects.multiple_bed_count_textbox)

        multiple2_hbox.addStretch(1)
        multiple2_hbox.addWidget(plot_objects.multiple_srf_label)
        multiple2_hbox.addWidget(plot_objects.multiple_srf_combo)
        multiple2_hbox.addStretch(1)
        multiple2_hbox.addWidget(plot_objects.multiple_bed_label)
        multiple2_hbox.addWidget(plot_objects.multiple_bed_combo)

        ##### The stuff that goes in the elsa tab
        plot_objects.elsa_data_combo = QtGui.QComboBox()
        plot_objects.elsa_data_combo.addItem('')
        plot_objects.new_elsa_data_button = QtGui.QPushButton('Add')
        self.connect(plot_objects.new_elsa_data_button,
                     QtCore.SIGNAL('clicked()'),
                     self._on_new_elsa_button_clicked)

        new_elsa_hbox = QtGui.QHBoxLayout()
        new_elsa_hbox.addWidget(plot_objects.elsa_data_combo)
        new_elsa_hbox.addStretch(1)
        new_elsa_hbox.addWidget(plot_objects.new_elsa_data_button)

        # This is where the various rows will be added for the scatter plots
        plot_objects.elsa_widget = qtWidgets.TextColorInterface(
            color_cb=self._elsa_color_cb, params_cb=self._elsa_params_cb,
            remove_cb=self._elsa_remove_row_cb)

        elsa_vbox = QtGui.QVBoxLayout()
        elsa_vbox.addWidget(plot_objects.camera_checkbox)
        elsa_vbox.addWidget(plotUtilities.HLine())
        elsa_vbox.addLayout(new_elsa_hbox)
        elsa_vbox.addWidget(plotUtilities.HLine())
        elsa_vbox.addWidget(plot_objects.elsa_widget)
        elsa_vbox.addStretch(1)

        elsa_widget = QtGui.QWidget()
        elsa_widget.setLayout(elsa_vbox)


        analysis_vbox = QtGui.QVBoxLayout()
        analysis_vbox.addWidget(plot_objects.simple_rcoeff_checkbox)
        analysis_vbox.addWidget(plotUtilities.HLine())
        analysis_vbox.addLayout(rcoeff_vbox)
        analysis_vbox.addWidget(plotUtilities.HLine())
        analysis_vbox.addLayout(hydrostatic_hbox)
        analysis_vbox.addWidget(plotUtilities.HLine())
        analysis_vbox.addLayout(multiple_vbox)
        analysis_vbox.addWidget(plotUtilities.HLine())
        analysis_vbox.addStretch(1)

        analysis_widget = QtGui.QWidget()
        analysis_widget.setLayout(analysis_vbox)


        # And, all of the inputs for configuring scale bars
        plot_objects.vert_scale_checkbox = QtGui.QCheckBox('Vertical Scale')
        self.connect(plot_objects.vert_scale_checkbox,
                     QtCore.SIGNAL('clicked()'),
                     self._on_vert_scale_checkbox_changed)
        plot_objects.vert_scale_length_label = QtGui.QLabel('Length: (m)')
        plot_objects.vert_scale_length_textbox = QtGui.QLineEdit()
        self.connect(plot_objects.vert_scale_length_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_vert_scale_length_textbox_edited)
        plot_objects.vert_scale_length_textbox.setText("%0.2f" % plot_params.vert_scale_length)
        plot_objects.vert_scale_length_textbox.setMinimumWidth(100)
        plot_objects.vert_scale_length_textbox.setMaximumWidth(120)
        plot_objects.vert_scale_origin_label = QtGui.QLabel('origin (x, y) (fraction):')
        plot_objects.vert_scale_x0_textbox = QtGui.QLineEdit()
        self.connect(plot_objects.vert_scale_x0_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_vert_scale_x0_textbox_edited)
        plot_objects.vert_scale_x0_textbox.setText("%0.2f" % plot_params.vert_scale_x0)
        plot_objects.vert_scale_x0_textbox.setMinimumWidth(60)
        plot_objects.vert_scale_x0_textbox.setMaximumWidth(80)
        plot_objects.vert_scale_y0_textbox = QtGui.QLineEdit()
        self.connect(plot_objects.vert_scale_y0_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_vert_scale_y0_textbox_edited)
        plot_objects.vert_scale_y0_textbox.setText("%0.2f" % plot_params.vert_scale_y0)
        plot_objects.vert_scale_y0_textbox.setMinimumWidth(60)
        plot_objects.vert_scale_y0_textbox.setMaximumWidth(80)

        plot_objects.horiz_scale_checkbox = QtGui.QCheckBox('Horizontal Scale')
        self.connect(plot_objects.horiz_scale_checkbox,
                     QtCore.SIGNAL('clicked()'),
                     self._on_horiz_scale_checkbox_changed)
        plot_objects.horiz_scale_length_label = QtGui.QLabel('Length: (km)')
        plot_objects.horiz_scale_length_textbox = QtGui.QLineEdit()
        self.connect(plot_objects.horiz_scale_length_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_horiz_scale_length_textbox_edited)
        plot_objects.horiz_scale_length_textbox.setText("%0.2f" % plot_params.horiz_scale_length)
        plot_objects.horiz_scale_length_textbox.setMinimumWidth(100)
        plot_objects.horiz_scale_length_textbox.setMaximumWidth(120)

        plot_objects.horiz_scale_origin_label = QtGui.QLabel('origin (x, y) (fraction):')
        plot_objects.horiz_scale_x0_textbox = QtGui.QLineEdit()
        self.connect(plot_objects.horiz_scale_x0_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_horiz_scale_x0_textbox_edited)
        plot_objects.horiz_scale_x0_textbox.setText("%0.2f" % plot_params.horiz_scale_x0)
        plot_objects.horiz_scale_x0_textbox.setMinimumWidth(60)
        plot_objects.horiz_scale_x0_textbox.setMaximumWidth(80)
        plot_objects.horiz_scale_y0_textbox = QtGui.QLineEdit()
        self.connect(plot_objects.horiz_scale_y0_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_horiz_scale_y0_textbox_edited)
        plot_objects.horiz_scale_y0_textbox.setText("%0.2f" % plot_params.horiz_scale_y0)
        plot_objects.horiz_scale_y0_textbox.setMinimumWidth(60)
        plot_objects.horiz_scale_y0_textbox.setMaximumWidth(80)

        vert_scale_length_hbox = QtGui.QHBoxLayout()
        vert_scale_pos_hbox = QtGui.QHBoxLayout()
        vert_scale_length_hbox.addWidget(plot_objects.vert_scale_checkbox)
        vert_scale_length_hbox.addStretch(1)
        vert_scale_length_hbox.addWidget(plot_objects.vert_scale_length_label)
        vert_scale_length_hbox.addWidget(plot_objects.vert_scale_length_textbox)
        vert_scale_pos_hbox.addStretch(1)
        vert_scale_pos_hbox.addWidget(plot_objects.vert_scale_origin_label)
        vert_scale_pos_hbox.addWidget(plot_objects.vert_scale_x0_textbox)
        vert_scale_pos_hbox.addWidget(plot_objects.vert_scale_y0_textbox)
        vert_scale_vbox = QtGui.QVBoxLayout()
        vert_scale_vbox.addLayout(vert_scale_length_hbox)
        vert_scale_vbox.addLayout(vert_scale_pos_hbox)

        horiz_scale_length_hbox = QtGui.QHBoxLayout()
        horiz_scale_pos_hbox = QtGui.QHBoxLayout()
        horiz_scale_length_hbox.addWidget(plot_objects.horiz_scale_checkbox)
        horiz_scale_length_hbox.addStretch(1)
        horiz_scale_length_hbox.addWidget(plot_objects.horiz_scale_length_label)
        horiz_scale_length_hbox.addWidget(plot_objects.horiz_scale_length_textbox)
        horiz_scale_pos_hbox.addStretch(1)
        horiz_scale_pos_hbox.addWidget(plot_objects.horiz_scale_origin_label)
        horiz_scale_pos_hbox.addWidget(plot_objects.horiz_scale_x0_textbox)
        horiz_scale_pos_hbox.addWidget(plot_objects.horiz_scale_y0_textbox)
        horiz_scale_vbox = QtGui.QVBoxLayout()
        horiz_scale_vbox.addLayout(horiz_scale_length_hbox)
        horiz_scale_vbox.addLayout(horiz_scale_pos_hbox)

        scale_vbox = QtGui.QVBoxLayout()
        scale_vbox.addLayout(vert_scale_vbox)
        scale_vbox.addLayout(horiz_scale_vbox)
        scale_vbox.addStretch(1)



        scale_widget = QtGui.QWidget()
        scale_widget.setLayout(scale_vbox)

        plot_objects.tabs.addTab(analysis_widget, "Analysis")
        plot_objects.tabs.addTab(elsa_widget, "ELSA")
        plot_objects.tabs.addTab(scale_widget, "Scale Bars")

        return plot_objects

    def data_blit(self):
        # type: () -> None
        """
        TODO
        """
        # NOTE: This purposely does NOT extend BasicRadarWindow.data_blit,
        # because there's an ordering issue.
        '''
        This redraws all the various rcoeff/pick/etc plots, but not the
        radar background.
        # TODO: I haven't tested  whether it would be faster to do it like
        # this or do a per-artist blit when it changes. However, this seems
        # easier/cleaner.
        '''
        t0 = time.time()

        self.plot_objects.canvas.restore_region(self.radar_restore)

        # These need to happen before set_visible because they may detect
        # bad parameters and set the flag accordingly
        self.maybe_update_rcoeff()
        self.maybe_update_simple_rcoeff()
        self.maybe_update_multiple() # this looks at flag to decide whether to recalc

        self.data_set_visible(self.plot_objects, self.plot_params)
        # TODO: If this series starts getting too slow, move the "set_data"
        # logic back to the callbacks that change it. However, there are
        # enough things that change the picks/max values that it's a lot
        # simpler to put all of that right here.

        # All of these set the data and call draw_artist, regardless of whether
        # it's visible or not.
        self.plot_curr_picks()
        self.plot_computed_horizons()

        self.plot_rcoeff()
        self.plot_simple_rcoeff()

        self.plot_objects.radar_ax.draw_artist(self.plot_objects.multiple_line)

        self.plot_scalebars()

        self.plot_objects.canvas.update()

        self.data_restore = self.plot_objects.canvas.copy_from_bbox(
            self.plot_objects.full_ax.bbox)

        t1 = time.time()
        #print("time for data blit:  %0.3f" % t1 - t0)
        self.cursor_blit()
        #print("time for cursor blit:  %0.3f" % time.time() - t1)


    def add_pick_label(self, label):
        # type: (str) -> None
        """
        TODO
        """
        super(ExperimentalRadarWindow, self).add_pick_label(label)

        # Update the RCoeff box options!
        # TODO: Figure out how to support more than just pik1 data!!
        # This will require supporting a bed_lel for each product, so we can't
        # use that as the unique key anymore ...
        if 'srf' in label:
            self.plot_objects.rcoeff_srf_combo.addItem(label)
            self.plot_objects.multiple_srf_combo.addItem(label)

        # UGH. h01 is the floating-ice bed picked to match the multiple.
        elif 'bed' in label or 'h01' in label:
            self.plot_objects.rcoeff_bed_combo.addItem(label)
            self.plot_objects.hydrostatic_bed_combo.addItem(label)
            self.plot_objects.multiple_bed_combo.addItem(label)

    def data_set_invisible(self, plot_objects):
        # type: (PlotObjects) -> None
        '''
        Set ALL overlays invisible.
        '''
        super(ExperimentalRadarWindow, self).data_set_invisible(plot_objects)
        plot_objects.rcoeff_sparkline.set_visible(False)
        plot_objects.simple_rcoeff_sparkline.set_visible(False)
        plot_objects.multiple_line.set_visible(False)
        plot_objects.vert_scale.set_visible(False)
        plot_objects.horiz_scale.set_visible(False)

    def data_set_visible(self, plot_objects, plot_params):
        # type: (PlotObjects, PlotParams) -> None
        '''
        Replot various data overlays based on configuration in plot_params.
        Does NOT turn everything on; only those that are enabled.
        '''
        super(ExperimentalRadarWindow, self).data_set_visible(plot_objects, plot_params)
        plot_objects.rcoeff_sparkline.set_visible(plot_params.rcoeff_visible)
        plot_objects.simple_rcoeff_sparkline.set_visible(plot_params.simple_rcoeff_visible)
        plot_objects.multiple_line.set_visible(plot_params.multiple_visible)
        plot_objects.vert_scale.set_visible(plot_params.vert_scale_visible)
        plot_objects.horiz_scale.set_visible(plot_params.horiz_scale_visible)

    def plot_scalebars(self):
        # type: () -> None
        """
        TODO
        """
        xlim = self.plot_objects.radar_ax.get_xlim()
        dist0 = self.transect_data.rpc.along_track_dist(
            [0, xlim[0]], self.plot_params.xunits)
        dist1 = self.transect_data.rpc.along_track_dist(
            [0, xlim[1]], self.plot_params.xunits)
        data_width = dist1 - dist0 # in meters

        ylim = self.plot_objects.radar_ax.get_ylim()
        data_height = np.abs(ylim[1] - ylim[0]) * 1.69 # in meters

        self.plot_objects.vert_scale.set_length(
            self.plot_params.vert_scale_length, data_height)
        self.plot_objects.vert_scale.set_origin(self.plot_params.vert_scale_x0,
                                                self.plot_params.vert_scale_y0)
        self.plot_objects.vert_scale.update()

        self.plot_objects.horiz_scale.set_length(
            self.plot_params.horiz_scale_length, (data_width/1000.))
        self.plot_objects.horiz_scale.set_origin(
            self.plot_params.horiz_scale_x0, self.plot_params.horiz_scale_y0)
        self.plot_objects.horiz_scale.update()

        for element in self.plot_objects.vert_scale.elements.values():
            self.plot_objects.radar_ax.draw_artist(element)
        for element in self.plot_objects.horiz_scale.elements.values():
            self.plot_objects.radar_ax.draw_artist(element)

    def plot_simple_rcoeff(self):
        # type: () -> None
        """
        TODO
        """
        for element in self.plot_objects.simple_rcoeff_sparkline.elements.values():
            self.plot_objects.radar_ax.draw_artist(element)

    def plot_rcoeff(self):
        # type: () -> None
        '''
        Computing the rcoeff in here doesn't make sense ... data will be set
        when one of the relevant callbacks changes.
        '''
        # QUESTION: What does this actually do??
        for element in self.plot_objects.rcoeff_sparkline.elements.values():
            self.plot_objects.radar_ax.draw_artist(element)

    def maybe_update_simple_rcoeff(self):
        # type: () -> None
        """
        TODO
        """
        if not self.plot_params.simple_rcoeff_visible:
            return
        if not self.plot_params.simple_rcoeff_needs_recalc:
            return
        if self.plot_params.active_horiz is None:
            msg = "simple_rcoeff requires pick file"
            plotUtilities.show_error_message_box(msg)
            self.plot_params.simple_rcoeff_visible = False
            self.plot_objects.simple_rcoeff_checkbox.setChecked(False)
            return

        pick_data = self.transect_data.pick_dict[self.plot_params.active_horiz]
        max_counts = pick_data.max_vals[:, PICK_VAL[self.plot_params.max_type]]
        tt = np.arange(0, self.radar_data.num_traces)

        # Raw values are reported in dBm, with a season-dependent offset.
        gain_offset = radarAnalysis.channel_offsets[pick_data.channel]
        pick_dB = max_counts/1000. + gain_offset
        good_idxs = np.where(np.isfinite(pick_dB))
        pick_dB = pick_dB[good_idxs]
        tt = tt[good_idxs]

        self.plot_objects.simple_rcoeff_sparkline.set_data(tt, pick_dB)
        self.plot_params.simple_rcoeff_needs_recalc = False

    def _on_simple_rcoeff_checkbox_changed(self):
        # type: () -> None
        """
        TODO
        """
        checked = self.plot_objects.simple_rcoeff_checkbox.isChecked()
        self.plot_params.simple_rcoeff_visible = checked
        self.data_blit()

    def maybe_update_rcoeff(self):
        # type: () -> None
        '''
        Only updates if one of the parameters has changed AND the plot
        is supposed to be visible.
        '''
        if self.plot_params.rcoeff_visible:
            if self.plot_params.rcoeff_needs_recalc:
                if self.check_rcoeff_params(disp=True):
                    srf_label = self.plot_params.rcoeff_srf_label
                    srf_picks = self.transect_data.pick_dict[srf_label]
                    srf_file = radutils.pickutils.get_pick_filename(
                        srf_picks.pst, srf_picks.product, srf_picks.channel, srf_picks.label)
                    bed_label = self.plot_params.rcoeff_bed_label
                    bed_picks = self.transect_data.pick_dict[bed_label]
                    bed_file = radutils.pickutils.get_pick_filename(
                        bed_picks.pst, bed_picks.product, bed_picks.channel, bed_picks.label)
                    iceloss = self.plot_params.rcoeff_iceloss
                    trace, db = radarAnalysis.calculate_rcoeff(
                        self.pst, self.plot_params.product, srf_file, bed_file, iceloss)
                    self.plot_params.rcoeff_needs_recalc = False
                    self.plot_objects.rcoeff_sparkline.set_data(trace, db)
                else:
                    self.plot_params.rcoeff_visible = False
                    self.plot_objects.rcoeff_checkbox.setChecked(False)

    def _on_rcoeff_checkbox_changed(self):
        # type: () -> None
        """
        TODO
        """
        checked = self.plot_objects.rcoeff_checkbox.isChecked()
        self.plot_params.rcoeff_visible = checked
        # TODO: This may call setChecked(False), which seems to disable the
        # callback the next time around ... but it works again on the 2nd click.
        self.data_blit()

    def _on_rcoeff_recalc_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        print("rcoeff recalc button clicked!")
        self.plot_params.rcoeff_needs_recalc = True
        self.data_blit()

    def _on_rcoeff_srf_combo_activated(self, label):
        # type: (unicode) -> None
        """
        TODO
        """
        srf_label = str(label)
        if srf_label != self.plot_params.rcoeff_srf_label:
            self.plot_params.rcoeff_srf_label = srf_label
            self.plot_params.rcoeff_needs_recalc = True
            # Can't condition this based on rcoeff_visible, since
            # maybe_update_rcoeff changes that, and we need to redraw to make
            # it disappear.
            self.data_blit()

    def _on_rcoeff_bed_combo_activated(self, label):
        # type: (unicode) -> None
        """
        TODO
        """
        bed_label = str(label)
        if bed_label != self.plot_params.rcoeff_bed_label:
            self.plot_params.rcoeff_bed_label = bed_label
            self.plot_params.rcoeff_needs_recalc = True
            self.data_blit()

    def _on_rcoeff_iceloss_textbox_edited(self, disp=False):
        # type: (bool) -> None
        """
        TODO
        """
        try:
            iceloss = float(self.plot_objects.rcoeff_iceloss_textbox.text())
        except ValueError:
            if disp:
                msg = "Please enter numeric value for iceloss."
                plotUtilities.show_error_message_box(msg)
            return

        if iceloss < 0:
            msg = "Non-physical value for ice loss! Should be positive dB/km."
            plotUtilities.show_error_message_box(msg)
            return

        if iceloss != self.plot_params.rcoeff_iceloss:
            self.plot_params.rcoeff_iceloss = iceloss
            self.plot_params.rcoeff_needs_recalc = True
            self.data_blit()

    def _on_hydrostatic_checkbox_changed(self, val):
        # type: (int) -> None
        """
        TODO
        """
        print("hydrostatic checkbox changed")

    def _on_hydrostatic_bed_combo_activated(self, label):
        # type: (unicode) -> None
        """
        TODO
        """
        print("hydrostatic bed combo activated for: %s" % label)

    def maybe_update_multiple(self):
        # type: () -> None
        """
        TODO
        """
        if not self.plot_params.multiple_visible:
            return
        if not self.plot_params.multiple_needs_recalc:
            return
        if not self.check_multiple_params(disp=True):
            self.plot_params.multiple_visible = False
            self.plot_objects.multiple_checkbox.setChecked(False)
            return

        srf_label = self.plot_params.multiple_srf_label
        srf_picks = self.transect_data.pick_dict[srf_label]
        srf_file = radutils.pickutils.get_pick_filename(
            srf_picks.pst, srf_picks.product, srf_picks.channel,
            srf_picks.label)
        srf_count = self.plot_params.multiple_srf_count
        bed_label = self.plot_params.multiple_bed_label
        if bed_label is not None:
            bed_picks = self.transect_data.pick_dict[bed_label]
            bed_file = radutils.pickutils.get_pick_filename(
                bed_picks.pst, bed_picks.product, bed_picks.channel,
                bed_picks.label)
        else:
            bed_file = ''
        bed_count = self.plot_params.multiple_bed_count

        trace, sample = radarAnalysis.calculate_multiple(
            self.pst, self.plot_params.product, srf_count, srf_file, bed_count, bed_file)
        self.plot_objects.multiple_line.set_data(trace, sample)
        self.plot_params.multiple_needs_recalc = False

    def _on_multiple_checkbox_changed(self):
        # type: () -> None
        """
        TODO
        """
        checked = self.plot_objects.multiple_checkbox.isChecked()
        self.plot_params.multiple_visible = checked
        self.data_blit()

    def _on_multiple_recalc_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        print("multiple recalc button clicked!")
        self.plot_params.multiple_needs_recalc = True
        self.data_blit()

    def _on_multiple_srf_combo_activated(self, label):
        # type: (unicode) -> None
        """
        TODO
        """
        srf_label = str(label)
        if srf_label != self.plot_params.rcoeff_srf_label:
            self.plot_params.multiple_srf_label = srf_label
            self.plot_params.multiple_needs_recalc = True
            self.data_blit()

    def _on_multiple_bed_combo_activated(self, label):
        # type: (unicode) -> None
        """
        TODO
        """
        bed_label = str(label)
        if bed_label != self.plot_params.rcoeff_bed_label:
            self.plot_params.multiple_bed_label = bed_label
            self.plot_params.multiple_needs_recalc = True
            self.data_blit()

    def _on_multiple_bed_count_textbox_edited(self):
        # type: () -> None
        """
        TODO
        """
        txt = self.plot_objects.multiple_bed_count_textbox.text()
        try:
            bed_count = int(txt)
        except ValueError:
            if txt != '':
                msg = "Could not convert %r to float." % (txt)
                plotUtilities.show_error_message_box(msg)
                self.plot_objects.multiple_bed_count_textbox.setText('0')
            return
        if bed_count != self.plot_params.multiple_bed_count:
            self.plot_params.multiple_bed_count = bed_count
            self.plot_params.multiple_needs_recalc = True
            self.data_blit()

    def _on_multiple_srf_count_textbox_edited(self, disp=False):
        # type: (bool) -> None
        """
        TODO
        """
        txt = self.plot_objects.multiple_srf_count_textbox.text()
        try:
            srf_count = int(txt)
        except ValueError:
            if txt != '':
                msg = "Please enter numeric value for plane-srf multiple count."
                plotUtilities.show_error_message_box(msg)
                self.plot_objects.multiple_srf_count_textbox.setText('0')
            return

        if srf_count < 0:
            if disp:
                msg = "plane-srf multiple count must be >= 0!"
                plotUtilities.show_error_message_box(msg)
                return

        if srf_count != self.plot_params.multiple_srf_count:
            self.plot_params.multiple_srf_count = srf_count
            self.plot_params.multiple_needs_recalc = True
            self.data_blit()

    def _on_vert_scale_checkbox_changed(self):
        # type: () -> None
        """
        TODO
        """
        checked = self.plot_objects.vert_scale_checkbox.isChecked()
        self.plot_params.vert_scale_visible = checked
        # TODO: This may call setChecked(False), which seems to disable the
        # callback the next time around ... but it works again on the 2nd click.
        self.data_blit()

    def _on_vert_scale_length_textbox_edited(self):
        # type: () -> None
        """
        Update plot_objects with new length and redraw, after sanity checking.
        """
        curr_length_str = '%r' % self.plot_params.vert_scale_length
        try:
            length = float(self.plot_objects.vert_scale_length_textbox.text())
        except ValueError:
            msg = "Please enter numerical value for length"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.vert_scale_length_textbox.setText(curr_length_str)
            return

        if length <= 0:
            msg = "Please enter positive length"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.vert_scale_length_textbox.setText(curr_length_str)
            return

        if length != self.plot_params.vert_scale_length:
            self.plot_params.vert_scale_length = length
            self.data_blit()

    def _on_vert_scale_x0_textbox_edited(self):
        # type: () -> None
        """
        Update plot_objects with new x0 and redraw
        """
        curr_x0_str = '%r' % self.plot_params.vert_scale_x0
        try:
            x0 = float(self.plot_objects.vert_scale_x0_textbox.text())
        except ValueError:
            msg = "Please enter numerical value for x0"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.vert_scale_x0_textbox.setText(curr_x0_str)
            return

        if x0 < 0.0 or x0 > 1.0:
            msg = "Please enter x0 in range [0, 1]"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.vert_scale_x0_textbox.setText(curr_x0_str)
            return

        if x0 != self.plot_params.vert_scale_x0:
            self.plot_params.vert_scale_x0 = x0
            self.data_blit()

    def _on_vert_scale_y0_textbox_edited(self):
        # type: () -> None
        """
        Update plot_objects with new y0 and redraw
        """
        curr_y0_str = '%r' % self.plot_params.vert_scale_y0
        try:
            y0 = float(self.plot_objects.vert_scale_y0_textbox.text())
        except ValueError:
            msg = "Please enter numerical value for y0"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.vert_scale_y0_textbox.setText(curr_y0_str)
            return

        if y0 < 0.0 or y0 > 1.0:
            msg = "Please enter y0 in range [0, 1]"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.vert_scale_y0_textbox.setText(curr_y0_str)
            return

        if y0 != self.plot_params.vert_scale_y0:
            self.plot_params.vert_scale_y0 = y0
            self.data_blit()

    def _on_horiz_scale_checkbox_changed(self):
        # type: () -> None
        """
        TODO
        """
        checked = self.plot_objects.horiz_scale_checkbox.isChecked()
        self.plot_params.horiz_scale_visible = checked
        # TODO: This may call setChecked(False), which seems to disable the
        # callback the next time around ... but it works again on the 2nd click.
        self.data_blit()

    def _on_horiz_scale_length_textbox_edited(self):
        # type: () -> None
        """
        Update plot_objects with new length and redraw, after sanity checking.
        """
        curr_length_str = '%r' % self.plot_params.horiz_scale_length
        try:
            length = float(self.plot_objects.horiz_scale_length_textbox.text())
        except ValueError:
            msg = "Please enter numerical value for length"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.horiz_scale_length_textbox.setText(curr_length_str)
            return

        if length <= 0:
            msg = "Please enter positive length"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.horiz_scale_length_textbox.setText(curr_length_str)
            return

        if length != self.plot_params.horiz_scale_length:
            self.plot_params.horiz_scale_length = length
            self.data_blit()

    # This is crying out for a lambda taking the textbox object and the var it
    # goes into ...
    def _on_horiz_scale_x0_textbox_edited(self):
        # type: () -> None
        """
        Update plot_objects with new x0 and redraw
        """
        curr_x0_str = '%r' % self.plot_params.horiz_scale_x0
        try:
            x0 = float(self.plot_objects.horiz_scale_x0_textbox.text())
        except ValueError:
            msg = "Please enter numerical value for x0"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.horiz_scale_x0_textbox.setText(curr_x0_str)
            return

        if x0 < 0.0 or x0 > 1.0:
            msg = "Please enter x0 in range [0, 1]"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.horiz_scale_x0_textbox.setText(curr_x0_str)
            return

        if x0 != self.plot_params.horiz_scale_x0:
            self.plot_params.horiz_scale_x0 = x0
            self.data_blit()

    def _on_horiz_scale_y0_textbox_edited(self):
        # type: () -> None
        """
        Update plot_objects with new y0 and redraw
        """
        curr_y0_str = '%r' % self.plot_params.horiz_scale_y0
        try:
            y0 = float(self.plot_objects.horiz_scale_y0_textbox.text())
        except ValueError:
            msg = "Please enter numerical value for y0"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.horiz_scale_y0_textbox.setText(curr_y0_str)
            return

        if y0 < 0.0 or y0 > 1.0:
            msg = "Please enter y0 in range [0, 1]"
            plotUtilities.show_error_message_box(msg)
            self.plot_objects.horiz_scale_y0_textbox.setText(curr_y0_str)
            return

        if y0 != self.plot_params.horiz_scale_y0:
            self.plot_params.horiz_scale_y0 = y0
            self.data_blit()

    def _on_new_elsa_button_clicked(self):
        # type: () -> None
        """
        TODO
        """
        # NB - used to be called on_new_pcor_button_clicked
        print("new elsa button clicked")

    def _elsa_remove_row_cb(self, datatype):
        # type: (str) -> None
        """
        TODO
        """
        # NB - used to be called pcor_remove_row_cb
        print("removing elsa row: %s" % datatype)

    def _elsa_color_cb(self, datatype, color):
        # type: (str, QtGui.QColor) -> None
        """
        TODO
        """
        # NB - used to be called pcor_color_cb
        print("elsa color cb called: %s, %r" % datatype, color)

    def _elsa_params_cb(self, datatype, params):
        # type: (str, Tuple[float, float, str]) -> None
        """
        TODO
        """
        # NB - used to be called pcor_params_cb
        print("elsa params cb called %s" % datatype)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    parser = argparse.ArgumentParser(
        description='RadarFigure - interface for viewing and picking UTIG radar data')
    parser.add_argument('pst', help='PST to display')
    parser.add_argument('--filename', help='Path to file')
    parser.add_argument('--experimental', action='store_true')
    args = parser.parse_args()


    if args.experimental:
        radar_window = ExperimentalRadarWindow(args.pst, args.filename)
    else:
        radar_window = BasicRadarWindow(args.pst, args.filename)

    radar_window.show()
    app.exec_()
