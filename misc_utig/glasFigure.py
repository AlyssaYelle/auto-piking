import matplotlib.pyplot as plt
import matplotlib.widgets as mpw
import numpy as np
import os
import sys
try:
    import typing
    from typing import Any, Callable, Optional, Tuple
    if typing.TYPE_CHECKING:
        import PyQt4.QtGui as QtGui
except:
    pass

from mplUtilities import XevasHorizSelector, XevasVertSelector

WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set')

# Not sure why this isn't added on melt...
sys.path.append(WAIS+'/syst/linux/py/zutils')
import zfile


def load_orbit(glas_line):
    # type: (int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]
    '''
    Loads all tracks from the given orbit, returns as a dict of
    {'yyyy.mm.dd':xyzgreo}
    '''
    basedir = WAIS + '/targ/tpro/GLAS/' + str(glas_line)
    dates = os.listdir(basedir)
    tracks = {} # type: Dict[str, Tuple[np.ndarray, np.ndarray]]
    for date in dates:
        ztim, data = zfile.read_zfile(basedir + '/' + date +
                                      '/f_las_srfelv/ztim_xyzgreo.bin')
        posix = zfile.ztim2posix(ztim)
        tracks[date] = (posix, data)
    return tracks

class GLASFigure:
    def __init__(self, glas_line, glas_zoom_cb=None):
        # type: (int, Optional[Callable[(int, Tuple[float, float])]]) -> None
        self.glas_zoom_cb = glas_zoom_cb
        self.glas_line = glas_line

        self.fig = plt.figure()

        # Variables controlling they layout of the various axes
        xx = 0.1 # Bottom-left corner of bottom axis
        yy = 0.2
        width = 0.8
        height = 0.25
        margin = 0.01
        zoom_width = 0.03

        self.ax1 = self.fig.add_axes([xx, yy, width, height])
        self.ax1.tick_params(axis='x', which='both', bottom='off', top='off',
                             labelbottom='off')
        self.ax1.yaxis.tick_right()
        self.ax2 = self.fig.add_axes([xx, yy+height+margin, width, height])
        self.ax2.tick_params(axis='x', which='both', bottom='off', top='off',
                             labelbottom='off')
        self.ax2.yaxis.tick_right()

        # Indices into the data found in ztim_xyzgreo.bin
        XX = 0; # PS71 x-coordinate
        YY = 1; # PS71 y-coordinate
        ZZ = 2; # WGS84 elevation
        GG = 3; # gain (i_gval_rcv ... I don't use this for anything)
        RR = 4; # crosstrack distance from reference track
        EE = 5; # solid earth tide - correction is included in elev
        OO = 6; # ocean tide - has been removed from elev. (by default,
                # the data includes this tide correction, which we don't trust.)

        # TODO: Figure out how to get consistent colors + legend w/ dates
        tracks = load_orbit(self.glas_line)
        for date in tracks:
            ztim, data = tracks[date]
            color = np.random.rand(3)
            self.ax1.plot(data[:,XX], data[:,ZZ], '.', color=color)
            self.ax2.plot(data[:,XX], data[:,ZZ]+data[:,OO], '.', color=color)

        nominal_track = np.genfromtxt(WAIS + '/targ/xtra/ALL/deva/glas/' + str(glas_line) + '.xy')

        # There's sometimes data that's not over Antarctica that shows up in
        # the GLAS lines ... this filters that out.
        xmin = np.min(nominal_track[:,0])
        xmax = np.max(nominal_track[:,0])

        self.xevas_horiz_ax = self.fig.add_axes([xx, yy-zoom_width-margin, width, zoom_width])
        self.horiz_xevas = XevasHorizSelector(self.xevas_horiz_ax,
                                              xmin, xmax, self.update_x_cb)
        ymin1, ymax1 = self.ax1.get_ylim()
        self.xevas_vert1_ax = self.fig.add_axes([xx-zoom_width-margin, yy, zoom_width, height])
        self.vert1_xevas = XevasVertSelector(self.xevas_vert1_ax,
                                             ymin1, ymax1, self.update_y_cb)
        ymin2, ymax2 = self.ax2.get_ylim()
        self.xevas_vert2_ax = self.fig.add_axes([xx-zoom_width-margin, yy+height+margin, zoom_width, height])
        self.vert2_xevas = XevasVertSelector(self.xevas_vert2_ax,
                                             ymin2, ymax2, self.update_y_cb)

        self.ax1.set_xlim([xmin,xmax])
        self.ax2.set_xlim([xmin,xmax])

        # Add a title!
        glas_text_ax = self.fig.add_axes([0.3, 3*margin, 0.1, 0.05])
        glas_text_ax.axis('off')
        glas_text = glas_text_ax.text(0, 0, 'curr GLAS line: %r' % self.glas_line)

        # Right-click to drag the zoom selection around.
        # NB - minspanx is in IMAGE coords, not pixel coords.
        self.pan_rs1 = mpw.RectangleSelector(self.ax1, self.pan_cb,
                                            drawtype='line', button=[3],
                                            minspanx=5, minspany=5)
        self.pan_rs1.set_active(True)
        self.pan_rs2 = mpw.RectangleSelector(self.ax2, self.pan_cb,
                                            drawtype='line', button=[3],
                                            minspanx=5, minspany=5)
        self.pan_rs2.set_active(True)

        # Left-click to zoom in farther on the zoom axis.
        self.zoom_rs1 = mpw.RectangleSelector(self.ax1, self.zoom_in_cb,
                                             drawtype='box', button=[1],
                                             minspanx=5, minspany=5)
        self.zoom_rs1.set_active(True)
        self.zoom_rs2 = mpw.RectangleSelector(self.ax2, self.zoom_in_cb,
                                             drawtype='box', button=[1],
                                             minspanx=5, minspany=5)
        self.zoom_rs2.set_active(True)

        # QUESTION: Which of these should I use here?
        # (I want to be able to test these figures stand-alone, in addition
        # to as part of the larger/integrated GUI)
        plt.show(block=False)
        plt.draw()

    def pan_cb(self, eclick, erelease):
        # type: (QtGui.QMouseEvent, QtGui.QMouseEvent) -> None
        '''
        Right-click on the most zoomed-in map drags it (not real-time).
        '''
        print "called pan_cb!"
        dx = eclick.xdata - erelease.xdata
        dy = eclick.ydata - erelease.ydata
        xlim = self.ax1.get_xlim() # type: Tuple[float, float]
        ylim = self.ax1.get_ylim() # type: Tuple[float, float]
        new_xlim = (xlim[0]+dx, xlim[1]+dx)
        new_ylim = (ylim[0]+dy, ylim[1]+dy)
        self.update_axes(new_xlim, new_ylim)
        plt.draw()

    ######
    # left-click-and-drag selects region of context map
    # TODO: This is very unsatisfyingly slow on OSX, does fine in Ubuntu12.04 VM.
    def zoom_in_cb(self, eclick, erelease):
        # type: (QtGui.QMouseEvent, QtGui.QMouseEvent) -> None
        '''
        Left-click on the context map selects new ROI.
        '''
        print "called zoom_in_cb!"
        # TODO: I think this is my problem for flipping axes! I probably need min/max here.
        minx = min([eclick.xdata, erelease.xdata])
        maxx = max([eclick.xdata, erelease.xdata])
        xlim = (minx, maxx)
        miny = min([eclick.ydata, erelease.ydata])
        maxy = max([eclick.ydata, erelease.ydata])
        ylim = (miny, maxy)
        self.update_axes(xlim, ylim)
        plt.draw()

    def update_axes(self, xlim, ylim):
        # type: (Tuple[float, float], Tuple[float, float]) -> None
        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)
        self.ax2.set_xlim(xlim)
        self.ax2.set_ylim(ylim)
        if self.glas_zoom_cb is not None:
            self.glas_zoom_cb(self.glas_line, xlim)
        self.horiz_xevas.update_range(xlim)
        self.vert1_xevas.update_range(ylim)
        self.vert2_xevas.update_range(ylim)
        plt.draw()

    def update_x_cb(self, xmin, xmax):
        # type: (float, float) -> None
        xlim = [xmin, xmax]
        self.ax1.set_xlim(xlim)
        self.ax2.set_xlim(xlim)
        if self.glas_zoom_cb is not None:
            self.glas_zoom_cb(self.glas_line, xlim)
        plt.draw()

    def update_y_cb(self, ymin, ymax):
        # type: (float, float) -> None
        ylim = [ymin, ymax]
        self.ax1.set_ylim(ylim)
        self.ax2.set_ylim(ylim)
        plt.draw()

    def close(self):
        # type: () -> None
        plt.close(self.fig)
