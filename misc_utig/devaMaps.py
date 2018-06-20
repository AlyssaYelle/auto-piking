import numpy as np
import os
import sys

import devaParameters as dp
import devaUtilities
import plotUtilities

WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set')
sys.path.append('%s/syst/linux/py' % (WAIS))
import plotutils
import plotutils.basemap
import plotutils.lakes
import plotutils.scalebar

# TODO: I don't like having this dependency on season ...
import waisutils
import waisutils.season

try:
    import typing
    from typing import Any, Dict, List, Optional, Set, Tuple
    if typing.TYPE_CHECKING:
        import matplotlib
        from matplotlib.figure import Figure
        import PyQt4.QtGui as QtGui
except:
    pass

# I'm trying a similar hierarchy of redrawing functions as radarFigure
# 1) full_redraw() - When the background changes, or zoom/pan action
# 2) data_blit() - When the transects / glas lines / lakes / etc. are changed.
# 3) cursor_blit() - When the cursor showing radargram position is changed.


class SelectionObjects(object):
    '''
    All the handles for various matplotlib objects used in the selection figure.

    '''
    def __init__(self, fig):
        # type: (Figure) -> None
        self.fig = fig

        self.ax = self.fig.add_axes([0,0,1,1], axisbg='silver')
        # modify display of cursor x/y locations at bottom right
        self.ax.format_coord = lambda x,y: 'x=%dkm y=%dkm' % (int(x/1000), int(y/1000))
        self.ax.tick_params(bottom='off', top='off', left='off', right='off',
                            labelbottom='off', labelleft='off')
        self.ax.axis('off')

        self.scalebar = plotutils.scalebar.Scalebar(self.ax, 0.05, 0.07, 0.1, 0.01,
                                                    barstyle='fancy', coords='frac',
                                                    unit_label='km', unit_factor=1000)

        self.cax = self.fig.add_axes([0.05, 0.92, 0.1, 0.01])
        self.cax.set_visible(False)

        # Shows which part of the GLAS line is displayed by the glas figure
        self.glas_region, = self.ax.plot(0,0, color=dp.highlighted_color,
                                         linewidth=dp.highlighted_linewidth,
                                         zorder=dp.highlighted_zorder)

        # Shows where on the flight the currently-displayed frame is
        self.camera_cursor, = self.ax.plot(0,0, color='black',
                                           marker='*', markersize=20,
                                           zorder=dp.highlighted_zorder)

        # TODO: The lakes should be moved to SelectionData, and only loaded when checked.
        lakes = plotutils.lakes.load_smith_lakes()
        self.smith_lakes = self.ax.add_collection(lakes)
        self.smith_lakes.set_zorder(dp.lake_zorder)

        self.radar_pst_handles = {} # type: Dict[str, Dict[str, matplotlib.lines.Line2D]]

        # TODO: This feels like something that should be in the data, but
        # since the background_dict requires pointers to fig/ax, it's here.
        self.backgrounds = plotutils.basemap.make_background_dict(self.fig,
                                                                  self.ax,
                                                                  self.cax)
        self.grounding_lines = devaUtilities.make_grounding_line_dict()
        # This bit should become obsolete, as it'll be set programmatically
        # when we make sure the dropdown menu makes sense ...
        self.background = self.backgrounds['modis_simple']
        self.background.set_background()
        self.ax.axis('equal')

        self.pst_lines = {} # type: Dict[str, Optional[matplotlib.lines.Line2D]]
        self.glas_lines = {} # type: Dict[int, Optional[matplotlib.lines.Line2D]]
        self.flight_lines = {} # type: Dict[str, np.ndarray]
        self.pcl_lines = {} # type: Dict[str, np.ndarray]

    def remove_radar_figure(self, pst):
        # type: (str) -> None
        if pst not in self.radar_pst_handles:
            msg = "Attempted to close pst %s, which is not in list!" % (pst)
            raise Exception(msg)
        for elem in self.radar_pst_handles[pst]:
            self.radar_pst_handles[pst][elem].remove()
        del self.radar_pst_handles[pst]


    def add_radar_figure(self, pst, color, data):
        # type: (str, QtGui.QColor, np.ndarray) -> None
        if pst in self.radar_pst_handles:
            msg = "There is already a window open for %s" % (pst)
            plotUtilities.show_error_message_box(msg)
            return

        pstx = data[:,1]
        psty = data[:,2]

        self.radar_pst_handles[pst] = {}
        self.radar_pst_handles[pst]['line'], = self.ax.plot(
            pstx, psty, color=color, linewidth=dp.shown_linewidth,
            zorder=dp.shown_zorder)
        # Highlights part of the radar line corresponding to radarFIgure
        self.radar_pst_handles[pst]['region'], = self.ax.plot(
            pstx, psty, color=dp.highlighted_color,
            linewidth=dp.highlighted_linewidth,
            zorder=dp.highlighted_zorder)
        # Shows which side of the PST region is the start.
        self.radar_pst_handles[pst]['start'], = self.ax.plot(
            pstx[0], psty[0], marker='.', color=dp.highlighted_color,
            markeredgewidth=dp.highlighted_linewidth+2,
            zorder=dp.highlighted_zorder)
        # Shows where on the PST we are
        self.radar_pst_handles[pst]['cursor'], = self.ax.plot(
            pstx[0], psty[0], marker='x', color=dp.highlighted_color,
            markeredgewidth=dp.highlighted_linewidth+1,
            zorder=dp.highlighted_zorder)

    def set_pst_color(self, pst, color):
        # type: (str, QtGui.QColor) -> None
        if pst not in self.radar_pst_handles:
            msg = "Tried to change color of pst (%s)  not in list!" % (pst)
            raise Exception(msg)
        self.radar_pst_handles[pst]['line'].set_color(color)

    def change_glas_visibility(self, visible):
        # type: (bool) -> None
        for line in self.glas_lines.keys():
            self.glas_lines[line].set_visible(visible)
        self.glas_region.set_visible(visible)

    def change_flight_visibility(self, visible):
        # type: (bool) -> None
        for line in self.flight_lines.keys():
            self.flight_lines[line].set_visible(visible)
        for line in self.pcl_lines.keys():
            self.pcl_lines[line].set_visible(visible)
        self.camera_cursor.set_visible(visible)

    def unselect_pst(self, pst):
        # type: (str) -> None
        if waisutils.season.pst_is_utig(pst):
            lc = dp.utig_line_color
        elif waisutils.season.pst_is_cresis(pst):
            lc = dp.cresis_line_color
        else:
            lc = dp.external_line_color
        self.pst_lines[pst].set_color(lc)
        self.pst_lines[pst].set_zorder(dp.line_zorder)

    def select_pst(self, pst, data):
        # type: (str, np.ndarray) -> None
        # selected PST that's not currently visible.
        # That's OK, we'll create and display it.
        if self.pst_lines[pst] is None:
            self.pst_lines[pst], = self.ax.plot(
                data[pst][:,1], data[:,2],
                color = dp.selected_color, linewidth=dp.pst_linewidth,
                picker=dp.line_pick_radius, zorder=dp.selected_zorder)
        else:
            self.pst_lines[pst].set_color(dp.selected_color)
            self.pst_lines[pst].set_zorder(dp.selected_zorder)

    def deselect_glas(self, glas):
        # type: (int) -> None
        self.glas_lines[glas].set_color(dp.glas_line_color)
        self.glas_lines[glas].set_linestyle(dp.glas_linestyle)
        self.glas_lines[glas].set_zorder(dp.line_zorder)

    def select_glas(self, glas):
        self.glas_lines[glas].set_color(dp.selected_color)
        self.glas_lines[glas].set_linestyle('solid')
        self.glas_lines[glas].set_zorder(dp.selected_zorder)

    def deselect_flight(self, flight):
        self.flight_lines[flight].set_color(dp.flight_line_color)
        self.flight_lines[flight].set_linestyle(dp.flight_linestyle)
        self.flight_lines[flight].set_zorder(dp.line_zorder)

    def select_flight(self, flight):
        self.flight_lines[flight].set_color(dp.selected_color)
        self.flight_lines[flight].set_linestyle('solid')
        self.flight_lines[flight].set_zorder(dp.selected_zorder)

    def set_pst_visible(self, pst, visible, data):
        # type: (str, bool, np.ndarray) -> None
        if visible:
            if self.pst_lines[pst] is not None:
                self.pst_lines[pst].set_visible(True)
            else:
                if waisutils.season.pst_is_utig(pst):
                    lc = dp.utig_line_color
                elif waisutils.season.pst_is_cresis(pst):
                    lc = dp.cresis_line_color
                else:
                    lc = dp.external_line_color
                    self.pst_lines[pst], = self.ax.plot(
                        data[:,1], data[:,2], color=lc,
                        linewidth=dp.pst_linewidth,
                        picker=dp.line_pick_radius, zorder=dp.line_zorder)
        else: # we don't want it to be visible
            if self.pst_lines[pst] is not None:
                self.pst_lines[pst].set_visible(False)

    def set_flight_visible(self, flight, visible):
        # type: (str, bool) -> None
        '''
        # TODO: rather than just setting visibility here, maybe plot it for first time?
        '''
        for key, line in self.flight_lines.iteritems():
            if flight in key:
                line.set_visible(visible)
        for key, line in self.pcl_lines.iteritems():
            if flight in key:
                line.set_visible(visible)



class SelectionData(object):
    def __init__(self):
        # type: () -> None
        # TODO: Only load these when needed? I suspect that this is the slow bit of starting deva
        self.transects = devaUtilities.load_transects(antarctic=True)
        #self.transects = {}
        self.glas = devaUtilities.load_glas_lines()
        self.flights = devaUtilities.load_flight_lines() # type: Dict[str, np.ndarray]
        self.pcl_data = devaUtilities.load_pcl_quality()






class SelectionConfig(object):
    '''
    Any variables required for replotting the selection figure from scratch.
    '''
    def __init__(self, active_projects, transects):
        # type () -> None
        self.active_projects = active_projects

        # Currently-selected PST and GLAS line.
        self.selected_pst = None # type: Optional[str]
        self.selected_glas = None # type: Optional[int]
        self.selected_flight = None # type: Optional[str]

        projects = set([pst.split('/')[0] for pst in transects])
        if active_projects is None:
            self.visible_projects = {proj:True for proj in projects}
        else:
            self.visible_projects = {proj:False for proj in projects}
            for proj in active_projects:
                self.visible_projects[proj] = True



class SelectionFigure():
    '''
    SelectionFigure is the figure where all user interaction occurs.

    Functionality includes:
    * Panning/zooming updates the associated context figures
    * Allows graphical selection of PST or GLAS line of interest
    * Shows which portion of the selected line is currently displayed
      by the RadarFigure or GLASFigure or CameraFigure.
    * Displays user-selected basemaps
    * Displayes user-selected grounding lines
    '''
    def __init__(self,
                 fig, # type: Figure
                 active_projects, # type: Optional[List[str]]
                 ax_changed_cb, # type: Optional[Callable[Tuple[float, float], Tuple[Float, Float]]]
                 selection_cb # type: Optional[Callable[List[str], List[str], List[str]]]
                ):
        # type: (...) -> None
        '''
        Inputs:
        * fig - already-created matplotlib figure to be used for plotting.
        * active_projects - only load transects in this list. If None, all
              known transects will be loaded.
        * ax_changed_cb(xlim, ylim) - function to call when the user changes
              pan/zoom; main window is in charge of keeping the context figures
              up-to-date with respect to the selection figure.
        * selection_cb(picked_psts, picked_glas_lines, picked_flights)
              function called when the user has clicked on the figure in
              selection mode.
              The main window is in charge of determining which of the onpick'd
              objects is actually desired.
        '''
        self.selection_objects = SelectionObjects(fig)
        self.selection_data = SelectionData()
        self.selection_config = SelectionConfig(active_projects, self.selection_data.transects.keys())

        self.ax_changed_cb = ax_changed_cb
        self.selection_cb = selection_cb

        # This is probably fairly slow ... it is plotting ALL objects that are visible
        self.plot_psts(self.selection_objects, self.selection_data, self.selection_config)
        self.plot_glas(self.selection_objects, self.selection_data, self.selection_config)
        self.plot_flights(self.selection_objects, self.selection_data, self.selection_config)

        # This is a failed attempt to limit zooming out to the original view.
        # It caused problems both with a bad aspect ratio at the start, and
        # with not actually zooming in when requested.
        # (actual code for failed attemps is in on_lim_changed)
        # self.fig.canvas.draw()
        # self.orig_xlim = self.ax.get_xlim()
        # self.orig_ylim = self.ax.get_ylim()

        # dict of dicts - 1st level is pst, 2nd level is all plot elements for
        # displaying that radargram. Currently includes:
        # 'line' - highlighted in a different color than normal selection
        # 'region' - the red "this is what's showing" line
        # 'start' - the red box drawn on deva's window
        # NB - I'm not sure that this is the correct type ... may be used to
        # hold more than just Line2D?

        # TODO: Separate visibility for PCL data, since it is cluttered.
        # TODO: Would be nice to turn on/off flight colors
        # load and plot (non-selectable) PCL data
        # TODO: Figure out what corrupt data is causing limits to get all borked.
        xlim = self.selection_objects.ax.get_xlim()
        ylim = self.selection_objects.ax.get_ylim()
        self.plot_pcl(self.selection_objects, self.selection_data, self.selection_config)
        self.selection_objects.ax.set_xlim(xlim)
        self.selection_objects.ax.set_ylim(ylim)

        # YAY! The onpick events fire before the press and release event, so
        # I can use the onpick to update the list of selected PSTs, and the
        # release to trigger the selection dialog.
        # Press/release will _always_ be called, while onpick will only
        # be called if we're not in pan/zoom mode.
        self.selection_objects.fig.canvas.mpl_connect('button_release_event',
                                    self._on_button_release)
        self.selection_objects.fig.canvas.mpl_connect('pick_event', self._on_pick)

        self.full_redraw()
        self.xlim = self.selection_objects.ax.get_xlim()
        self.ylim = self.selection_objects.ax.get_ylim()

        # Can't connect these until all data has been loaded,
        # b/c it makes it super-slow. Also will die if before we've initialized
        # self.{xlim, ylim}
        # I wish I could just connect to a "lim_changed" event ... but
        # the axes change independently?!?
        self.selection_objects.ax.callbacks.connect('xlim_changed', self._on_lim_changed)
        self.selection_objects.ax.callbacks.connect('ylim_changed', self._on_lim_changed)

        # Sets that are updated by the onpick callbacks.
        self.picked_psts = set([]) # type: Set[str]
        self.picked_glas = set([]) # type: Set[int]
        self.picked_flights = set([]) # type: Set[str]

    def cursor_set_invisible(self, plot_objects):
        pass
    def cursor_set_visible(self, plot_objects):
        pass
    def data_set_invisible(self, plot_objects):
        pass
    def data_set_visible(self, plot_objects):
        pass
    def full_redraw(self):
        self.selection_objects.fig.canvas.draw()
    def data_blit(self):
        pass
    def cursor_blit(self):
        pass

    # TODO: Maybe these should be methods in SelectionObjects?
    def plot_pcl(self, objects, data, config):
        for card, data in data.pcl_data.iteritems():
            if 'tof1' in card:
                continue
            if 'tof0' in card:
                linewidth = dp.card0_linewidth
                zorder = dp.card0_zorder
            else:
                linewidth = dp.card1_linewidth
                zorder = dp.card1_zorder

            great_idxs, = np.where(data[:,4] > 40)
            ok_idxs, = np.where((10 < data[:,4]) & (data[:,4] <= 40))
            bad_idxs, = np.where((0 < data[:,4]) & (data[:,4] <= 10))
            none_idxs, = np.where(0 == data[:,4])
            if great_idxs.size > 0:
                key = '%s_great' % (card)
                xx = data[great_idxs,1]
                yy = data[great_idxs,2]
                objects.pcl_lines[key], = objects.ax.plot(xx, yy, '.', color='blue',
                                                    linewidth=linewidth,
                                                    zorder=zorder)
                objects.pcl_lines[key].set_visible(False)
            if ok_idxs.size > 0:
                key = '%s_ok' % (card)
                xx = data[ok_idxs,1]
                yy = data[ok_idxs,2]
                objects.pcl_lines[key], = objects.ax.plot(xx, yy, '.', color='green',
                                                    linewidth=linewidth,
                                                    zorder=zorder)
                objects.pcl_lines[key].set_visible(False)
            if bad_idxs.size > 0:
                key = '%s_bad' % (card)
                xx = data[bad_idxs,1]
                yy = data[bad_idxs,2]
                objects.pcl_lines[key], = objects.ax.plot(xx, yy, '.', color='orange',
                                                    linewidth=linewidth,
                                                    zorder=zorder)
                objects.pcl_lines[key].set_visible(False)
            if none_idxs.size > 0:
                key = '%s_none' % (card)
                xx = data[none_idxs,1]
                yy = data[none_idxs,2]
                objects.pcl_lines[key], = objects.ax.plot(xx, yy, '.', color='red',
                                                    linewidth=linewidth,
                                                    zorder=zorder)
                objects.pcl_lines[key].set_visible(False)


    def plot_flights(self, objects, data, config):
        for fline, fdata in data.flights.iteritems():
            objects.flight_lines[fline], = objects.ax.plot(fdata[:,1], fdata[:,2],
                                                           color=dp.flight_line_color,
                                                           linewidth=dp.flight_linewidth,
                                                           linestyle=dp.flight_linestyle,
                                                           picker=dp.line_pick_radius,
                                                           zorder=dp.line_zorder)
            # TDOO: This should be inferred from config
            objects.flight_lines[fline].set_visible(False)

    def plot_glas(self, objects, data, config):

        for gline, gdata in data.glas.iteritems():
            objects.glas_lines[gline], = objects.ax.plot(gdata[:,0], gdata[:,1],
                                                         color=dp.glas_line_color,
                                                         linewidth=dp.glas_linewidth,
                                                         linestyle=dp.glas_linestyle,
                                                         picker=dp.line_pick_radius,
                                                         zorder=dp.line_zorder)
            # TODO: This should be set based on config's glas visibility
            objects.glas_lines[gline].set_visible(False)

    def plot_psts(self, objects, data, config):
        for pst, data in data.transects.iteritems():
            proj, _, _ = pst.split('/')
            if config.active_projects is not None and not config.visible_projects[proj]:
                objects.pst_lines[pst] = None
            else:
                if waisutils.season.pst_is_utig(pst):
                    lc = dp.utig_line_color
                elif waisutils.season.pst_is_cresis(pst):
                    lc = dp.cresis_line_color
                else:
                    lc = dp.external_line_color
                objects.pst_lines[pst], = objects.ax.plot(data[:,1], data[:,2],
                                                          color=lc,
                                                          linewidth=dp.pst_linewidth,
                                                          picker=dp.line_pick_radius,
                                                          zorder=dp.line_zorder)



    def add_radar_figure(self, pst, color):
        # type: (str, QtGui.QColor) -> None
        self.selection_objects.add_radar_figure(pst, color, self.selection_data.transects[pst])
        self.full_redraw()

    def set_project_visible(self, proj, visible):
        # type: (str, bool) -> None
        '''
        Called by main deva interface when user has changed which projects are
        visible.
        Creates the plot if necessary, otherwise changes visibility.
        TODO: Do I want to actually remove the invisible lines to improve
        visualizer performance?
        '''
        for pst, data in self.selection_data.transects.iteritems():
            pst_proj, _, _ = pst.split('/')
            if pst_proj == proj:
                self.selection_objects.set_pst_visible(pst, visible, data)

    def set_flight_visible(self, flight, visible):
        # type: (str, bool) -> None
        '''
        Called by main deva interface when user has changed which flights are
        visible.
        TODO: Do I want to actually remove the invisible lines to improve
        visualizer performance?
        '''
        self.selection_objects.set_flight_visible(flight, visible)

    def redraw(self):
        # type: () -> None
        '''
        This way the main deva window can make multiple changes before calling
        redraw.
        '''
        self.full_redraw()

    def set_pst_color(self, pst, color):
        # type: (str, QtGui.QColor) -> None
        self.selection_objects.set_pst_color(pst, color)
        self.full_redraw()

    def on_radar_window_quit(self, pst):
        # type: (str) -> None
        self.selection_objects.remove_radar_figure(pst)
        self.full_redraw()

    def get_transect_list(self):
        # type: () -> List[str]
        return self.selection_data.transects.keys()

    def get_glas_list(self):
        # type: () -> List[int]
        return self.selection_data.glas.keys()

    def get_flight_list(self):
        # type: () -> List[str]
        return self.selection_data.flights.keys()

    def change_glas_visibility(self, visible):
        # type: (bool) -> None
        self.selection_objects.change_glas_visibility(visible)

    def change_flight_visibility(self, visible):
        # type: (bool) -> None
        self.selection_objects.change_flight_visibility(visible)

    # TODO: It might be cleaner to include the currently-displayed colors
    # here as well ... rather than redrawing the PST.
    def set_selected_pst(self, pst):
        # type: (str) -> None
        if self.selection_config.selected_pst == pst:
            return
        if self.selection_config.selected_pst is not None:
            self.selection_objects.unselect_pst(self.selection_config.selected_pst)

        self.selection_config.selected_pst = pst
        if pst is not None:
            if pst not in self.selection_data.transects:
                msg = "Requested PST: %r, which isn't in list." % (pst)
                raise Exception(msg)
            self.selection_objects.select_pst(pst, self.selection_data.transects[pst])
        self.full_redraw()

    def set_selected_glas(self, glas):
        # type: (int) -> None
        if self.selection_config.selected_glas == glas:
            return

        if self.selection_config.selected_glas is not None:
            self.selection_objects.deselect_glas(self.selection_config.selected_glas)
        self.selection_config.selected_glas = glas
        self.selection_objects.select_glas(glas)
        self.full_redraw()

    def set_selected_flight(self, flight):
        # type: (str) -> None
        old_flight = self.selection_config.selected_flight
        if old_flight == flight:
            return
        if old_flight is not None:
            self.selection_objects.deselect_flight(old_flight)
        self.selection_config.selected_flight = flight
        self.selection_objects.select_flight(flight)
        self.full_redraw()

    def set_background(self, label):
        # type: (str) -> Tuple[float, float]
        if label not in self.selection_objects.backgrounds:
            # This is an exception since the value is from a combo box ...
            msg =  "Uh-oh! %r not found. %r" % (label, self.selection_objects.backgrounds.keys())
            raise Exception(msg)
        # TODO: There's got to be a cleaner way of passing the set_background
        # function to the on_clicked function with the appropriate axis bound
        # to the first argument.
        self.selection_objects.background.clear()
        self.selection_objects.background = self.selection_objects.backgrounds[label]
        new_clim = self.selection_objects.background.set_background()
        self.full_redraw()
        return new_clim

    def set_background_clim(self, clim):
        # type: (Tuple[float, float]) -> None
        self.selection_objects.background.on_clim_changed(clim)
        self.full_redraw()

    def set_grounding_line(self, label):
        # type: (str) -> None
        devaUtilities.set_grounding_line(self.selection_objects.ax,
                                         self.selection_objects.grounding_lines,
                                         label)
        self.full_redraw()

    def set_glas_region(self, glas_line, xlim):
        # type: (int, Tuple[float, float]) -> None
        '''
        Passed into GLASFigure to update the portion of the GLAS line
        that's highlighted.
        '''
        self._show_glas_region(glas_line, xlim)
        self.full_redraw()

    def set_camera_cursor(self, coord):
        # type: (Tuple[float, float]) -> None
        '''
        Passed into CameraFigure to update the portion of the flight
        that's highlighted.
        '''
        self._show_camera_cursor(coord)
        self.full_redraw()

    def set_pst_region(self, pst, tlim):
        # type: (str, Tuple[float, float]) -> None
        '''
        Called by the radar figure (or other data figures) to update the
        portion of the PST that's displayed based on what part of the
        radargram is being displayed.
        * pst - string
        * tlim - (t0, t1) in posix time giving portion of PST to display
        '''
        self._show_pst_region(pst, tlim)
        self.full_redraw()

    def set_cursor(self, pst, tt):
        # type: (str, float) -> None
        '''
        Called by the radar figure (or other data figures) to have the map
        show a marker at the corresponding time on the PST.
        radargram is being displayed.
        * pst - string
        * tt - in posix time giving portion of PST to display
        '''
        self._show_cursor(pst, tt)
        self.full_redraw()

    def _on_button_release(self, event):
        # type: (QtGui.QMouseEvent) -> None
        '''
        Since an onpick event will often trigger multiple PSTs
        and/or GLAS lines, we only use the onpick callback to mark which
        ones were nearby, and don't do anything with them until
        all those callbacks have been called.
        Button release is guaranteed to happen after onpick, so this
        is where the selection work is triggered.
        '''
        if len(self.picked_psts) > 0 or len(self.picked_glas) > 0 or len(self.picked_flights) > 0:
            self.selection_cb(self.picked_psts, self.picked_glas, self.picked_flights)
        self.picked_psts.clear()
        self.picked_glas.clear()
        self.picked_flights.clear()

# TODO: This doesn't seem to be handling flight_lines correctly....
    def _on_pick(self, event):
        # type: (QtGui.QMouseEvent) -> None
        for pkey, pval in self.selection_objects.pst_lines.iteritems():
            if event.artist == pval and pval.get_visible():
                self.picked_psts.add(pkey)
                return
        for gkey, gval in self.selection_objects.glas_lines.iteritems():
            if event.artist == gval and gval.get_visible():
                self.picked_glas.add(gkey)
                return
        for fkey, fval in self.selection_objects.flight_lines.iteritems():
            if event.artist == fval and fval.get_visible():
                self.picked_flights.add(fkey)
                return

    def _on_lim_changed(self, ax):
        # type: (matplotlib.axes.Axes) -> None
        new_xlim = ax.get_xlim()
        new_ylim = ax.get_ylim()

        # Adding new lines to plots can trigger xlim_callback and ylim_callback
        # even when nothing truly changed.
        if self.xlim == new_xlim and self.ylim == new_ylim:
            return

        self.ax_changed_cb(new_xlim, new_ylim)
        # Attempting to limit zooming/panning to original dimensions.
        # x0 = max(new_xlim[0], self.orig_xlim[0])
        # x1 = min(new_xlim[1], self.orig_xlim[1])
        # y0 = max(new_ylim[0], self.orig_ylim[0])
        # y1 = min(new_ylim[1], self.orig_ylim[1])
        # if x0 > new_xlim[0] or x1 < new_xlim[1]:
        #     xlim = [x0, x1]
        #     self.ax.set_xlim(xlim)
        #     self.ax.axis('equal')
        # else:
        #     xlim = new_xlim
        # if y0 > new_ylim[0] or y1 < new_ylim[1]:
        #     ylim = [y0, y1]
        #     self.ax.set_ylim(ylim)
        #     self.ax.axis('equal')
        # else:
        #     ylim = new_ylim
        # self.ax_changed_cb(xlim, ylim)

    def _show_glas_region(self, glas_line, xlim):
        # type: (int, Tuple[float, float]) -> None
        # This is an exception b/c it should have been caught earlier.
        if glas_line not in self.selection_data.glas:
            raise Exception("GLAS line %r unknown to DEVA" % (glas_line))
        line = self.selection_data.glas[glas_line]
        xx = line[:,0]
        yy = line[:,1]
        idxs = [idx for idx,val in enumerate(xx) if xlim[0] <= val <= xlim[1]]
        self.selection_objects.glas_region.set_data(xx[idxs], yy[idxs])

# TODO: Should this have one for each flight, or just the one?
    def _show_camera_cursor(self, coord):
        # type: (Tuple[float, float]) -> None
        self.selection_objects.camera_cursor.set_data(coord[0], coord[1])

    def _show_pst_region(self, pst, tlim):
        # type: (str, Tuple[float, float]) -> None
        '''
        * pst - string
        * tlim - (t0, t1) in posix time giving portion of PST to display
        '''
        if pst not in self.selection_data.transects:
            raise Exception("Requested unrecognized pst: %r!" % (pst))
        if pst not in self.selection_objects.radar_pst_handles:
            raise Exception("Trying to update pst not in pst_handles: %r" % (pst))
        transect = self.selection_data.transects[pst]
        tt = transect[:,0]
        xx = transect[:,1]
        yy = transect[:,2]
        idxs = [idx for idx,val in enumerate(tt) if tlim[0] <= val <= tlim[1]]
        self.selection_objects.radar_pst_handles[pst]['region'].set_data(xx[idxs], yy[idxs])
        self.selection_objects.radar_pst_handles[pst]['start'].set_data(xx[idxs][0], yy[idxs][0])

    def _show_cursor(self, pst, input_time):
        # type: (str, float) -> None
        '''
        * pst - string
        * tt - in posix time giving portion of PST to display
        '''
        if pst not in self.selection_data.transects:
            raise Exception("Requested unrecognized pst: %r!" % (pst))
        if pst not in self.selection_objects.radar_pst_handles:
            raise Exception("Trying to update pst not in pst_handles: %r" % (pst))
        transect = self.selection_data.transects[pst]
        tt = transect[:,0]
        xx = transect[:,1]
        yy = transect[:,2]

        input_x = np.interp(input_time, tt, xx)
        input_y = np.interp(input_time, tt, yy)
        self.selection_objects.radar_pst_handles[pst]['cursor'].set_data(input_x, input_y)

class ContextFigures():
    '''
    Keeps two context maps updated, showing an Antarctica-wide view,
    and a more zoomed-in view.
    '''
    def __init__(self, wide_fig, mid_fig):
        # type: (Figure, Figure) -> None
        self.mid_fig = mid_fig
        self.wide_fig = wide_fig

        self.mid_ax = self.mid_fig.add_axes([0,0,1,1])
        self.mid_ax.axis('off')
        self.mid_ax.format_coord = lambda x,y: 'x=%dkm y=%dkm' % (int(x/1000), int(y/1000))
        self.mid_scalebar = plotutils.scalebar.Scalebar(self.mid_ax, 0.05, 0.07, 0.2, 0.03,
                                                        barstyle='simple', coords='frac',
                                                        unit_label='km', unit_factor=1000)

        self.wide_ax = self.wide_fig.add_axes([0,0,1,1])
        self.wide_ax.axis('off')
        self.wide_ax.format_coord = lambda x,y: 'x=%dkm y=%dkm' % (int(x/1000), int(y/1000))

        self.wide_scalebar = plotutils.scalebar.Scalebar(self.wide_ax, 0.05, 0.07, 0.2, 0.03,
                                                         barstyle='simple', coords='frac',
                                                         unit_label='km', unit_factor=1000)

        self.mid_backgrounds = plotutils.basemap.make_background_dict(
            self.mid_fig, self.mid_ax)
        self.wide_backgrounds = plotutils.basemap.make_background_dict(
            self.wide_fig, self.wide_ax)

        self.mid_background = self.mid_backgrounds['modis_simple']
        self.mid_background.set_background()
        self.wide_background = self.wide_backgrounds['modis_simple']
        self.wide_background.set_background()

        self.mid_ax.axis('equal')
        self.wide_ax.axis('equal')

        self.mid_rect, = self.mid_ax.plot(0, 0, color=dp.rect_color,
                                          zorder=dp.rect_zorder)
        self.wide_rect, = self.wide_ax.plot(0, 0, color=dp.rect_color,
                                            zorder=dp.rect_zorder)

        self.mid_fig.canvas.draw()
        self.wide_fig.canvas.draw()

    def set_background(self, label):
        # type: (str) -> None
        # TODO: this is cumbersome. Would be better to avoid having the clear()
        # by not creating the dict of class instances at the get-go.
        self.mid_background.clear()
        self.mid_background = self.mid_backgrounds[label]
        self.mid_background.set_background()
        self.mid_fig.canvas.draw()

    def set_zoom_limits(self, zoom_xlim, zoom_ylim):
        # type: (Tuple[float, float], Tuple[float, float]) -> None
        '''
        Called to keep the context plots in sync with the main map.

        Inputs:
        * zoom_xlim, zoom_ylim - limits of main map.
        '''
        xx = sum(zoom_xlim) / 2.
        yy = sum(zoom_ylim) / 2.
        zoom_dx = zoom_xlim[1] - zoom_xlim[0]
        zoom_dy = zoom_ylim[1] - zoom_ylim[0]

        wide_rect_frac = 8.
        wide_xlim = self.wide_ax.get_xlim()
        wide_ylim = self.wide_ax.get_ylim()
        wide_dx = wide_xlim[1] - wide_xlim[0]
        wide_dy = wide_ylim[1] - wide_ylim[0]
        mid_dx = max(wide_dx/wide_rect_frac, zoom_dx)
        mid_dy = max(wide_dy/wide_rect_frac, zoom_dy)
        mid_xlim = (xx - mid_dx/2., xx + mid_dx/2.)
        mid_ylim = (yy - mid_dy/2., yy + mid_dy/2.)
        self.mid_ax.set_xlim(mid_xlim)
        self.mid_ax.set_ylim(mid_ylim)

        wide_xrect, wide_yrect = self._get_rect(mid_xlim, mid_ylim)
        self.wide_rect.set_data(wide_xrect, wide_yrect)
        mid_xrect, mid_yrect = self._get_rect(zoom_xlim, zoom_ylim)
        self.mid_rect.set_data(mid_xrect, mid_yrect)

        self.mid_fig.canvas.draw()
        self.wide_fig.canvas.draw()

    @staticmethod
    def _get_rect(xbounds, ybounds):
        # type: (Tuple[float, float], Tuple[float, float]) -> Tuple[List[float], List[float]]
        '''
        Turns input list of [xmin, xmax] [ymin, ymax] into corners
        of a rectangle suitable for plotting.
        '''
        xx = [xbounds[0], xbounds[0], xbounds[1], xbounds[1], xbounds[0]]
        yy = [ybounds[0], ybounds[1], ybounds[1], ybounds[0], ybounds[0]]
        return xx, yy
