#!/usr/bin/env python2.7

import argparse
import numpy as np
import os
import signal
import sys
import time

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.backend_bases

# # Monkeypatching to get NavigationToolbar's save button to respect savefig.facecolor
# import plotUtilities
# matplotlib.backend_bases.FigureCanvasBase.print_figure = plotUtilities.print_figure
# matplotlib.rcParams['savefig.facecolor'] = 'silver'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.backend_bases import key_press_handler
import matplotlib.colors
from matplotlib.figure import Figure

# For development on melt, where we don't have PySide.
# However, PyQt is GPL'd, so we really shouldn't be using it. (PySide is LGPL)
# Before distributing/releasing this software, uncomment the PySide lines.
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
# import PySide.QtCore as QtCore
# import PySide.QtGui as QtGui
# matplotlib.rcParams['backend.qt4']='PySide'

from cameraFigure import CameraWindow
from glasFigure import GLASFigure
from radarFigure import ExperimentalRadarWindow as RadarWindow
import devaMaps
import devaUtilities # Only used for the GL dict
import plotUtilities
import mplUtilities
import qtWidgets

WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set')
sys.path.append('%s/syst/linux/py' % (WAIS))
import plotutils
import plotutils.basemap
import plotutils.monkeypatch

# Monkeypatching to get NavigationToolbar's save button to respect savefig.facecolor
matplotlib.backend_bases.FigureCanvasBase.print_figure = plotutils.monkeypatch.print_figure
matplotlib.rcParams['savefig.facecolor'] = 'silver'

try:
    import typing
    from typing import Any, Dict, List, Optional, Tuple
except:
    pass


# Trying to catch ctrl-c. This still requires tabbing over to the window to
# have it disappear, but that's better than a mouse click.
def handleIntSignal(signum, frame):
    # type: (int, Any) -> None
    QtGui.qApp.closeAllWindows()
signal.signal(signal.SIGINT, handleIntSignal)



class DevaWindow(QtGui.QMainWindow):
    def __init__(self, active_projects, pst, parent=None):
        # type: (str, str, Optional[Any]) -> None
        '''
        Inputs:
        * active_projects - comma-separated list of which projects to display.
        * pst - if set, pop this radar window up at the start.
        '''
        super(DevaWindow, self).__init__(parent)

        # restrict ourselves to plotting PSTs from these projects.
        self.active_projects = None # type: Optional[List[str]]
        if active_projects is not None:
            self.active_projects = active_projects.split(',')
        self.active_flights = None # type: Optional[List[str]]
        self.input_pst = pst
        self.selected_pst = None # type: Optional[str]
        self.selected_glas = None # type: Optional[int]
        self.selected_flight = None # type: Optional[str]

        self.glas_fig = None # type: Optional[Figure]
        self.camera_fig = None # type: Optional[Figure]
        self.radar_figs = {} # type: Dict[str, Figure]

        # Creates the axes that are passed into the other constructors
        self.create_layout()

        self.context_figures = devaMaps.ContextFigures(self.wide_fig,
                                                       self.mid_fig)
        self.selection_figure = devaMaps.SelectionFigure(self.zoom_fig,
                                                         self.active_projects,
                                                         self.ax_changed_cb,
                                                         self.selection_cb)
        self.pst_list = self.selection_figure.get_transect_list()
        self.glas_list = self.selection_figure.get_glas_list()
        self.flight_list = self.selection_figure.get_flight_list()

        self.setup_project_menu()
        self.setup_flight_menu()

        if self.input_pst in self.pst_list:
            self.selected_pst = self.input_pst
            self.pst_text.setText('PST: %s' % (self.selected_pst))
            self.selection_figure.set_selected_pst(self.selected_pst)
            self._on_show_radar_button_clicked()
        elif self.input_pst is not None:
            msg = ("You requested PST %r, " % self.input_pst +
                   "which is unknown to DEVA.\n\n"
                   "Please report this to Laura.\n\n"
                   "In the meantime, it may be possible to load it in a "
                   "standalone radarFigure instance.")
            plotUtilities.show_error_message_box(msg)

    def setup_project_menu(self):
        # type: () -> None
        '''
        Adds the appropriate items to the project menu. This requries the
        selection figure to have been initialized so we can query it for a
        list of available psts.
        '''
        projects = set([pst.split('/')[0] for pst in self.pst_list])
        self.project_actions = {} # type: Dict[str, QtGui.QAction]
        for proj in sorted(projects):
            self.project_actions[proj] = self.project_toolmenu.addAction(proj)
            # 2nd parameter is whether the box is checked.
            proj_cb = lambda x, proj=proj: self._on_project_menu_triggered(proj, x)
            self.project_actions[proj].triggered.connect(proj_cb)
            self.project_actions[proj].setCheckable(True)
            if self.active_projects is None or proj in self.active_projects:
                self.project_actions[proj].setChecked(True)
            else:
                self.project_actions[proj].setChecked(False)

    def setup_flight_menu(self):
        # type: () -> None
        '''
        Adds the appropriate items to the project menu. This requries the
        selection figure to have been initialized so we can query it for a
        list of available psts.
        '''
        projects = set([pst.split('/')[0] for pst in self.pst_list])
        self.flight_actions = {} # type: Dict[str, QtGui.QAction]
        for flight in sorted(self.flight_list):
            self.flight_actions[flight] = self.flight_toolmenu.addAction(flight)
            # 2nd parameter is whether the box is checked.
            flight_cb = lambda x, flight=flight: self._on_flight_menu_triggered(flight, x)
            self.flight_actions[flight].triggered.connect(flight_cb)
            self.flight_actions[flight].setCheckable(True)
            if self.active_flights is None or flight in self.active_flights:
                self.flight_actions[flight].setChecked(True)
            else:
                self.flight_actions[flight].setChecked(False)

    def ax_changed_cb(self, xlim, ylim):
        # type: (Tuple[float, float], Tuple[float, float]) -> None
        '''
        Called by the selection figure when axis limits have changed,
        allowing the main deva window to notify the context figures.
        '''
        self.context_figures.set_zoom_limits(xlim, ylim)

    # TODO: Add handling of selected flights!
    def selection_cb(self, picked_psts, picked_glas, picked_flights):
        # type: (List[str], List[int], List[str]) -> None
        '''
        Called by the selection figure when onpick has yielded a list of
        potential PSTs and/or GLAS lines. The main Deva window is in charge
        of actually switching the selected PST.
        '''
        print "selection callback called! flights:", picked_flights
        if len(picked_psts) > 0:
            # TODO: It would be nice to have this highlight the PST
            # when it's selected in the list, and only properly call selected
            # when 'OK' is pushed. However, this would require diving deeper
            # into QInputDialog than using the getItem convenience function.
            if len(picked_psts) > 1:
                pst_strings = [str(pst) for pst in picked_psts]
                text, ok = QtGui.QInputDialog.getItem(self,
                                                      'Select PST',
                                                      'Select PST',
                                                      pst_strings,
                                                      editable=False)
            else:
                text, ok = picked_psts.pop(), True
            if ok:
                self.selected_pst = str(text)
                # TODO: There should be a function for updating everything
                # that needs to change when a new PST is selected.
                self.pst_text.setText('PST: %s' % (self.selected_pst))
                self.selection_figure.set_selected_pst(self.selected_pst)

        if len(picked_glas) > 0 and self.glas_enabled_checkbox.isChecked():
            if len(picked_glas) > 1:
                glas_strings = [str(glas) for glas in picked_glas]
                text, ok = QtGui.QInputDialog.getItem(self,
                                                      'Select GLAS line',
                                                      'Select GLAS line',
                                                      glas_strings,
                                                      editable=False)
            else:
                text, ok = picked_glas.pop(), True
            if ok:
                self.selected_glas = int(text)
                # TODO: There should be a function for updating everything
                # that needs to change when a new GLAS line is selected.
                self.glas_text.setText('GLAS: %d' % (self.selected_glas))
                self.selection_figure.set_selected_glas(self.selected_glas)

        if len(picked_flights) > 0 and self.flight_enabled_checkbox.isChecked():
            if len(picked_flights) > 1:
                flight_strings = [str(flight) for flight in picked_flights]
                text, ok = QtGui.QInputDialog.getItem(self,
                                                      'Select Flight',
                                                      'Select Flight',
                                                      flight_strings,
                                                      editable=False)
            else:
                text, ok = picked_flights.pop(), True
            if ok:
                self.selected_flight = text
                self.flight_text.setText('Flight: %s' % (self.selected_flight))
                self.selection_figure.set_selected_flight(self.selected_flight)

    def create_layout(self):
        # type: () -> None
        self.setWindowTitle('deva')

        ###################
        # Set up the three canvases used for drawing the zoom plots
        # (question - is it better to use the same canvas, or different?
        #  ... it's just a question of canvas vs. subplot)
        self.main_frame = QtGui.QWidget()

        self.dpi = 100
        # This is the one with the selectable PSTs
        self.zoom_fig = Figure((12, 12), dpi = self.dpi)

        self.zoom_canvas = FigureCanvas(self.zoom_fig)
        self.zoom_canvas.setParent(self.main_frame)
        self.zoom_toolbar = mplUtilities.NavigationToolbar(self.zoom_canvas,
                                                            self.main_frame)
        # Keyboard shortcuts for navigation toolbar.
        # I'm only bothering to do it for the main one, since saving is
        # rare enough that I don't care about having to click.
        self.zoom_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.zoom_canvas.mpl_connect('key_press_event',
                                     self._on_zoom_canvas_key_press)

        # Medium-sized context map
        self.mid_fig = Figure((6,6), dpi = self.dpi)
        self.mid_canvas = FigureCanvas(self.mid_fig)
        self.mid_canvas.setParent(self.main_frame)
        self.mid_toolbar = mplUtilities.SaveToolbar(self.mid_canvas,
                                                     self.main_frame)

        # This one always shows all of Antarctica
        self.wide_fig = Figure((6,6), dpi = self.dpi)
        self.wide_canvas = FigureCanvas(self.wide_fig)
        self.wide_canvas.setParent(self.main_frame)
        self.wide_toolbar = mplUtilities.SaveToolbar(self.wide_canvas,
                                                      self.main_frame)

        ######################
        # Set up all the controls widgets.
        # combo box for the middle and zoom plots' backgrounds

        self.mid_bg_label = QtGui.QLabel('Context Background')
        self.mid_bg_combo = QtGui.QComboBox()
        self.mid_bg_combo.activated[str].connect(
            self._on_mid_bg_combo_activated)

        self.zoom_bg_label = QtGui.QLabel('Main Background')
        self.zoom_bg_combo = QtGui.QComboBox()
        self.zoom_bg_combo.activated[str].connect(
            self._on_zoom_bg_combo_activated)

        tmp_backgrounds = plotutils.basemap.make_background_dict()
        sorted_backgrounds = sorted(tmp_backgrounds.keys())
        for background in sorted_backgrounds:
            self.mid_bg_combo.addItem(background)
            self.zoom_bg_combo.addItem(background)
        # Context figure defaults to modis basemap
        self.mid_bg_combo.setCurrentIndex(
            sorted_backgrounds.index('modis_simple'))
        self.zoom_bg_combo.setCurrentIndex(
            sorted_backgrounds.index('modis_simple'))

        # combo box for the closest-zoom-level grounding line
        self.zoom_gl_label = QtGui.QLabel('Main GL')
        self.zoom_gl_combo = QtGui.QComboBox()
        self.zoom_gl_combo.addItem('')
        tmp_grounding_lines = devaUtilities.make_grounding_line_dict()
        sorted_grounding_lines = sorted(tmp_grounding_lines.keys())
        for gl in sorted_grounding_lines:
            self.zoom_gl_combo.addItem(gl)
        self.zoom_gl_combo.activated[str].connect(
            self._on_zoom_gl_combo_activated)
        self.gl_vbox = QtGui.QVBoxLayout()
        self.gl_vbox.addWidget(self.zoom_gl_label)
        self.gl_vbox.addWidget(self.zoom_gl_combo)
        self.gl_vbox.addStretch(1)

        # Button to exit (the little one in the corner is a PITA.
        self.quit_hbox = QtGui.QHBoxLayout()
        self.quit_button = QtGui.QPushButton('Quit')
        self.connect(self.quit_button, QtCore.SIGNAL('clicked()'),
                     self._on_quit_button_clicked)
        self.quit_hbox.addStretch(1)
        self.quit_hbox.addWidget(self.quit_button)

        # Create a button for selecting PSTs
        self.show_pst_vbox = QtGui.QVBoxLayout()
        self.pst_text = QtGui.QLabel('PST: %s' % self.selected_pst)
        self.pst_entry_hbox = QtGui.QHBoxLayout()
        self.pst_entry_textbox = QtGui.QLineEdit()
        self.pst_entry_button = QtGui.QPushButton('Select PST')
        self.connect(self.pst_entry_button, QtCore.SIGNAL('clicked()'),
                     self._on_pst_entry_button_clicked)
        self.pst_entry_hbox.addWidget(self.pst_entry_textbox)
        self.pst_entry_hbox.addWidget(self.pst_entry_button)
        self.show_radar_button = QtGui.QPushButton('Show Radar')
        self.show_pst_vbox.addWidget(self.pst_text)
        self.show_pst_vbox.addLayout(self.pst_entry_hbox)
        self.show_pst_vbox.addWidget(self.show_radar_button)
        self.show_pst_vbox.addStretch(1)
        self.connect(self.show_radar_button, QtCore.SIGNAL('clicked()'),
                     self._on_show_radar_button_clicked)

        # Key for which PSTs are selected
        self.pst_key_vbox = QtGui.QVBoxLayout()
        self.pst_key = qtWidgets.ColorKeyInterface(self, self.set_pst_color)
        self.pst_key_vbox.addWidget(self.pst_key)
        self.pst_key_vbox.addStretch(1)

        # Controls for which projects are displayed.
        self.project_vbox = QtGui.QVBoxLayout()
        self.project_toolbutton = QtGui.QToolButton(self)
        self.project_toolbutton.setText('Visible Projects')
        self.project_toolmenu = QtGui.QMenu(self)
        self.project_toolbutton.setMenu(self.project_toolmenu)
        self.project_toolbutton.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.show_all_projects_button = QtGui.QPushButton('Show All')
        self.connect(self.show_all_projects_button, QtCore.SIGNAL('clicked()'),
                     self._on_show_all_projects_button_clicked)
        self.hide_all_projects_button = QtGui.QPushButton('Hide All')
        self.connect(self.hide_all_projects_button, QtCore.SIGNAL('clicked()'),
                     self._on_hide_all_projects_button_clicked)

        self.project_vbox.addWidget(self.project_toolbutton)
        self.project_vbox.addWidget(self.show_all_projects_button)
        self.project_vbox.addWidget(self.hide_all_projects_button)
        self.project_vbox.addStretch(1)

        # The everything involving selecting/displaying GLAS lines
        self.glas_vbox = QtGui.QVBoxLayout()
        # hbox for displaying currently selected + enable checkbox
        self.glas_enabled_hbox = QtGui.QHBoxLayout()
        self.glas_text = QtGui.QLabel('GLAS: %r' % self.selected_glas)
        # Whether to even display the GLAS lines
        self.glas_enabled_checkbox = QtGui.QCheckBox('Show GLAS')
        self.connect(self.glas_enabled_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_glas_enabled_checkbox_changed)
        self.glas_enabled_vbox = QtGui.QVBoxLayout()
        self.glas_enabled_vbox.addWidget(self.glas_enabled_checkbox)
        self.glas_enabled_hbox.addWidget(self.glas_text)
        self.glas_enabled_hbox.addStretch(1)
        self.glas_enabled_hbox.addWidget(self.glas_enabled_checkbox)

        self.glas_entry_hbox = QtGui.QHBoxLayout()
        self.glas_entry_textbox = QtGui.QLineEdit()
        self.glas_entry_textbox.setMinimumWidth(70)
        self.glas_entry_textbox.setMaximumWidth(90)
        self.glas_entry_button = QtGui.QPushButton('Select GLAS')
        self.connect(self.glas_entry_button, QtCore.SIGNAL('clicked()'),
                     self._on_glas_entry_button_clicked)
        self.glas_entry_hbox.addWidget(self.glas_entry_textbox)
        self.glas_entry_hbox.addWidget(self.glas_entry_button)
        self.show_glas_button = QtGui.QPushButton('Show GLAS')

        self.glas_vbox.addLayout(self.glas_enabled_hbox)
        self.glas_vbox.addLayout(self.glas_entry_hbox)
        self.glas_vbox.addWidget(self.show_glas_button)
        self.glas_vbox.addStretch(1)
        self.connect(self.show_glas_button, QtCore.SIGNAL('clicked()'),
                     self._on_show_glas_button_clicked)


        # The everything involving selecting/displaying Flight lines
        self.flight_vbox = QtGui.QVBoxLayout()
        # hbox for displaying currently selected + enable checkbox
        self.flight_enabled_hbox = QtGui.QHBoxLayout()
        self.flight_text = QtGui.QLabel('Flight: %r' % self.selected_flight)
        # Whether to even display the GLAS lines
        self.flight_enabled_checkbox = QtGui.QCheckBox('Show Flights')
        self.connect(self.flight_enabled_checkbox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self._on_flight_enabled_checkbox_changed)
        self.flight_enabled_vbox = QtGui.QVBoxLayout()
        self.flight_enabled_vbox.addWidget(self.flight_enabled_checkbox)
        self.flight_enabled_hbox.addWidget(self.flight_text)
        self.flight_enabled_hbox.addStretch(1)
        self.flight_enabled_hbox.addWidget(self.flight_enabled_checkbox)

        self.flight_entry_hbox = QtGui.QHBoxLayout()
        self.flight_entry_textbox = QtGui.QLineEdit()
        self.flight_entry_textbox.setMinimumWidth(70)
        self.flight_entry_textbox.setMaximumWidth(90)
        self.flight_entry_button = QtGui.QPushButton('Select Flight')
        self.connect(self.flight_entry_button, QtCore.SIGNAL('clicked()'),
                     self._on_flight_entry_button_clicked)
        self.flight_entry_hbox.addWidget(self.flight_entry_textbox)
        self.flight_entry_hbox.addWidget(self.flight_entry_button)
        self.show_camera_button = QtGui.QPushButton('Show Camera')

        self.flight_vbox.addLayout(self.flight_enabled_hbox)
        self.flight_vbox.addLayout(self.flight_entry_hbox)
        self.flight_vbox.addWidget(self.show_camera_button)
        self.flight_vbox.addStretch(1)
        self.connect(self.show_camera_button, QtCore.SIGNAL('clicked()'),
                     self._on_show_camera_button_clicked)

        # Controls for which projects are displayed.
        self.flight_selection_vbox = QtGui.QVBoxLayout()
        self.flight_toolbutton = QtGui.QToolButton(self)
        self.flight_toolbutton.setText('Visible Flights')
        self.flight_toolmenu = QtGui.QMenu(self)
        self.flight_toolbutton.setMenu(self.flight_toolmenu)
        self.flight_toolbutton.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.show_all_flights_button = QtGui.QPushButton('Show All')
        self.connect(self.show_all_flights_button, QtCore.SIGNAL('clicked()'),
                     self._on_show_all_flights_button_clicked)
        self.hide_all_flights_button = QtGui.QPushButton('Hide All')
        self.connect(self.hide_all_flights_button, QtCore.SIGNAL('clicked()'),
                     self._on_hide_all_flights_button_clicked)

        self.flight_selection_vbox.addWidget(self.flight_toolbutton)
        self.flight_selection_vbox.addWidget(self.show_all_flights_button)
        self.flight_selection_vbox.addWidget(self.hide_all_flights_button)
        self.flight_selection_vbox.addStretch(1)

        self.transect_controls_vbox = QtGui.QVBoxLayout()
        self.transect_controls_vbox.addLayout(self.show_pst_vbox)
        self.transect_controls_vbox.addLayout(self.pst_key_vbox)
        self.transect_controls_vbox.addWidget(plotUtilities.HLine())
        self.transect_controls_vbox.addLayout(self.project_vbox)
        self.transect_controls_vbox.addWidget(plotUtilities.HLine())
        self.transect_controls_vbox.addLayout(self.glas_vbox)
        self.transect_controls_vbox.addWidget(plotUtilities.HLine())
        self.transect_controls_vbox.addLayout(self.flight_vbox)
        self.transect_controls_vbox.addLayout(self.flight_selection_vbox)



        self.clim_vbox = QtGui.QVBoxLayout()
        self.clim_label = QtGui.QLabel('basemap clim:')
        self.clim_min_hbox = QtGui.QHBoxLayout()
        self.clim_min_label = QtGui.QLabel('min')
        self.clim_min_textbox = QtGui.QLineEdit()
        self.clim_min_textbox.setMinimumWidth(70)
        self.clim_min_textbox.setMaximumWidth(90)
        # Ugh. textEdited goes at every time a new character is entered...
        #self.connect(self.clim_min_textbox, QtCore.SIGNAL('textEdited(QString)'),
        # returnPressed() ignores whether it's read only or not ...I'm dealing
        # with that by having it set to 'None', and the callback checks for that.
        self.connect(self.clim_min_textbox, QtCore.SIGNAL('returnPressed()'),
                     self._on_clim_textbox_edited)
        self.clim_min_hbox.addWidget(self.clim_min_label)
        self.clim_min_hbox.addWidget(self.clim_min_textbox)
        self.clim_min_hbox.addStretch(1.0)
        self.clim_max_hbox = QtGui.QHBoxLayout()
        self.clim_max_label = QtGui.QLabel('max')
        self.clim_max_textbox = QtGui.QLineEdit()
        self.clim_max_textbox.setMinimumWidth(70)
        self.clim_max_textbox.setMaximumWidth(90)

        self.connect(self.clim_max_textbox, QtCore.SIGNAL('returnPressed()'),
                     self._on_clim_textbox_edited)
        self.clim_max_hbox.addWidget(self.clim_max_label)
        self.clim_max_hbox.addWidget(self.clim_max_textbox)
        self.clim_max_hbox.addStretch(1.0)
        # For now, starting with modis_simple is hard-coded, and there is no cmap for that one
        self.clim_min_textbox.setReadOnly(True)
        self.clim_min_textbox.setText('None')
        self.clim_max_textbox.setReadOnly(True)
        self.clim_max_textbox.setText('None')

        self.clim_vbox.addWidget(self.clim_label)
        self.clim_vbox.addLayout(self.clim_min_hbox)
        self.clim_vbox.addLayout(self.clim_max_hbox)
        self.clim_vbox.addStretch(1.0)



        # Trying to rearrange all controls...
        # First column is context maps + context controls
        # second column is just the selection figure
        # third column is the various controls + the quit button.

        ######### 1
        self.context_vbox = QtGui.QVBoxLayout()
        self.context_vbox.addWidget(self.wide_canvas)
        self.context_vbox.addWidget(self.wide_toolbar)
        self.context_vbox.addWidget(self.mid_canvas)
        self.context_vbox.addWidget(self.mid_toolbar)
        self.context_vbox.addStretch(1.0)
        self.context_vbox.addWidget(plotUtilities.HLine())
        self.context_vbox.addWidget(self.mid_bg_label)
        self.context_vbox.addWidget(self.mid_bg_combo)
        self.context_vbox.addWidget(plotUtilities.HLine())
        self.context_vbox.addWidget(self.zoom_bg_label)
        self.context_vbox.addWidget(self.zoom_bg_combo)
        self.context_vbox.addLayout(self.clim_vbox)
        self.context_vbox.addWidget(plotUtilities.HLine())
        self.context_vbox.addLayout(self.gl_vbox)
        self.context_vbox.addWidget(plotUtilities.HLine())


        #########  2

        self.selection_vbox = QtGui.QVBoxLayout()
        self.selection_vbox.addWidget(self.zoom_canvas)
        self.selection_vbox.addWidget(self.zoom_toolbar)


        ######### 3
        self.controls_vbox = QtGui.QVBoxLayout()
        self.transect_widget = QtGui.QWidget()
        self.transect_widget.setLayout(self.transect_controls_vbox)

        self.survey_planning_widget = QtGui.QWidget()
        self.flight_planning_widget = QtGui.QWidget()

        self.controls_tabs = QtGui.QTabWidget()
        self.controls_tabs.addTab(self.transect_widget, "transects")
        #self.controls_tabs.addTab(self.survey_planning_widget, "survey")
        #self.controls_tabs.addTab(self.flight_planning_widget, "flight")

        self.controls_vbox.addWidget(self.controls_tabs)
        self.controls_vbox.addStretch(1)
        self.controls_vbox.addLayout(self.quit_hbox)


        self.hbox = QtGui.QHBoxLayout()

        context_widget = QtGui.QWidget()
        context_widget.setLayout(self.context_vbox)
        context_size = QtCore.QSize(450, 2000)
        context_widget.setMaximumSize(context_size)
        self.hbox.addWidget(context_widget)

        self.hbox.addLayout(self.selection_vbox)

        controls_widget = QtGui.QWidget()
        controls_widget.setLayout(self.controls_vbox)
        controls_size = QtCore.QSize(700, 2000)
        controls_widget.setMaximumSize(controls_size)
        self.hbox.addWidget(controls_widget)

        self.main_frame.setLayout(self.hbox)

        self.setCentralWidget(self.main_frame)

    def _on_project_menu_triggered(self, proj, checked):
        # type: (str, bool) -> None
        self.selection_figure.set_project_visible(proj, checked)
        self.selection_figure.redraw()

    def _set_all_projects(self, checked):
        # type: (bool) -> None
        for proj, action in self.project_actions.iteritems():
            action.setChecked(checked)
            self.selection_figure.set_project_visible(proj, checked)
        self.selection_figure.redraw()

    def _on_show_all_projects_button_clicked(self):
        # type: () -> None
        self._set_all_projects(True)

    def _on_hide_all_projects_button_clicked(self):
        # type: () -> None
        self._set_all_projects(False)

    def _on_flight_menu_triggered(self, flight, checked):
        # type: (str, bool) -> None
        self.selection_figure.set_flight_visible(flight, checked)
        self.selection_figure.redraw()

    def _set_all_flights(self, checked):
        # type: (bool) -> None
        for flight, action in self.flight_actions.iteritems():
            action.setChecked(checked)
            self.selection_figure.set_flight_visible(flight, checked)
        self.selection_figure.redraw()

    def _on_show_all_flights_button_clicked(self):
        # type: () -> None
        self._set_all_flights(True)

    def _on_hide_all_flights_button_clicked(self):
        # type: () -> None
        self._set_all_flights(False)


    def _on_clim_textbox_edited(self):
        # type: () -> None
        min_txt = self.clim_min_textbox.text()
        max_txt = self.clim_max_textbox.text()
        try:
            min_value = float(min_txt)
            max_value = float(max_txt)
        except ValueError:
            msg = ''
            if min_txt == 'None' or max_txt == None:
                msg = "Current basemap does not support modifying clim"
            elif min_txt == '' or min_txt == '':
                msg = "Could not convert %r or %r to float." % (min_txt, max_txt)
            plotUtilities.show_error_message_box(msg)
            return
        if min_value >= max_value:
            msg = 'min value must be less than max value!'
            plotUtilities.show_error_message_box(msg)
            return
        self.selection_figure.set_background_clim((min_value, max_value))

    def _on_zoom_canvas_key_press(self, event):
        # type: (QtGui.QMouseEvent) -> None
        key_press_handler(event, self.zoom_canvas, self.zoom_toolbar)

    # TODO: This is awful. Belongs in the ContextFigure, not DevaWindow!
    def _on_mid_bg_combo_activated(self, label):
        # type: (unicode) -> None
        self.context_figures.set_background(str(label))

    def _on_zoom_bg_combo_activated(self, label):
        # type: (unicode) -> None
        new_clim = self.selection_figure.set_background(str(label))
        # TODO: Get clim (if available ... if not, deactivate the box.)
        if new_clim is None:
            self.clim_min_textbox.setReadOnly(True)
            self.clim_max_textbox.setReadOnly(True)
            self.clim_min_textbox.setText('None')
            self.clim_max_textbox.setText('None')
        else:
            self.clim_min_textbox.setReadOnly(False)
            self.clim_max_textbox.setReadOnly(False)
            self.clim_min_textbox.setText(str(new_clim[0]))
            self.clim_max_textbox.setText(str(new_clim[1]))

    def _on_zoom_gl_combo_activated(self, label):
        # type: (unicode) -> None
        self.selection_figure.set_grounding_line(str(label))

    def _on_quit_button_clicked(self):
        # type: () -> None
        self.close()

    def on_radar_window_quit(self, pst):
        # type: (str) -> None
        del self.radar_figs[pst]
        self.pst_key.remove_row(pst)
        self.selection_figure.on_radar_window_quit(pst)

    def _on_show_radar_button_clicked(self):
        # type: () -> None
        self.selection_figure.set_selected_pst(None)
        pst = self.selected_pst
        self.selected_pst = None
        self.pst_text.setText('PST: None')

        if pst is None:
            msg = "Must select PST first."
            plotUtilities.show_error_message_box(msg)
            return

        if pst in self.radar_figs:
            msg = "There is already a window open for %s" % (pst)
            plotUtilities.show_error_message_box(msg)
            return

        # Actually create the RadarWindow, cause it to pop up
        color = 'purple'
        self.selection_figure.add_radar_figure(pst, color)
        self.pst_key.add_row(pst, color)

        xlim_cb = lambda x: self.selection_figure.set_pst_region(pst, x)
        cursor_cb = lambda x: self.selection_figure.set_cursor(pst, x)
        close_cb = lambda: self.on_radar_window_quit(pst)
        self.radar_figs[pst] = RadarWindow(pst, parent=self,
                                           parent_xlim_changed_cb=xlim_cb,
                                           parent_cursor_cb=cursor_cb,
                                           close_cb=close_cb)
        self.radar_figs[pst].show()

        ## I couldn't figure out how to connect to the QApplication.aboutToQuit signal :(
        ## NB - need to add "from __future__ import print_function" for the test to work.
        #self.radar_figs[pst].aboutToQuit.connect(
        #    lambda: print("radar window for %s is closing" % (pst)))

    def _on_show_glas_button_clicked(self):
        # type: () -> None
        if self.selected_glas is None:
            msg = "Must select GLAS line first."
            plotUtilities.show_error_message_box(msg)
            return
        if self.glas_fig is not None:
            self.glas_fig.close()
        self.glas_fig = GLASFigure(self.selected_glas,
                                   self.selection_figure.set_glas_region)

    def _on_show_camera_button_clicked(self):
        # type: () -> None
        if self.selected_flight is None:
            msg = "Must select flight first."
            plotUtilities.show_error_message_box(msg)
            return
        if self.camera_fig is not None:
            self.camera_fig.close()
        season, flight = self.selected_flight.split('/')
        self.camera_fig = CameraWindow(season, flight,
                                       parent_cursor_cb=self.selection_figure.set_camera_cursor)
        self.camera_fig.show()

    def _on_pst_entry_button_clicked(self):
        # type: () -> None
        pst = str(self.pst_entry_textbox.text())
        if pst not in self.pst_list:
            msg = ("You requested PST %r, which is unknown to DEVA." % (pst) +
                   "\n\nIf you didn't exclude the corresponding project "
                   "at the command line, please report this to Laura. \n\n"
                   "In the meantime, it may be possible to load it in a "
                   "standalone radarFigure instance.")
            plotUtilities.show_error_message_box(msg)
            return
        self.pst_entry_textbox.setText('')
        self.selected_pst = pst
        self.pst_text.setText('PST: %s' % (self.selected_pst))
        self.selection_figure.set_selected_pst(self.selected_pst)

    def _on_glas_entry_button_clicked(self):
        # type: () -> None
        glas_text = self.glas_entry_textbox.text()
        try:
            glas_line = int(glas_text)
        except ValueError:
            msg = "Input GLAS line must be integer"
            plotUtilities.show_error_message_box(msg)
            return
        if glas_line not in self.glas_list:
            msg = ("You requested GLAS line: %r, " % glas_line +
                   "which is unknown to DEVA. \n\n"
                   "Please report this to Laura")
            plotUtilities.show_error_message_box(msg)
            return
        self.glas_entry_textbox.setText('')
        self.selected_glas = glas_line
        self.glas_text.setText('GLAS: %r' % self.selected_glas)
        self.selection_figure.set_selected_glas(self.selected_glas)

    def _on_flight_entry_button_clicked(self):
        # type: () -> None
        flight = str(self.flight_entry_textbox.text())
        if flight not in self.flight_list:
            msg = ("You requested flight: %r, " % flight +
                   "which is unknown to DEVA. \n\n"
                   "Please report this to Laura")
            plotUtilities.show_error_message_box(msg)
            return
        self.flight_entry_textbox.setText('')
        self.selected_flight = flight
        self.flight_text.setText('Flight: %r' % self.selected_flight)
        self.selection_figure.set_selected_flight(self.selected_flight)

    def _on_glas_enabled_checkbox_changed(self, _val):
        # type: (int) -> None
        visible = self.glas_enabled_checkbox.isChecked()
        self.selection_figure.change_glas_visibility(visible)
        self.zoom_fig.canvas.draw()

    def _on_flight_enabled_checkbox_changed(self, _val):
        # type: (int) -> None
        visible = self.flight_enabled_checkbox.isChecked()
        self.selection_figure.change_flight_visibility(visible)
        self.zoom_fig.canvas.draw()

    def set_pst_color(self, pst, color):
        # type: (str, QtGui.QColor) -> None
        '''
        Pass-through method required b/c the selection_figure hasn't been
        initialized at the time that the layout is being set up.
        '''
        self.selection_figure.set_pst_color(pst, color)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    parser = argparse.ArgumentParser(description = 'deva')
    parser.add_argument('-project', nargs='?', default=None,
                        help='Comma-separated list of Project lines to show. Defaults to all.')
    parser.add_argument('-pst', nargs='?', default=None,
                        help='Initial radar pst to show. (optional)')
    args = parser.parse_args()
    print("deva called with project = %r, pst = %r" % (args.project, args.pst))

    deva_window = DevaWindow(args.project, args.pst)
    deva_window.show()
    app.exec_()
