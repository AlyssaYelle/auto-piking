#!/usr/bin/env python2.7

# TODO: Additional integration with deva would be awesome! Ideally, this would
# take the form of letting radarWindow open a camera as well. That shouldn't
# be too hard - would require switching from coord to poxis in the callback.


import argparse
import numpy as np
import os
import signal
import sys

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# For development on melt, where we don't have PySide.
# However, PyQt is GPL'd, so we really shouldn't be using it. (PySide is LGPL)
# Before distributing/releasing this software, uncomment the PySide lines.
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
# import PySide.QtCore as QtCore
# import PySide.QtGui as QtGui
# matplotlib.rcParams['backend.qt4']='PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# TODO: I'm not sure where's the best spot to put this, given that all
# of my pyside code has something like it ...
# Trying to catch ctrl-c
def handleIntSignal(signum, frame):
    QtGui.qApp.closeAllWindows()
signal.signal(signal.SIGINT, handleIntSignal)

WAIS = os.getenv('WAIS')
if WAIS is None:
    raise Exception('WAIS is not set.')


class ImageInfo(object):
    def __init__(self, line, flight_path):
        '''
        * line - string of format [posix xx yy filename]
        * flight_path - FlightPath; used for recovering positions where
                jhead didn't show anything and for interpolating alternative
                timing info.
        '''
        tokens = line.split()
        self.posix = float(tokens[0])
        if tokens[1] == 'nan' or tokens[2] == 'nan':
            self.coord = flight_path.coord_from_posix(self.posix)
            self.gps_coord = False
        else:
            self.coord = np.array(map(float, tokens[1:3]))
            self.gps_coord = True

        self.filename = tokens[3]

        # find the closest point in the (heavily subsampled) flight_path,
        # then do an interpolation to estimate timestamp based on x and y position.
        if self.gps_coord:
            dx = flight_path.xx - self.coord[0]
            dy = flight_path.yy - self.coord[1]
            dists = np.sqrt(dx**2 + dy**2)
            idx = np.argmin(dists)
            if idx > 0:
                start_idx = idx - 1
            else:
                start_idx = 0
            if idx < len(dx) - 1:
                end_idx = idx + 2
            else:
                end_idx = len(dx)
            segt = flight_path.posix[start_idx:end_idx]
            segx = flight_path.xx[start_idx:end_idx]
            segy = flight_path.yy[start_idx:end_idx]

            self.interp_posix_x = np.interp(self.coord[0], segx, segt)
            self.interp_posix_y = np.interp(self.coord[1], segy, segt)
        else:
            self.interp_posix_x = None
            self.interp_posix_y = None




class FlightPath(object):
    def __init__(self, season, flight):
        # TODO: Would be good to have this data live in $WAIS/targ/xtra/ALL/deva/
        filename = ('%s/home/wais/lindzey/data/flight_visualizer/plane_traces/%s.%s.plane.txyz'
                    % (WAIS, season, flight))
        # Ugh. The flight names aren't consistent between index files and
        # flight_visualizer. Sometimes the 'a' sticks around, sometimes it
        # doesn't :-\ I think this is a problem in the hierarchy, not my code,
        # so just gotta suck it up.
        if not os.path.isfile(filename):
            filename = ('%s/home/wais/lindzey/data/flight_visualizer/plane_traces/%s.%s.plane.txyz'
                        % (WAIS, season, flight[0:3]))

        data = np.loadtxt(filename)
        self.posix = data[:,0]
        self.xx = data[:,1]
        self.yy = data[:,2]

    def coord_from_posix(self, in_posix):
        xx = np.interp(in_posix, self.posix, self.xx)
        yy = np.interp(in_posix, self.posix, self.yy)
        return np.array([xx, yy])


def load_image_database(season, flight):
    '''

    '''
    flight_path = FlightPath(season, flight)

    database = []
    # SUPER HACKY trying to deal with erratic presence of 'a' on some flights.
    index_filename = '%s/targ/xtra/ALL/deva/camera/%s.%s.index' % (WAIS, season, flight)
    if not os.path.isfile(index_filename):
        index_filename = '%s/targ/xtra/ALL/deva/camera/%s.%s.index' % (WAIS, season, flight.strip('a'))
    if not os.path.isfile(index_filename):
        index_filename = '%s/targ/xtra/ALL/deva/camera/%s.%sa.index' % (WAIS, season, flight)

    for line in open(index_filename):
        img = ImageInfo(line, flight_path)
        database.append(img)
    # TODO: actually construct an ORDERED database s.t. indices are in time =)
    # I'm pretty sure that there are functions that would allow me to sort
    # based on posix... OR, just guarantee that the index files are ordered =)
    return database


# TODO: Figure out what the parent_cb should be in. posix time makes the most sense?
class CameraWindow(QtGui.QMainWindow):
    def __init__(self, season, flight, posix=None, coord=None, parent=None,
                 parent_cursor_cb=None, close_cb=None):
        '''
        params:
        * season, flight - the program only supports paging through one
              flight of data at a time.
        * posix - timestamp to try to open at
        * coord - try to open at image closest to these coords
              (only one of posix, coords can be set)
        * parent_cursor_cb - callback to radarFigure when the image has been
          changed. Takes PS71s [xx,yy] coordinate
        * close_cb - callback to radarFigure when window closed (so it will know
          to open another one if necessary)
        '''
        super(CameraWindow, self).__init__(parent)
        self.season = season
        self.flight = flight
        self.parent_cursor_cb = parent_cursor_cb
        self.close_cb = close_cb

        # This is set in set_image, but want to initialize here
        self.image_index = None

        self.create_layout()

        # TODO: Load database, num_images = then len(database)
        self.database = load_image_database(season, flight)

        if posix is None and coord is None:
            image_index = 0
        elif posix is not None and coord is not None:
            msg = "cameraFigure only supports setting posix OR coords"
            raise Exception(msg)
        elif posix is not None:
            # TODO: actually find closest index to posix!
            image_index = 0
        elif coord is not None:
            # TODO: Actually find closest index to coords!
            image_index = 0

        self.set_image(image_index)

    def set_image(self, image_index):
        '''
        Sets viewed image to that in image.
        '''
        if image_index < 0 or image_index >= len(self.database):
            msg = "Invalid index %d. Database has %d entries" % (image_index, len(self.database))
            raise Exception(msg)
        self.image_index = image_index

        imageinfo = self.database[self.image_index]

        filename = imageinfo.filename
        im = plt.imread(filename)
        self.image_ax.imshow(im)
        self.canvas.draw()

        self.image_text.setText("Image: %d" % (self.image_index))
        self.posix_text.setText("Posix: %0.1f" % (imageinfo.posix))
        if imageinfo.gps_coord:
            self.interp_posix_text.setText("~Posix: %0.2f, %0.2f"
                                           % (imageinfo.interp_posix_x, imageinfo.interp_posix_y))
        else:
            self.interp_posix_text.setText("~Posix: n/a")
        self.coord_text.setText("PS71: %0.2f %0.2f" % (imageinfo.coord[0], imageinfo.coord[1]))
        self.filename_text.setText("file: %s" % (imageinfo.filename.split('orig/xped/')[1]))

        # TODO: Update slider bar
        # TODO: activate/deactivate next/prev buttons =)
        if self.parent_cursor_cb is not None:
            self.parent_cursor_cb(imageinfo.coord)


    # Increment/decrement image, and call appropriate callbacks, after checking
    # that it's possible to increment/decrement.
    # TODO: Update these callbacks to use correct units! I'm thinking posix?
    # TODO: Might be good to deactivate buttons?
    #       (But that would be a pain...easier to just pop up a message window.)
    def _on_next_button_clicked(self):
        if self.image_index + 1 < len(self.database):
            self.set_image(self.image_index + 1)

    def _on_prev_button_clicked(self):
        if self.image_index > 0:
            self.set_image(self.image_index - 1)

    def _on_first_button_clicked(self):
        self.set_image(0)

    def _on_last_button_clicked(self):
        self.set_image(len(self.database) - 1)

    def _on_quit_button_clicked(self):
        if self.close_cb is not None:
            self.close_cb()
        self.close()

# TODO: Convert this to posix timestamp, or just disable it?
# There's no guarantee of unque filenames in general camera data.

    def _on_image_textbox_edited(self):
        txt = self.image_textbox.text()
        try:
            value = int(txt)
        except ValueError:
            if txt != '':
                msg = "Could not convert %r to int." % (txt)
                print msg
                self.image_textbox.setText('')
            return
        self.image_textbox.setText('')
        self.set_image(value)


# TODO: add slider =)
    def create_layout(self):
        '''

        '''
        self.main_frame = QtGui.QWidget()
        self.fig = Figure((12, 9))
        self.image_ax = self.fig.add_axes([0,0,1,1])
        self.image_ax.axis('off')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)


        self.posix_text = QtGui.QLabel()
        self.interp_posix_text = QtGui.QLabel()
        self.coord_text = QtGui.QLabel()
        self.filename_text = QtGui.QLabel()
        info_vbox = QtGui.QVBoxLayout()
        info_vbox.addWidget(self.posix_text)
        info_vbox.addWidget(self.interp_posix_text)
        info_vbox.addWidget(self.coord_text)
        info_vbox.addWidget(self.filename_text)

        self.first_button = QtGui.QPushButton('<< First')
        self.connect(self.first_button, QtCore.SIGNAL('clicked()'),
                     self._on_first_button_clicked)

        self.prev_button = QtGui.QPushButton('< Prev')
        self.connect(self.prev_button, QtCore.SIGNAL('clicked()'),
                     self._on_prev_button_clicked)

        self.next_button = QtGui.QPushButton('Next > ')
        self.connect(self.next_button, QtCore.SIGNAL('clicked()'),
                     self._on_next_button_clicked)

        self.last_button = QtGui.QPushButton('Last >> ')
        self.connect(self.last_button, QtCore.SIGNAL('clicked()'),
                     self._on_last_button_clicked)

        self.quit_button = QtGui.QPushButton('Quit')
        self.connect(self.quit_button, QtCore.SIGNAL('clicked()'),
                     self._on_quit_button_clicked)

        self.image_text = QtGui.QLabel()

        self.image_textbox = QtGui.QLineEdit()
        self.image_textbox.setMinimumWidth(100)
        self.image_textbox.setMaximumWidth(100)
        self.connect(self.image_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_image_textbox_edited)

        image_vbox = QtGui.QVBoxLayout()
        image_vbox.addWidget(self.image_text)
        image_vbox.addWidget(self.image_textbox)

        control_hbox = QtGui.QHBoxLayout()
        control_hbox.addLayout(info_vbox)
        control_hbox.addStretch(1)
        control_hbox.addWidget(self.first_button)
        control_hbox.addWidget(self.prev_button)
        control_hbox.addLayout(image_vbox)
        # TODO: Add display and ability to set current image index
        control_hbox.addWidget(self.next_button)
        control_hbox.addWidget(self.last_button)
        control_hbox.addStretch(1)
        control_hbox.addWidget(self.quit_button)

        main_vbox = QtGui.QVBoxLayout()
        main_vbox.addWidget(self.canvas)
        main_vbox.addLayout(control_hbox)
        self.main_frame.setLayout(main_vbox)
        self.setCentralWidget(self.main_frame)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    parser = argparse.ArgumentParser(
        description='CameraFigure - interface for stepping through UTIG ELF data')
    parser.add_argument('season', help='')
    parser.add_argument('flight', help='Flight. For ICP3, include the a,b, etc.')
    parser.add_argument('--posix',
                        help='Attempt to initialize with image taken closest to this time')
    parser.add_argument('--coord', help='[x,y] in PS71s coordinates. Attempt to initialize with image taken closest to this position.')
    args = parser.parse_args()

    camera_window = CameraWindow(args.season, args.flight, args.posix, args.coord)
    camera_window.show()
    app.exec_()
