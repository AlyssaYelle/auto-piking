import itertools

# For development on melt, where we don't have PySide.
# However, PyQt is GPL'd, so we really shouldn't be using it. (PySide is LGPL)
# Before distributing/releasing this software, uncomment the PySide lines.
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
# import PySide.QtCore as QtCore
# import PySide.QtGui as QtGui
# matplotlib.rcParams['backend.qt4']='PySide'

# SDK attempted to install typing on melt, but it failed.
# This is fully optional, and LEL can run it on her machine.
try:
    import typing
    from typing import Any, Callable, Dict, List, Optional, Tuple
except:
    pass

import plotUtilities

class DoubleSlider(QtGui.QWidget):
    '''
    Widget that provides two sliders as a way to update the integer min/max
    value for a range.
    * Does not force textbox values to be within the range of the bar.
    NB - does force minval <= maxval
    TODO: Would be nicer if it was a single slider flanked by text boxes,
    but I didn't immediately see how to do that.
    '''
    def __init__(self,
                 parent=None, # type: Optional[Any]
                 new_lim_cb=None, # type: Optional[Callable[Tuple[int, int]]]
                 curr_lim=(0, 1) # type: Tuple[int, int]
                ):
        # type: (...) -> None
        '''
        * new_lim_cb([min,max]) - callback to call whenever either side
          of the limit changes.
        '''
        super(DoubleSlider, self).__init__(parent)
        self.parent = parent
        self.new_lim_cb = new_lim_cb

        self.curr_lim = curr_lim

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        # TODO - have sliders only call update when _released_,
        # otherwise it may try to redraw the image tons.
        self.min_slider_label1 = QtGui.QLabel('MIN')
        self.min_slider_label2 = QtGui.QLabel(str(self.curr_lim[0]))
        self.min_slider_label3 = QtGui.QLabel(str(self.curr_lim[1]))
        self.min_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.min_slider.setRange(self.curr_lim[0], self.curr_lim[1])
        self.min_slider.setValue(self.curr_lim[0])
        self.min_slider.setTracking(False)
        self.connect(self.min_slider, QtCore.SIGNAL('valueChanged(int)'),
                     self._on_min_slider_changed)

        self.min_slider_textbox = QtGui.QLineEdit()
        self.min_slider_textbox.setMinimumWidth(90)
        self.min_slider_textbox.setMaximumWidth(120)
        self.min_slider_textbox.setText(str(self.curr_lim[0]))
        self.connect(self.min_slider_textbox, QtCore.SIGNAL('editingFinished()'),
                     self._on_min_slider_textbox_edited)

        min_slider_hbox = QtGui.QHBoxLayout()
        min_slider_hbox.addWidget(self.min_slider_label1)
        min_slider_hbox.addStretch(1)
        min_slider_hbox.addWidget(self.min_slider_label2)
        min_slider_hbox.addWidget(self.min_slider)
        min_slider_hbox.addWidget(self.min_slider_label3)
        min_slider_hbox.addStretch(1)
        min_slider_hbox.addWidget(self.min_slider_textbox)

        # And now for the max-bounds slider ...
        self.max_slider_label1 = QtGui.QLabel('MAX')
        self.max_slider_label2 = QtGui.QLabel(str(self.curr_lim[0]))
        self.max_slider_label3 = QtGui.QLabel(str(self.curr_lim[1]))
        self.max_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.max_slider.setRange(self.curr_lim[0], self.curr_lim[1])
        self.max_slider.setValue(self.curr_lim[1])
        # Don't continually call valueChanged
        self.max_slider.setTracking(False)
        self.connect(self.max_slider, QtCore.SIGNAL('valueChanged(int)'),
                     self._on_max_slider_changed)

        self.max_slider_textbox = QtGui.QLineEdit()
        self.max_slider_textbox.setMaximumWidth(90)
        self.max_slider_textbox.setMaximumWidth(120)
        self.max_slider_textbox.setText(str(self.curr_lim[1]))
        self.connect(self.max_slider_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_max_slider_textbox_edited)

        max_slider_hbox = QtGui.QHBoxLayout()
        max_slider_hbox.addWidget(self.max_slider_label1)
        max_slider_hbox.addStretch(1)
        max_slider_hbox.addWidget(self.max_slider_label2)
        max_slider_hbox.addWidget(self.max_slider)
        max_slider_hbox.addWidget(self.max_slider_label3)
        max_slider_hbox.addStretch(1)
        max_slider_hbox.addWidget(self.max_slider_textbox)

        self.layout.addLayout(min_slider_hbox)
        self.layout.addLayout(max_slider_hbox)

    # TODO: It would be nice to have this as a decorator, rather
    # than calling both independently, but that got complicated ...
    def disconnect_callbacks(self):
        # type: () -> None
        '''
        Disables all callbacks associated with updating the sliders and
        textboxes. Does NOT disable the new_lim_cb
        (which really would be setting it to None).
        '''
        self.disconnect(self.min_slider_textbox,
                        QtCore.SIGNAL('editingFinished()'),
                        self._on_min_slider_textbox_edited)
        self.disconnect(self.min_slider,
                        QtCore.SIGNAL('valueChanged(int)'),
                        self._on_min_slider_changed)
        self.disconnect(self.max_slider_textbox,
                        QtCore.SIGNAL('editingFinished()'),
                        self._on_max_slider_textbox_edited)
        self.disconnect(self.max_slider,
                        QtCore.SIGNAL('valueChanged(int)'),
                        self._on_max_slider_changed)

    def reconnect_callbacks(self):
        # type: () -> None
        self.connect(self.min_slider_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_min_slider_textbox_edited)
        self.connect(self.min_slider,
                     QtCore.SIGNAL('valueChanged(int)'),
                     self._on_min_slider_changed)
        self.connect(self.max_slider_textbox,
                     QtCore.SIGNAL('editingFinished()'),
                     self._on_max_slider_textbox_edited)
        self.connect(self.max_slider,
                     QtCore.SIGNAL('valueChanged(int)'),
                     self._on_max_slider_changed)

    def set_range(self, lim):
        # type: (Tuple[int, int]) -> None
        '''
        Resetting the range of the slider automatically sets it to
        be at the full range.
        Does not trigger any callbacks.
        '''
        rmin, rmax = lim
        self.disconnect_callbacks()
        self.max_slider.setRange(rmin, rmax)
        self.max_slider_label2.setText(str(rmin))
        self.max_slider_label3.setText(str(rmax))
        self.min_slider.setRange(rmin, rmax)
        self.min_slider_label2.setText(str(rmin))
        self.min_slider_label3.setText(str(rmax))
        self.reconnect_callbacks()
        self.set_value(lim)

    def set_value(self, lim):
        # type: (Tuple[int, int]) -> None
        '''
        Updates the slider values, w/o changing their range.
        Does not trigger callbacks.
        '''
        self.curr_lim = lim
        rmin, rmax = lim
        self.disconnect_callbacks()
        self.max_slider.setValue(rmax)
        self.max_slider_textbox.setText(str(rmax))
        self.min_slider.setValue(rmin)
        self.min_slider_textbox.setText(str(rmin))
        self.reconnect_callbacks()

    def _on_min_slider_changed(self):
        # type: () -> None
        input_min = self.min_slider.value()
        self.update_min_value(input_min)

    def _on_min_slider_textbox_edited(self):
        # type: () -> None
        input_min = int(self.min_slider_textbox.text())
        self.update_min_value(input_min)

    def update_min_value(self, input_min):
        # type: (int) -> None
        # min can't be bigger than max
        cmin = min(self.curr_lim[1], input_min)
        self.disconnect_callbacks()
        self.min_slider.setValue(cmin)
        self.min_slider_textbox.setText(str(cmin))
        self.reconnect_callbacks()
        self.curr_lim = (cmin, self.curr_lim[1])
        if self.new_lim_cb is not None:
            self.new_lim_cb(self.curr_lim)

    def _on_max_slider_changed(self):
        # type: () -> None
        input_max = self.max_slider.value()
        self.update_max_value(input_max)

    def _on_max_slider_textbox_edited(self):
        # type: () -> None
        input_max = int(self.max_slider_textbox.text())
        self.update_max_value(input_max)

    def update_max_value(self, input_max):
        # type: (int) -> None
        # max can't be smaller than min
        cmax = max(self.curr_lim[0], input_max)
        self.disconnect_callbacks()
        self.max_slider.setValue(cmax)
        self.max_slider_textbox.setText(str(cmax))
        self.reconnect_callbacks()
        self.curr_lim = (self.curr_lim[0], cmax)
        if self.new_lim_cb is not None:
            self.new_lim_cb(self.curr_lim)


class ColorKeyInterface(QtGui.QWidget):
    '''
    Widget that provides a way to select the color of a label.
    '''
    def __init__(self,
                 parent=None, # type: Optional[Any]
                 color_cb=None # type: Optional[Callable[str, QtGui.QColor]]
                 ):
        # type: (...) -> None
        '''
        * color_cb(label, color) - callback to call when color is changed.
        '''
        super(ColorKeyInterface, self).__init__(parent)
        self.parent = parent
        self.color_cb = color_cb

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.labels = [] # type: List[str]
        self.textlabels = {} # type: Dict[str, QtGui.QColor]
        self.colorbuttons = {} # type: Dict[str, QtGui.QPushButton]
        self.row_hboxes = {} # type: Dict[str, QtGui.QHBoxLayout]

    def on_color_button_clicked(self, label):
        # type: (str) -> None
        # Pop up color dialog
        # if color not none, change button color AND call self.color_cb(color)
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            print "User requested %s" % (color.name())
            self.colorbuttons[label].setStyleSheet('QPushButton {background-color: %s}' % (color.name()))
            self.color_cb(label, str(color.name()))

    def add_row(self, label, color):
        # type: (str, QtGui.QColor) -> None
        '''
        The label here will be used when calling color_cb.
        '''
        self.labels.append(label)
        self.colorbuttons[label] = QtGui.QPushButton('')
        self.connect(self.colorbuttons[label], QtCore.SIGNAL('clicked()'),
                     lambda: self.on_color_button_clicked(label))
        self.colorbuttons[label].setStyleSheet('QPushButton {background-color: %r}' % (color))
        self.colorbuttons[label].setFixedSize(20, 20)

        self.textlabels[label] = QtGui.QLabel(label)

        self.row_hboxes[label] = QtGui.QHBoxLayout()
        self.row_hboxes[label].addWidget(self.colorbuttons[label])
        self.row_hboxes[label].addWidget(self.textlabels[label])
        self.layout.addLayout(self.row_hboxes[label])

    def remove_row(self, label):
        # type: (str) -> None
        self.labels.remove(label)
        self.layout.removeItem(self.row_hboxes[label])
        self.textlabels[label].deleteLater()
        del self.textlabels[label]
        self.colorbuttons[label].deleteLater()
        del self.colorbuttons[label]

class TextColorInterface(QtGui.QWidget):
    '''
    Widget that provides:
    a label, two text entry boxes, a color selector, and a button
    for each row added.
    '''
    def __init__(self,
                 parent=None, # type: Optional[Any]
                 color_cb=None, # type: Optional[Callable[str, QtGui.QColor]]
                 params_cb=None, # type: Optional[Callable[str, Tuple[float, float, str]]]
                 remove_cb=None  # type: Optional[Callable[str]]
                ):
        # type: (...) -> None
        '''
        * color_cb(label, color)
        * params_cb(label, params), where params is (float, float, str)
        * remove_cb(label)
        '''
        # TODO: I'm not sure what this is needed for?
        super(TextColorInterface, self).__init__(parent)
        self.parent = parent

        self.color_cb = color_cb
        self.params_cb = params_cb
        self.remove_cb = remove_cb

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.labels = [] # type: List[str]
        self.text_labels = {} # type: Dict[str, QtGui.QLabel]
        #self.spinboxes = {}
        self.first_textboxes = {} # type: Dict[str, QtGui.QLineEdit]
        self.second_textboxes = {} # type: Dict[str, QtGui.QLineEdit]
        self.color_buttons = {} # type: Dict[str, QtGui.QPushButton]
        self.colors = {} # type: Dict[str, QtGui.QColor]
        self.val1 = {} # type: Dict[str, float]
        self.val2 = {} # type: Dict[str, float]
        self.remove_buttons = {} # type: Dict[str, QtGui.QPushButton]
        self.row_hboxes = {} # type: Dict[str, QtGui.QHBoxLayout]

    def _on_color_button_clicked(self, label):
        # type: (str) -> None
        # Pop up color dialog
        # if color not none, change button color AND call self.color_cb(color)
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            stylestr = 'QPushButton {background-color: %s}' % (color.name())
            self.color_buttons[label].setStyleSheet(stylestr)
            self.colors[label] = str(color.name())
            if self.color_cb is not None:
                self.color_cb(label, str(color.name()))

    def _on_remove_button_clicked(self, label):
        # type: (str) -> None
        self.remove_row(label)
        if self.remove_cb is not None:
            self.remove_cb(label)

    def _on_textbox_edited(self, textbox, label):
        # type: (int, str) -> None
        try:
            val1 = float(self.first_textboxes[label].text())
            self.val1[label] = val1
        except:
            msg = "unable to cast textbox to float!"
            plotUtilities.show_error_message_box(msg)
            self.first_textboxes[label].setText(str(self.val1[label]))
            return
        try:
            val2 = float(self.second_textboxes[label].text())
            self.val2[label] = val2
        except:
            msg = "unable to cast textbox to float!"
            plotUtilities.show_error_message_box(msg)
            self.first_textboxes[label].setText(str(self.val2[label]))
            return

        params = (self.val1[label], self.val2[label], self.colors[label])
        if self.params_cb is not None:
            self.params_cb(label, params)

    def _on_spinbox_changed(self, label):
        # type: (str) -> None
        print "spinbox for data %s changed to %f" % (label, self.spinboxes[label].value())

    def add_row(self, label, box1_val, box2_val, color):
        # type: (str, float, float, QtGui.QColor) -> None
        self.labels.append(label)
        self.val1[label] = box1_val
        self.val2[label] = box2_val
        self.colors[label] = color

        self.text_labels[label] = QtGui.QLabel(label)

        # I couldn't figure out how to attach to the valueChanged signal,
        # so for now, I'm just going to stick with the text-only entry method.
        # self.spinboxes[label] = QtGui.QDoubleSpinBox()
        # self.spinboxes[label].setMinimum(0.0)
        # self.spinboxes[label].setMaximum(1.0)
        # self.spinboxes[label].setSingleStep(0.05)
        # #self.spinboxes[label].valueChanged().connect(lambda: self._on_spinbox_changed(label))
        # self.connect(self.spinboxes[label],
        #              #QtCore.SIGNAL('QtGui.QDoubleSpinBox.valueChanged()'),
        #              QtCore.SIGNAL('valueChanged(int)'),
        #              lambda: self._on_spinbox_changed(label))

        self.first_textboxes[label] = QtGui.QLineEdit()
        self.first_textboxes[label].setMaximumWidth(60)
        self.first_textboxes[label].setMaximumWidth(70)
        self.first_textboxes[label].setText(str(box1_val))
        self.connect(self.first_textboxes[label],
                     QtCore.SIGNAL('editingFinished()'),
                     lambda: self._on_textbox_edited(1, label))

        self.second_textboxes[label] = QtGui.QLineEdit()
        self.second_textboxes[label].setMaximumWidth(60)
        self.second_textboxes[label].setMaximumWidth(70)
        self.second_textboxes[label].setText(str(box2_val))
        self.connect(self.second_textboxes[label],
                     QtCore.SIGNAL('editingFinished()'),
                     lambda: self._on_textbox_edited(2, label))

        self.color_buttons[label] = QtGui.QPushButton('')
        self.connect(self.color_buttons[label], QtCore.SIGNAL('clicked()'),
                     lambda: self._on_color_button_clicked(label))
        self.color_buttons[label].setStyleSheet('QPushButton {background-color: %r}' % (color))
        self.color_buttons[label].setFixedSize(20, 20)

        self.remove_buttons[label] = QtGui.QPushButton('remove')
        self.connect(self.remove_buttons[label], QtCore.SIGNAL('clicked()'),
                     lambda: self._on_remove_button_clicked(label))

        self.row_hboxes[label] = QtGui.QHBoxLayout()
        self.row_hboxes[label].addWidget(self.color_buttons[label])
        self.row_hboxes[label].addWidget(self.text_labels[label])
        self.row_hboxes[label].addStretch(1)
        self.row_hboxes[label].addWidget(self.first_textboxes[label])
        #self.row_hboxes[label].addWidget(self.spinboxes[label])
        self.row_hboxes[label].addWidget(self.second_textboxes[label])
        self.row_hboxes[label].addWidget(self.remove_buttons[label])

        self.layout.addLayout(self.row_hboxes[label])

    def remove_row(self, label):
        # type: (str) -> None
        self.labels.remove(label)
        self.layout.removeItem(self.row_hboxes[label])

        for elem in [self.color_buttons, self.text_labels, self.first_textboxes, self.second_textboxes, self.remove_buttons]:
            elem[label].deleteLater()
            del elem[label]


class RadioCheckInterface(QtGui.QWidget):
    '''
    Widget that provides a set of radio buttons and check boxes for the
    same list of labels. Useful for editing picks, where we want to control
    the visibility of multiple maxima, but only be actively editing one
    pick file at a time.
    '''
    def __init__(self,
                 parent=None, # type: Optional[Any]
                 radio_cb=None, # type: Optional[Callable[str]]
                 check_cb=None, # type: Optional[Callable[str, bool]]
                 color_cb=None, # type: Optional[Callable[str, any]]
                ):
        # type: (...) -> None
        '''
        * radio_cb(string) - callback to call when any radio button clicked.
          (if one is selected, others are all unclicked).
          Arg is the button's label.
        * check_cb(string, bool) - callback for whenever a checkbox changes
          state. Args are the box's label, and it's resulting state.
        * color_cb(string, color) - callback for setting a pick line to a
          different color
        '''
        super(RadioCheckInterface, self).__init__(parent)
        self.parent = parent
        self.radio_cb = radio_cb
        self.check_cb = check_cb
        self.color_cb = color_cb

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        # After grouping by rows, the title no longer aligned.
        # self.radiobutton_label = QtGui.QLabel('active')
        # self.checkbox_label = QtGui.QLabel('show')
        # self.textlabel_label = QtGui.QLabel('pick file')
        # self.colorbutton_label = QtGui.QLabel('color')

        # self.title_hbox = QtGui.QHBoxLayout()
        # self.title_hbox.addWidget(self.radiobutton_label)
        # self.title_hbox.addWidget(self.checkbox_label)
        # self.title_hbox.addStretch(1)
        # self.title_hbox.addWidget(self.textlabel_label)
        # self.title_hbox.addWidget(self.colorbutton_label)

        # self.layout.addLayout(self.title_hbox)

        self.row_hboxes = {} # type: Dict[str, QtGui.QHBoxLayout]
        self.checkboxes = {} # type: Dict[str, QtGui.QCheckBox]
        self.radiobuttons = {} # type: Dict[str, QtGui.QRadioButton]
        self.textlabels = {} # type: Dict[str, QtGui.QLabel]
        self.colorbuttons = {} # type: Dict[str, QtGui.QPushButton]
        # used for looking up colors
        self.colors = {} # type: Dict[str, QtGui.QColor]

        # When adding buttons to the group, set the id to the label's index
        # in the labels array.
        self.labels = [] # type: List[str]
        self.radio_group = QtGui.QButtonGroup()
        self.connect(self.radio_group, QtCore.SIGNAL('buttonPressed(int)'),
                     self.on_radio_button_pressed)

        # Same for the checkbox goup, but this is a non-exclusive group.
        self.checkbox_group = QtGui.QButtonGroup()
        self.checkbox_group.setExclusive(False)
        self.connect(self.checkbox_group, QtCore.SIGNAL('buttonPressed(int)'),
                     self.on_checkbox_pressed)

        # TODO: come up with a better set of default colors?
        # TODO: Generate better initial colors than random ...
        # color = '#%06x' % np.random.randint(0xFFFFFF)
        self.pick_color_gen = itertools.cycle(['green', 'red', 'blue', 'magenta', 'cyan', 'purple'])

    def get_color(self, label):
        # type: (str) -> QtGui.QColor
        return self.colors[label]

    def add_row(self, label):
        # type: (str) -> None
        self.labels.append(label)

        self.radiobuttons[label] = QtGui.QRadioButton('')
        self.radio_group.addButton(self.radiobuttons[label],
                                   self.labels.index(label))

        self.checkboxes[label] = QtGui.QCheckBox('')
        self.checkbox_group.addButton(self.checkboxes[label],
                                      self.labels.index(label))

        self.textlabels[label] = QtGui.QLabel(label)

        self.colorbuttons[label] = QtGui.QPushButton('')
        self.colorbuttons[label].setFixedSize(20, 20)
        color = self.pick_color_gen.next()
        self.connect(self.colorbuttons[label], QtCore.SIGNAL('clicked()'),
                     lambda: self.on_color_button_clicked(label))
        self.colorbuttons[label].setStyleSheet('QPushButton {background-color: %s}' % (color))
        self.colors[label] = color

        self.row_hboxes[label] = QtGui.QHBoxLayout()
        self.row_hboxes[label].addWidget(self.radiobuttons[label])
        self.row_hboxes[label].addWidget(self.checkboxes[label])
        self.row_hboxes[label].addStretch(1)
        self.row_hboxes[label].addWidget(self.textlabels[label])
        self.row_hboxes[label].addWidget(self.colorbuttons[label])

        self.layout.addLayout(self.row_hboxes[label])

    def on_color_button_clicked(self, label):
        # type: (str) -> None
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.colorbuttons[label].setStyleSheet('QPushButton {background-color: %s}' % (color.name()))
            if self.color_cb is not None:
                self.color_cb(label, str(color.name()))

    def on_radio_button_pressed(self, button_id):
        # type: (int) -> None
        label = self.labels[button_id]
        if self.radio_cb is not None:
            self.radio_cb(label)

    def on_checkbox_pressed(self, button_id):
        # type: (int) -> None
        label = self.labels[button_id]
        # For some reason, this returns 1 when event causes box to be
        # unchecked, and 0 when it winds up checked.
        checked = self.checkboxes[label].isChecked()
        if self.check_cb is not None:
            self.check_cb(label, not checked)

    def activate_radio_checkbox(self):
        # type: () -> None
        '''
        This causes the checkbox of the currently-active radio button to be
        selected. It is needed for programatically showing the max when
        autopick is recalculated.
        '''
        for button_idx, button_id in enumerate(self.labels):
            if self.radiobuttons[button_id].isChecked():
                self.checkboxes[button_id].setCheckState(-1)
                self.check_cb(button_id, True)
