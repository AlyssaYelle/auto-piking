# Not sure where this file belongs ... it's certainly not deva-specific, but
# I also don't see huge use for it elsewhere, unlike my data-wrangling code.

# For development on melt, where we don't have PySide.
# However, PyQt is GPL'd, so we really shouldn't be using it. (PySide is LGPL)
# Before distributing/releasing this software, uncomment the PySide lines.
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
# import PySide.QtCore as QtCore
# import PySide.QtGui as QtGui
# matplotlib.rcParams['backend.qt4']='PySide'


def show_error_message_box(msg):
    # type: (str) -> None
    '''
    Pops up dialog box with input message and waits for user to hit 'ok'.
    '''
    msgbox = QtGui.QMessageBox()
    msgbox.setText(msg)
    msgbox.exec_()

def HLine():
    # type: () -> None
    '''
    Creates a horizontal line that can be added to a layout.
    '''
    line = QtGui.QFrame()
    line.setFrameShape(QtGui.QFrame.HLine)
    line.setFrameShadow(QtGui.QFrame.Sunken)
    return line

def VLine():
    # type: () -> None
    '''
    Creates a vertical line that can be added to a layout.
    '''
    line = QtGui.QFrame()
    line.setFrameShape(QtGui.QFrame.VLine)
    line.setFrameShadow(QtGui.QFrame.Sunken)
    return line
