import matplotlib
import matplotlib.widgets as mpw
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT

# SDK attempted to install typing on melt, but it failed.
# This is fully optional, and LEL can run it on her machine.
try:
    import typing
    from typing import Any, Tuple
except:
    pass

def get_ax_shape(fig, ax):
    # type: (Any, Any) -> Tuple[int, int]
    '''
    returns axis width in pixels; used for being a bit clever about how much
    of the image we draw.
    '''
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return fig.dpi*bbox.width, fig.dpi*bbox.height


class UnzoomableAxes(matplotlib.axes.Axes):
    name = "unzoomable"
    def can_pan(self):
        # type: () -> bool
        return False
    def can_zoom(self):
        # type: () -> bool
        return False
matplotlib.projections.register_projection(UnzoomableAxes)


class NavigationToolbar(NavigationToolbar2QT):
    '''
    Toolbar that only displays the Pan, Zoom and Save Icons.
    (home/fwd/back don't work correctly here)
    '''
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ['Pan', 'Zoom', 'Save']]

    def __init__(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        super(NavigationToolbar, self).__init__(*args, **kwargs)
        # get rid of the one with the green checkbox
        self.layout().takeAt(3)


class SaveToolbar(NavigationToolbar2QT):
    ''' Toolbar that only displays the Save Icon. '''
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ['Save']]

    def __init__(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        super(SaveToolbar, self).__init__(*args, **kwargs)
        # get rid of the one with the green checkbox
        self.layout().takeAt(1)


class XevasHorizSelector:
    def __init__(self, ax, min_data, max_data, update_cb=None, margin_frac=0):
        # type: (Any, float, float, Any, float) -> None
        '''
        * ax - axes on which to add the selector.
        * {min,max}_data - in data units, min/max of full plot
        * margin - what fraction of the data range each margin consumes.
        * update_cb - will be called with (xmin, xmax) in axis units when the
                      selector is updated
        '''
        self.update_cb = update_cb
        self.xevas_horiz_ax = ax
        self.xevas_horiz_ax.axis('off')

        self.min_data = min_data
        self.max_data = max_data

        margin_width = (max_data - min_data) * margin_frac
        self.xevas_horiz_ax.set_xlim([min_data-margin_width,
                                      max_data+margin_width])

        self.xevas_horiz_margin = self.xevas_horiz_ax.axvspan(
            min_data-margin_width, max_data+margin_width,
            facecolor='grey', edgecolor='none')
        self.xevas_horiz_bg = self.xevas_horiz_ax.axvspan(
            min_data, max_data, facecolor='darkgrey', edgecolor='none')
        self.xevas_horiz_fg = self.xevas_horiz_ax.axvspan(
            min_data, max_data, facecolor='k', alpha=0.5, edgecolor='none')

        self.xevas_horiz_ss = mpw.SpanSelector(
            self.xevas_horiz_ax, self.horiz_span_cb, 'horizontal')

    def horiz_span_cb(self, xmin, xmax):
        # type: (int, int) -> None
        if xmin == xmax:
            print "BAD SPAN - ZERO LENGTH"
            return
        new_min = max(self.min_data, xmin)
        new_max = min(self.max_data, xmax)
        self.xevas_horiz_fg.remove()
        self.xevas_horiz_fg = self.xevas_horiz_ax.axvspan(
            new_min, new_max, facecolor='k', alpha=0.5, edgecolor='none')
        if self.update_cb is not None:
            self.update_cb(new_min, new_max)

    def update_selection(self, xlim):
        # type: (Tuple[int, int]) -> None
        new_min = max(self.min_data, xlim[0])
        new_max = min(self.max_data, xlim[1])
        self.xevas_horiz_fg.remove()
        self.xevas_horiz_fg = self.xevas_horiz_ax.axvspan(
            new_min, new_max, facecolor='k', alpha=0.5, edgecolor='none')


class XevasVertSelector:
    def __init__(self, ax, min_data, max_data, update_cb=None, margin_frac=0):
        # type: (Any, float, float, Any, float) -> None
        '''
        * ax - axes on which to add the selector.
        * {min,max}_data - in axis units, min/max of full plot
        * update_cb - will be called with (xmin, xmax) in axis units when the
                      selector is updated
        '''
        self.update_cb = update_cb
        self.xevas_vert_ax = ax
        self.xevas_vert_ax.axis('off')

        self.min_data = min_data
        self.max_data = max_data

        margin_width = (max_data - min_data) * margin_frac

        self.xevas_vert_ax.set_ylim([min_data-margin_width,
                                     max_data+margin_width])
        self.xevas_vert_margin = self.xevas_vert_ax.axhspan(
            min_data-margin_width, max_data+margin_width,
            facecolor='grey', edgecolor='none')
        self.xevas_vert_bg = self.xevas_vert_ax.axhspan(
            min_data, max_data, facecolor='darkgrey', edgecolor='none')
        # This is done with alpha s.t. the selector can show.
        # The others couldn't be, since I wanted the outer one darker
        # than the inner one, and alphas add ...
        self.xevas_vert_fg = self.xevas_vert_ax.axhspan(
            min_data, max_data, facecolor='k', alpha=0.5, edgecolor='none')

        self.xevas_vert_ss = mpw.SpanSelector(
            self.xevas_vert_ax, self.vert_span_cb, 'vertical')

    def vert_span_cb(self, ymin, ymax):
        # type: (int, int) -> None
        if ymin == ymax:
            print "BAD SPAN - ZERO LENGTH"
            return
        new_min = max(self.min_data, ymin)
        new_max = min(self.max_data, ymax)
        self.xevas_vert_fg.remove()
        self.xevas_vert_fg = self.xevas_vert_ax.axhspan(
            new_min, new_max, facecolor='k', alpha=0.5, edgecolor='none')
        if self.update_cb is not None:
            self.update_cb(new_min, new_max)

    def update_selection(self, ylim):
        # type: (Tuple[int, int]) -> None
        new_min = max(self.min_data, ylim[0])
        new_max = min(self.max_data, ylim[1])

        self.xevas_vert_fg.remove()
        self.xevas_vert_fg = self.xevas_vert_ax.axhspan(
            new_min, new_max, facecolor='k', alpha=0.5, edgecolor='none')
