background_zorder = 0 # For all the background images

gl_zorder = 0.5 # want grounding lines on top of bg images
lake_zorder = 0.5 # same for lakes ...

# For the PSTs and GLAS lines
line_zorder = 1
line_pick_radius = 5

utig_line_color = 'k'
cresis_line_color = 'dimgrey'
external_line_color = 'darkgrey'
pst_linewidth = 1
glas_line_color = 'grey'
glas_linestyle = 'dashed'
glas_linewidth = 1
flight_line_color = 'black'
flight_linestyle = 'solid'
flight_linewidth = 1
# Will be plotted on top of each other.
card0_linewidth = 2
card1_linewidth = 2
# tof0/tof1 need to be below/above black fligth line.
card0_zorder = 0.5
card1_zorder = 1.5

# For the red context rectangles
rect_zorder = 1
rect_color = 'r'

# Used to indicate PSTs that have radarFigures open.
shown_zorder = 3
shown_linewidth = 1

# For showing the currently-selected line in deva plot
selected_zorder = 2
selected_color = 'b'
selected_linewidth = 1

# portion of the line displayed in {radar,GLAS}figure
highlighted_zorder = 4
highlighted_color = 'r'
highlighted_linewidth = 1
