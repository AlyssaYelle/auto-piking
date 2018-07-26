# Automatically Mapping the Antarctic Ice Sheet

Glaciologists can learn a lot from ice-penetrating radar data. After analyzing the radar image via "picking" the surface and bottom of the ice sheet, we can then retrieve information such the thickness of the ice and the reflectivity/specularity of the bottom of the ice. Knowing the reflectivity and specularity is essential for detecting subglacial water. Currently, picking the image is a time consuming manual task. My goal is to automate this process. 

If you aren't familiar with ice radar and would like a quick introduction, please feel free to check out [this post](https://alyssayelle.github.io/2018/06/19/into-to-ice-radar.html) on my blog.


## Edge Detection

[Blog post on attempts to segment radar transects using a canny filter](https://alyssayelle.github.io/2018/07/26/ice2.html)


## Mapping the Ice Sheet Surface

Mapping the surface of the ice is a much more simple problem than mapping the bottom. Barring the presence of mountains, the surface generally presents as an extremely bright, fairly straight line. A rather naive way to go about mapping the surface is to simply read the radar image as an array of pixel intensities and find the maximum pixel for each column. This actually produces pretty good results! See below for an expert picked surface (red) and the auto picked surface (yellow).

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/example_srf_expertpicked.png "Expert picked surface")
![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/srf_autopicked.png "Auto picked surface")


## Generic Simulation of Radar Images

Based on the results from canny edge detection and my surface map, I think I will need to create my own model of a radar transect in order to get the results I want.

[Here](https://github.com/AlyssaYelle/auto-piking/blob/master/models/radargram_sim.py) is some extremely hack-y code used to simulate a radar transect. For this simulation I made a few assumptions.
- Between horizontal distance *d* and *d+1*, the surface or bed can jump up or down within some range of pixels with some probability *p*.
- The surface and bed lines each live within some vertical range of pixels
- The surface line can never be lower than the bed line
I played with the actual numbers until I got some simulations that look like what I might actually see in a real radar gram (I told you, this simulation is *really* hack-y). 

See two examples below.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim2.png "Simulation representing shallow ice")![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim5.png "Simulation representing deep ice")

I think the next step will be to study some actual expert-picked surface and bed lines to see what the actual distribution of vertical pixel jumps looks like, and build a model based off that.










