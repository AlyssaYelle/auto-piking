# Automatically Mapping the Antarctic Ice Sheet

Glaciologists can learn a lot from ice-penetrating radar data. After analyzing the radar image via "picking" the surface and bottom of the ice sheet, we can then retrieve information such the thickness of the ice and the reflectivity/specularity of the bottom of the ice. Knowing the reflectivity and specularity is essential for detecting subglacial water. Currently, picking the image is a time consuming manual task. My goal is to automate this process. 


## Edge Detection

My first thought was that edge detection could be useful, at least for 'ideal' radar images (by ideal I mean that the image is free of artifacts caused by things like crevasses or water at the surface of the ice). See below for an example of an ideal radar image. Note that the surface and bottom of the ice are brighter, or have a higher pixel intensity, than their relative vertical surroundings.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/example_bedunpicked.png "Example of an ideal ice-penetrating radar image")

Scikit-image's feature module conveniently has a built in canny edge detection function. Canny edge detection works by first applying a Gaussian filter with some user-defined sigma to smooth the image. It then finds the intensity gradients of the image and "thins" potential edges and then suppresses weak edges using hysteresis. It outputs a 2-D binary array edge map. See below for canny edge maps of varying sigmas applied over the original radar image.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/edges_im_overlay.png "Trying out Canny edge detection with various sigmas")

At sigma = 10, the canny filter captures the upper and lower boundaries of the surface and bottom of the ice fairly well:

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/bed_boundary.png "At sigma = 10 the Canny filter very roughly identifies the ice-air interface and ice-bedrock interface")

Below is the radar image picked by an expert (blue) and with upper and lower boundary of the bottom of the ice identified by the canny filter (pink). The canny filter successfully finds the bottom of the ice, albeit the result is more noisy than I prefer.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/example_bedpicked_zoom.png "Human picker vs. canny filter")

Let me stress that while the results seen above are promising, this method fails when the radar image is not basically visually perfect. I ran it on another transect from the same region of Antartica that yielded a slightly less clear radar image, and the results were... not good. See below.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/edges_bad.png "Canny filter unable to reliably capture bed")


## Mapping the Ice Sheet Surface

Mapping the surface of the ice is a much more simple problem than mapping the bottom. Barring the presence of mountains, the surface generally presents as an extremely bright, fairly straight line. A rather naive way to go about mapping the surface is to simply read the radar image as an array of pixel intensities and find the maximum pixel for each column. This actually produces pretty good results! See below for an expert picked surface (red) and the auto picked surface (yellow).

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/example_srf_expertpicked.png "Expert picked surface")
![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/srf_autopicked.png "Auto picked surface")


## Understanding Radar Images via Simulation

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim1.png "Simulation 1")![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim2.png "Simulation 2")![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim3.png "Simulation 3")![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim4.png "Simulation 4")![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim5.png "Simulation 5")![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/models/figs/sim6.png "Simulation 6")










