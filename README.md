# Automatically mapping the Antarctic Ice Sheet

Glaciologists can learn a lot from ice-penetrating radar data. After analyzing the radar image via "picking" the surface and bottom of the ice sheet, we can then retrieve information such the thickness of the ice and the reflectivity/specularity of the bottom of the ice. Knowing the reflectivity and specularity is essential for detecting subglacial water. Currently, picking the image is a time consuming manual task. My goal is to automate this process. 


## Edge Detection

My first thought was that edge detection could be useful, at least for 'ideal' radar images (by ideal I mean that the image is free of artifacts caused by things like crevasses or water at the surface of the ice). See below for an example of an ideal radar image. Note that the surface and bottom of the ice are brighter, or have a higher pixel intensity, than their relative vertical surroundings.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/bed_example.png "Example of an ideal ice-penetrating radar image")

Skikit-image's feature module conveniently has a built in canny edge detection function. Canny edge detection works by first applying a Gaussian filter with some user-defined sigma to smooth the image. It then finds the intensity gradients of the image and "thins" potential edges and then suppresses weak edges using hysteresis. It outputs a 2-D binarry array edge map. See below for canny edge maps of varying sigmas applied over the original radar image.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/edges_im_overlay.png "Trying out Canny edge detection with various sigmas")

At sigma = 10, the canny filter captures the upper and lower boundaries of the surface and bottom of the ice fairly well:

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/bed_boundary.png "At sigma = 10 the Canny filter very roughly identifies the ice-air interface and ice-bedrock interface")

Below is the radar image picked by an expert (blue) and with upper and lower boundary of the bottom of the ice identified by the canny filter (pink). The canny filter successfully finds the bottom of the ice, albeit the result is more noisy than I prefer.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/example_bedpicked_zoom.png "Human picker vs. canny filter")

I don't think that edge detection is the ideal solution to my problem since most radar images are not ideal like the one provided here as an example, but given that the results are pretty good with this quick and dirty approach, I think a better solution is within my reach.






