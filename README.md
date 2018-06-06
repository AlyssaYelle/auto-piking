# Automatically mapping the Antarctic Ice Sheet

Glaciologists can learn a lot from ice-penetrating radar data. After analyzing the radar image via "picking" the surface and bottom of the ice sheet, we can then retrieve information such the thickness of the ice and the reflectivity/specularity of the bottom of the ice. Knowing the reflectivity and specularity is essential for detecting subglacial water. Currently, picking the image is a time consuming manual task. My goal is to automate this process. 


## Edge Detection

My first thought was that edge detection could be useful, at least for 'ideal' radar images (by ideal I mean that the image is free of artifacts caused by things like crevasses or water at the surface of the ice). See below for an example of an ideal radar image. Note that the surface and bottom of the ice are brighter, or have a higher pixel intensity, than their relative vertical surroundings.

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/bed_example.png "Example of an ideal ice-penetrating radar image")
Example of an ideal ice-penetrating radar image



![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/edges.png "Trying out Canny edge detection with various sigmas")

![alt text](https://github.com/AlyssaYelle/auto-piking/blob/master/imgproc/example_imgs/bed_boundary.png "At sigma = 10 the Canny filter very roughly identifies the ice-air interface and ice-bedrock interface")




