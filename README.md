## Project: Search and Sample Return

### Project objective

The main purpose of this project is to develop a program to move a Rover autonomously. First, we will focus on detecting navigable and non-navigable terrain. Later, we will
program the algorithms to command the rover in order to map the navigable terrain. In addition, given that there are  sample rocks in the scenario, we will write a function
to detect these rocks and we will define an autonomous mode to collect them.

[//]: # (Image References)

[image1]: ./misc/perspective_transform.png
[image2]: ./misc/perspective_transform_mb.png
[image3]: ./misc/navigable_terrain.png 
[image4]: ./misc/navigable_terrain_mb.png 
[image5]: ./misc/rock_sample_detection.png 
[image6]: ./misc/rock_sample_detection_mb.png
[image7]: ./misc/process_image.PNG 
[image8]: ./misc/process_image.gif 

[image9]: ./misc/navigable_terrain.png 
[image10]: ./misc/navigable_terrain.png 
[image11]: ./misc/navigable_terrain.png 
[image12]: ./misc/navigable_terrain.png 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and sample rocks.
In this section, how to detect navigable, non-navigable terrain and rock samples will be addressed. 

The first step is to apply the perspective transform in order to have a top-down view from the scenario. The following function carries out this task. A new output variable, mask, was
added in order to deal with images wich have been transformed.

```
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask
```

With provided data:
![alt text][image1]

With recorded data:
![alt text][image2]

With those images we can apply a filter to detect navigable and non-navigable terrain. Threshold of RGB > 160 
is found to perform well in terms of determing navigable terrain. A new function has been defined to detect non-navigable terrain. The method used here is similar to that in 
`color_thresh` but zero values are set when the pixel is above all three threshold values. In addition, the mask previously mentioned must be applied given that the perspective
view does not extend to the whole image.
```
def obstacle_thresh(img, mask, rgb_thresh=(160, 160, 160)):
    # Create an array of ones same xy size as img, but single channel
    color_unselect = np.ones_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "False"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of ones with the boolean array and set to 0
    color_unselect[above_thresh] = 0
    # Apply perspective mask
    obstacle_area = np.float32(color_unselect*mask)
	# Return the binary image
    return obstacle_area
```
With provided data:
![alt text][image3]
With recorded data:
![alt text][image4]

Sample rocks are detected applying a filter to find yellow colors. Function `color_thresh` receives an image and changes the color-space to HSV to apply an upper and lower threshold.
```
def rocks_thresh(img):
    lower_yellow = np.array([14,100,100])
    upper_yellow = np.array([34,255,255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    rocks_select = cv2.bitwise_and(img,img, mask= mask)
    
    rocks_select_bin = color_thresh(rocks_select, rgb_thresh=(5, 5, 5))
    
    return rocks_select_bin
```
With provided data:
![alt text][image5]
With recorded data:
![alt text][image6]

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and sample rocks into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
Function `process_image()` carries out the procedures to map the terrain, this is,
 detect navigable, non-navigable terrain and sample rocks in rover centric coordinates
 and transform them to worldmap coordinates. The following steps are perform:
 * Apply the perspective transform
 * Apply color threshold to detect navigable and non-navigable terrain and  sample rocks
 * Convert thresholded image pixel values to rover-centric coords
 * Crop images in order to increase map fidelity. This action is carried out by the function `crop_xy(xpix, ypix, crop_value)` 
 which simply crops the picture with the pixels closer to the rover.
 ```
 def crop_xy(xpix, ypix, crop_value):
    ypix_crop = ypix[xpix<crop_value]
    xpix_crop = xpix[xpix<crop_value]
    return xpix_crop, ypix_crop
 ```
 * Convert rover-centric values to world coords considering the rover position and orientation
 * Compute mean angle from navigable pixels.
 
 The following image shows the steps carry out.
![alt text][image7]

Below a gif annimation which shows the video output has been included.

![alt text][image8] 
### Autonomous Navigation and Mapping
In this section, how to achieve the autonomous navigation will be discussed.

#### 1. Perception step
This perception step is the first point related with autonomous navigation. Here the camera image is received and processed. The content is very similar to the previously
mentioned `process_image()` function, thus, in this point the two main differences will be addressed.

* Check if there is any sample rock detected in the current camera image. This checking will be saved in the variable `Rover.sample_in_sight` which will be useful to define
the rover mode. It worths mention that this checking is carried out in the orginal picture (non-cropped). Thus, the cropped one is used for mapping and the original is used
 for detecting.
* Before computing the mapping images, it is check if the roll and pich angles are below a threshold, in this case 2 degrees. Thus, images taken with these angles are ruled
out to improve map fidelity. Given that the angle is between 0 and 360 degrees, a function named `wrap_angle_180(angle)` has been defined in `supporting_functions.py` to wrap
the angle between -180 and 180 degrees. Thus, we only have to check if the absolute wraped angle (`npich` or `nroll`) is below 2 degrees.

#### 2. Decision step
In this function it is defined what to do depending on the images taken by the rover. Four different modes will be considered: forward, stop, approaching and unsticking. In addition,
several functions have been developed in order to control the rover and to check if it is stuck or in a looping (continuosly turning). 

##### Functions
First, functions will be addressed in order to understand the modes inputs and outputs later.
* Control laws. Two control laws have been implemented in order to command velocity and orientation. 
**Defining a velocity control helps to achieve faster movements given that the rover
will try to reach an specific velocity with the whole throttle range. 
##### Forward mode
In this 


#### 3. Autonomous mode, results and improvements

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]

