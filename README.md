## Project: Search and Sample Return

### Project objective

The main purpose of this project is to develop a program to move a Rover autonomously. First, we will focus on detecting navigable and non-navigable terrain. Later, we will
program the algorithms to command the rover in order to map the navigable terrain. In addition, given that there are rock samples in the scenario, we will write a function
to detect these rocks and we will define an autonomous mode to collect them.

[//]: # (Image References)

[image1]: ./misc/perspective_transform.png
[image2]: ./misc/perspective_transform_mb.png
[image3]: ./misc/navigable_terrain.png 
[image4]: ./misc/navigable_terrain_mb.png 
[image5]: ./misc/rock_sample_detection.png 
[image6]: ./misc/rock_sample_detection_mb.png
[image7]: ./misc/process_image.png 
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
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
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

Rock samples are detected applying a filter to find yellow colors. Function `color_thresh` receives an image and changes the color-space to HSV to apply an upper and lower threshold.
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

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
Function `process_image()` carries out the procedures to map the terrain, this is,
 detect navigable, non-navigable terrain and rock samples in rover centric coordinates
 and transform them to worldmap coordinates. The following steps are perform:
 * Apply the perspective transform
 * Apply color threshold to detect navigable and non-navigable terrain and rock samples
 * Convert thresholded image pixel values to rover-centric coords
 * Crop images in order to increase map fidelity
 * Convert rover-centric values to world coords considering the rover position and orientation
 * Compute mean angle from navigable pixels.
 The following image shows the steps carry out.
![alt text][image7]

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]

