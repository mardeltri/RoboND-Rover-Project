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
[image9]: ./misc/rover_60p_85f.PNG 
[image10]: ./misc/rover_70p_87f_337s.PNG 
[image11]: ./misc/rover_94p_83f_950s.PNG

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
![Perspective transform][image1]

With recorded data:
![Perspective transform with recorded data][image2]

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
![Navigable terrain][image3]
With recorded data:
![Navigable terrain with recorded data][image4]

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
![Sample rock detection][image5]
With recorded data:
![Sample rock detection with recorded data][image6]

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
![Process image][image7]

Below a gif annimation which shows the video output has been included.

![Process image GIF][image8] 
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
+ Control laws. Two control laws have been implemented in order to command velocity and orientation.
	+ **Velocity control** `control_vel(Rover,refvel)` Defining a velocity control helps to achieve faster movements given that the rover will try to reach an specific velocity with the whole throttle range. 
	This controller is a PI with a saturation in the integral term when the control signal (throttle) is saturated (-1,1). Negative throttle values have been considered to slow
	down the rover when needed.
	```
	def control_vel(Rover,refvel):
		error = (refvel - Rover.vel)
		ctrl_throttle = Rover.Kp_vel*error + Rover.Ki_vel *1/25* Rover.int_error_vel
		if abs(ctrl_throttle)<1:
			Rover.int_error_vel = Rover.int_error_vel + error

		Rover.throttle = np.clip(ctrl_throttle,-1,1)
		
		return Rover
	```
	+ **Orientation control** `control_yaw(Rover)` This orientation controller is used when the rover is stuck. As it will be explained later, when the rover is stuck, a new orientation will be
	set as reference and, in that position, the rover will try to go forward. This controller consists in a simple proportional controller.
	```
	def control_yaw(Rover):
		ctrl_steer = Rover.Kp_yaw*wrap_angle_180(Rover.yawref - Rover.nyaw)
		Rover.steer = np.clip(ctrl_steer,-15,15)
		return Rover
	```
+ Checking if the rover is stuck or in a looping.
	+ **Checking sticking** `check_sticking(Rover)`. This function checks the rover velocity during a certain period of time. If it is lower than the threshold, the rover is considered to be
	stuck and the unsticking mode is set. In addition, this function computes the orientation reference (`yawref`) which later will use the orientation controller. As it can be seen, the rover 
	always turns in the same direction and it does not depend on the detected navigable terrain. This last case was programmedbut it did not provide good results given that, 
	when the rover is stuck in the rocks, the navigable direction does not correspond neccesarily with the direction to escape from that place.

	```
	def check_sticking(Rover):
		# Check for collision
		total_time_stopped = 0
		if abs(Rover.vel)<0.2 and (Rover.time_stopped == 0):
			Rover.time_stopped = time.time()
		elif abs(Rover.vel)<0.2 and (Rover.time_stopped != 0):
			total_time_stopped = time.time()-Rover.time_stopped
		else:
			Rover.time_stopped = 0
		if total_time_stopped>Rover.max_time_stopped:
			print('Rover stuck')
			Rover.time_stopped = 0 
			Rover.yawref = wrap_angle_180(Rover.nyaw - Rover.unstick_angle) 
			Rover.mode = 'unsticking'
		return Rover
	```
	
	+ **Checking looping** `check_looping(Rover)`. The structure of this function is quite similar to the previous one, however, in this case, the yaw reference is computed
	depending on the way that the rover is turning.
	```
	def check_looping(Rover):
		#Checking for looping
		total_time_looping = 0
		flag_angle = abs(abs(Rover.steer)-Rover.stuck_steer_angle)<0.5
		flag_speed = abs(Rover.vel)>0.5
		if flag_angle and flag_speed and (Rover.time_looping == 0):
			Rover.time_looping = time.time()
		elif flag_angle and flag_speed and (Rover.time_looping != 0):
			total_time_looping = time.time()-Rover.time_looping
		else:
			Rover.time_looping = 0
		if total_time_looping>Rover.max_time_looping:
			print('Rover in a loop')
			Rover.time_looping = 0
			if Rover.steer>0:
				Rover.yawref = wrap_angle_180(Rover.nyaw - Rover.unstick_angle)
			else:
				Rover.yawref = wrap_angle_180(Rover.nyaw + Rover.unstick_angle)
			Rover.mode = 'unsticking'
		return Rover
	```
##### Forward mode
This mode moves the rover forward when there is enough navigable terrain. In this case, the rover tries to reach the forward velocity defined in the `Rover.vel_fwd` variable.
If there is not enough navigable terrain, the rover mode `stop` is set. It must be highlighted that the rover steer angle has been modified in order to deviate lightly the rover
towards the left wall. To do so, an offset has been added to the navigation angle.

##### Stop mode
This mode turns the rover if there is not enough navigable terrain and speed it up in case that there is. This mode has not been modified.

##### Approaching mode
This mode is set when the rover detects	a sample rock. Here, an approaching velocity is imposed until the rover can take the sample rock. In this case, the steer angle is given
by the mean rocks angle position computed (Rover.rocks_angles) by the perception step. When the sample rock is picked up the forward mode is set.

##### Unsticking mode
This mode basically waits the rover to reach the orientation reference value imposed by `check_sticking`. To do so, this mode uses the function `control_yaw`. When the rover
reaches the reference angle, the forward mode is set.


#### 3. Autonomous mode, results and improvements

In this section, first, the rover state machine will be discussed. Later, results will be showed and finally some improvements will be suggested.

##### Rover state machine

The rover starts in forward mode and go through the scenario close to the left wall. When it detects a sample rock the approaching mode is set, so the rover slows down and turns
towards the sample rock until it reaches it. In case it get stuck, unsticking mode is set and the forward mode is activated again. In satisfactory case, the rover picks up the
sample rock and continues mapping in forward mode. In case the rover is stuck, it turns right 15 degrees and tries to go forward again. It repeats this procedure until it is unstuck.

##### Results
As can be seen the rover is able to pick up sample rocks and to map more than 60% of the terrain. During simulations, the rover has been able to get unstuck with the implemented 
procedure. Sometimes, it doesn't pick up the sample rock the first time it sees it, however it usually picks it up when it goes back.

![Results_Rover_70p_87f_337s][image10]
![Results_Rover_94p_83f_950s][image11]

##### Improvements
I would have liked to improve the way the rover detects the sample rocks given that, as mentioned previously, sometimes it detects them but it is not able to pick the up the first time.
I think that two methods could be integrated: the one it is used here, and another one, that in case it does not pick the rock the first time, it makes the rover turn until it sees 
the sample rock, and, when oriented, try again to pick it up. Indeed, the first method that I implemented to pick up sample rocks consisted in: stopping the rover, turning it towards the sample
rock and going foward until it reached the sample rock. However, this method failed because sometimes it did not reach the sample. 

Furthermore, I would have liked to implement the last step to achieve the starting point with the sample rocks. I think that is will be possible by commanding a yaw angle which makes 
the rover achieves the starting point. This angle should be warped with the possible navigable angles.



