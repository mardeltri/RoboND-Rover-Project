import numpy as np
import cv2
from supporting_functions import wrap_angle_180


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

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
    # Return the binary image
    obstacle_area = np.float32(color_unselect*mask)
    return obstacle_area

def rocks_thresh(img):
    lower_yellow = np.array([14,100,100])
    upper_yellow = np.array([34,255,255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    rocks_select = cv2.bitwise_and(img,img, mask= mask)
    
    rocks_select_bin = color_thresh(rocks_select, rgb_thresh=(5, 5, 5))
    
    return rocks_select_bin

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask

# Crop x and y pixel values to improve fidelity
def crop_xy(xpix, ypix, crop_value):
    ypix_crop = ypix[xpix<crop_value]
    xpix_crop = xpix[xpix<crop_value]
    return xpix_crop, ypix_crop

def samples_diff(Rover):
    samples_posx = Rover.samples_pos[0][:]
    samples_posy = Rover.samples_pos[1][:]
    samples_posx_detected = Rover.samples_pos_detected[:,0]
    mask = np.in1d(samples_posx, samples_posx_detected,invert=True)
    samples_posx_diff = samples_posx[mask]
    samples_posy_diff = samples_posy[mask]

    return samples_posx_diff, samples_posy_diff
def update_rocks(Rover):
    # Check whether any rock detections are present in worldmap
    rock_world_pos = Rover.worldmap[:,:,1].nonzero()
    # If there are, we'll step through the known sample positions
    # to confirm whether detections are real

    samples_posx_diff, samples_posy_diff = samples_diff(Rover)
    samples_located = Rover.samples_located;
    if rock_world_pos[0].any():
        for idx in range(len(samples_posx_diff)):
            test_rock_x = samples_posx_diff[idx]
            test_rock_y = samples_posy_diff[idx]
            rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1])**2 + \
                                (test_rock_y - rock_world_pos[0])**2)
            # if rocks were detected within 3 meters of known sample positions
            # consider it a success and plot the location of the known
            # sample on the map
            if np.min(rock_sample_dists) < 3:
                Rover.samples_pos_detected[samples_located,:] = [test_rock_x,test_rock_y]
                Rover.samples_located += 1
    
    return Rover

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                      [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_area = obstacle_thresh(warped, mask)
    rocks_area = rocks_thresh(warped)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obs_area*255
    Rover.vision_image[:,:,1] = rocks_area*255
    Rover.vision_image[:,:,2] = threshed*255
    # 5) Convert map image pixel values to rover-centric coords
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw

    xpix, ypix = rover_coords(threshed)
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    Rover = update_rocks(Rover)
    if abs(Rover.npitch)<2 and abs(Rover.nroll)<2:
        xpix_obs, ypix_obs = rover_coords(obs_area)
        xpix_rocks, ypix_rocks = rover_coords(rocks_area)
        # Crop values
        xpix_crop, ypix_crop = crop_xy(xpix, ypix, 20)
        xpix_obs_crop, ypix_obs_crop = crop_xy(xpix_obs, ypix_obs, 20)
        xpix_rocks_crop, ypix_rocks_crop = crop_xy(xpix_rocks, ypix_rocks, 30)
        # 6) Convert rover-centric pixel values to world coordinates
        world_size = 200
        scale = 10
        x_pix_world, y_pix_world = pix_to_world(xpix_crop, ypix_crop, xpos, ypos, yaw, world_size, scale)
        x_pix_obs_world, y_pix_obs_world = pix_to_world(xpix_obs_crop, ypix_obs_crop, xpos, ypos, yaw, world_size, scale)
        x_pix_rck_world, y_pix_rck_world = pix_to_world(xpix_rocks_crop, ypix_rocks_crop, xpos, ypos, yaw, world_size, scale)
        # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        # Worldmap is updadted in case roll and pitch angles are close to zero.
    
        Rover.worldmap[y_pix_obs_world,x_pix_obs_world, 0] += 1
        Rover.worldmap[y_pix_rck_world,x_pix_rck_world, 1] += 1
        Rover.worldmap[y_pix_world,x_pix_world, 2] += 1

    return Rover