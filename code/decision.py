import numpy as np
from supporting_functions import wrap_angle_180
import time

flag_print = 0;

# Compute reference yaw when searching a rock
def compute_yawref(Rover):
    xlastrock = Rover.samples_pos_detected[Rover.samples_located-1,0]
    ylastrock = Rover.samples_pos_detected[Rover.samples_located-1,1]
    yawref = np.arctan2(ylastrock-Rover.pos[1],xlastrock-Rover.pos[0])*180/np.pi
    Rover.yawref = wrap_angle_180(yawref)
    return Rover

# Orientation controller
def control_yaw(Rover):
    
    ctrl_steer = Rover.Kp_yaw*wrap_angle_180(Rover.yawref - Rover.nyaw)
    Rover.steer = np.clip(ctrl_steer,-15,15)
    return Rover

# Velocity controller
def control_vel(Rover,refvel):
    error = (refvel - Rover.vel)
    ctrl_throttle = Rover.Kp_vel*error + Rover.Ki_vel *1/25* Rover.int_error_vel
    if abs(ctrl_throttle)<1:
        Rover.int_error_vel = Rover.int_error_vel + error

    Rover.throttle = np.clip(ctrl_throttle,-1,1)
    return Rover

# This function checks if the rover is stuck
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

# This function checks if the rover is in a loop
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
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    global flag_print
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!
    
    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        if Rover.flag_print:
            print(Rover.mode)
        # Check for Rover.mode status
        # Check for sticking and looping if not unsticking
        if Rover.mode != 'unsticking' and Rover.picking_up == 0:
            if Rover.sample_in_sight:
                Rover.mode = 'approaching'
            Rover = check_sticking(Rover)
            Rover = check_looping(Rover)
        # Check for Rover.mode status
        #print(Rover.mode)
        if Rover.mode == 'unsticking':
            # The rover is stuck
            Rover.throttle = 0
            Rover.brake = 0
            if abs(Rover.nyaw-Rover.yawref)<1:
                Rover.mode = 'forward'
            else:
                Rover = control_yaw(Rover)

        elif Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                Rover = control_vel(Rover, Rover.vel_fwd)
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                nav_angle = np.mean(Rover.nav_angles* 180/np.pi)
                Rover.steer = np.clip(nav_angle + Rover.deviation, -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
        elif Rover.mode == 'approaching':
            if Rover.near_sample == 0:
                Rover = control_vel(Rover, Rover.vel_apch)
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                if Rover.sample_in_sight:
                    Rover.time_approaching = 0
                    nav_angle = np.mean(Rover.rocks_angles* 180/np.pi)
                    Rover.steer = np.clip(nav_angle, -15, 15)
                    Rover.prev_steer = Rover.steer
                else:
                    Rover.steer = Rover.prev_steer
                # Stay in this mode at least one second
                total_time_approaching = 0
                if Rover.time_approaching == 0:
                    Rover.time_approaching = time.time()
                else:
                    total_time_approaching = time.time()-Rover.time_approaching
                if total_time_approaching>Rover.max_time_approaching:
                    Rover.time_approaching = 0 
                    Rover.mode = 'forward'
            else:
                Rover.brake = Rover.brake_set
                Rover.send_pickup = True
                if Rover.picking_up == 0:
                    Rover.mode = 'forward'
        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.time_stopped = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover