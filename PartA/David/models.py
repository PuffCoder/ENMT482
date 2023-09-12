"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference

from constents import *
from transform import *


def fwd_odm(prev_pose, od_mov):
    """
    prev_pose is the previous pose (x, y, theta)
    od_mov is the odometry move (d, Phi1, Phi2)

    outputs new pose
    """

    x = prev_pose[0]
    y = prev_pose[1]
    theta = prev_pose[2]

    d = od_mov[0]
    Phi1 = od_mov[1]
    Phi2 = od_mov[2]


    newx = x + d * cos(theta + Phi1)
    newy = y + d * sin(theta + Phi1)
    newtheta = theta + Phi1 + Phi2

    return (newx, newy, newtheta)


def fwd_odm2(prev_pose, od_mov):
    """
    prev_pose is the previous pose (x, y, theta)
    od_mov is the odometry move (d, Phi1, Phi2)

    outputs global pose change
    """

    x = prev_pose[0]
    y = prev_pose[1]
    theta = prev_pose[2]

    d = od_mov[0]
    Phi1 = od_mov[1]
    Phi2 = od_mov[2]


    dx = d * cos(theta + Phi1)
    dy = d * sin(theta + Phi1)
    dtheta = Phi1 + Phi2

    return(dx, dy, dtheta)



def rev_odm(curr_pose, prev_pose):
    """
    curr_pose  is the current pose (x, y, theta)
    prev_pose  is the previous pose (x, y, theta)

    output is the odometry move (d, Phi1, Phi2)
    """

    dx = curr_pose[0] - prev_pose[0]
    dy = curr_pose[1] - prev_pose[1]
    prev_theta = prev_pose[2]
    curr_theta = curr_pose[2]


    angle = arctan2(dy, dx)

    Phi1 = angle - prev_theta

    Phi2 = curr_theta - angle

    d = sqrt(dx**2 + dy**2)

    return (d, Phi1, Phi2)





def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """
    
    M = particle_poses.shape[0]
    
    # TODO.  For each particle calculate its predicted pose plus some
    # additive error to represent the process noise.  With this demo
    # code, the particles move in the -y direction with some Gaussian
    # additive noise in the x direction.  Hint, to start with do not
    # add much noise.

    #time is in ns 1e-9
    dt = dt * 1e-9
    
    if dt ==0:
        return particle_poses

    for m in range(M):

        theta = particle_poses[m, 2]

        v = speed_command[0]
        omega = speed_command[1]
        
        if motion_model_velocity: #Velocity

            if omega == 0: #straight
                vel_dx = v * cos(theta) * dt
                vel_dy = v * sin(theta) * dt
                vel_dtheta = 0

            else:
                vel_dx = -v / omega * sin(theta) + v / omega * sin(theta + omega * dt)
                vel_dy = v / omega * cos(theta) - v / omega * cos(theta + omega * dt)
                vel_dtheta = omega * dt
            


        if motion_model_odom:
            odom_mov = rev_odm(odom_pose, odom_pose_prev)

            #particle_poses[m] = fwd_odm(particle_poses[m], odom_mov)

            #odom_dpose = fwd_odm2(particle_poses[m], odom_mov)
            (odom_dx, odom_dy, odom_dtheta) = fwd_odm2(particle_poses[m], odom_mov)




        #fusion
        w = motion_weighting
        dx = w * odom_dx + (1-w) * vel_dx
        dy = w * odom_dy + (1-w) * vel_dy
        dtheta = w * odom_dtheta + (1-w) * vel_dtheta
        
        

        
        
        #process noise
        if motion_model_noise:
            noise_x= np.random.normal(0, motion_sigma_x)
            noise_y= np.random.normal(0, motion_sigma_y)
            noise_theta= np.random.normal(0, motion_sigma_theta)
            
        #local noise
        if motion_model_noise_alt:
            localnoise_x = np.random.normal(0, motion_sigma_x)
            localnoise_y = np.random.normal(0, motion_sigma_y)

            noise_x = localnoise_x * cos(theta) - localnoise_y * sin(theta)
            noise_y = localnoise_y * sin(theta) + localnoise_y * cos(theta)
            noise_theta = np.random.normal(0, motion_sigma_theta)



        particle_poses[m, 0] += dx + noise_x
        particle_poses[m, 1] += dy + noise_y
        particle_poses[m, 2] = wraptopi(theta + dtheta + noise_theta)

    
    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """
    
    

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    
    # TODO.  For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the beacon location.


    if sensor_model_on:
        #becon conditioning
        #camera_to_robot = (0.1, 0.1, 0)
        becon_pose_robot = transform_pose(camera_to_robot, beacon_pose )

        #print(beacon_pose)
        #print(becon_pose_robot)
        
        #liekelyhood functions
        becon_range = np.sqrt((becon_pose_robot[0])**2 + (becon_pose_robot[1])**2)
        becon_angle = becon_pose_robot[2]

        #print(becon_range)
        #print(becon_angle * 180 / np.pi)
    
    
    for m in range(M):

        if sensor_model_on:
        
            x_b = beacon_loc[0]
            y_b = beacon_loc[1]
            x_p = particle_poses[m][0] #particle position in map frame
            y_p = particle_poses[m][1]
            theta_p = particle_poses[m][2]
            
            
            range_p2b = np.sqrt((x_b - x_p)**2 + (y_b - y_p)**2) #range from particle to becon
            b_angle_map = arctan2((y_b - y_p), (x_b-x_p))

            
            angle_p2b = angle_difference(theta_p, b_angle_map)

            rangeerror = gauss(becon_range - range_p2b, 0, sigma_r)
            angleerror = gauss(becon_angle - angle_p2b, 0, sigma_theta)
            
            particle_weights[m] = rangeerror * angleerror

            #print(rangeerror, angleerror)

        else:
            particle_weights[m] = 1

    return particle_weights



#particle_poses = np.array([[1, 1, np.pi],
#                  [2, 1, np.pi],
#                  [3, 1, np.pi]])
#beacon_pose = [2, -2, -1 *np.pi / 4]
#beacon_loc = [1, 3, 0]

particle_poses = np.array([[1, 1, np.pi / 4],
                  [2, 1, 0],
                  [3, 1, 0]])
beacon_pose = [0, 1, np.pi / 2]
beacon_loc = [2, 2, 0]

particle_weights = sensor_model(particle_poses, beacon_pose, beacon_loc)
print(particle_weights)
