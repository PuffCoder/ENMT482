#constents



#controls
sensor_model_on = True
#sensor_model_on = True

motion_model_velocity = True#True
motion_model_odom = True#False
#motion_model_mode = "both"
motion_weighting = 1.0  #1.0 odometry,  0.0 velocity

motion_model_noise = False
motion_model_noise_alt = True

##motion model
motion_sigma_x = 0.01
motion_sigma_y = 0.01
motion_sigma_theta = 0.01

#sensor model

camera_to_robot = (0, 0, 0)#(0.1, 0.1, 0)

sigma_r = 0.2#0.5
sigma_theta = 0.8#1.0
