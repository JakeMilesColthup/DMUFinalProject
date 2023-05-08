from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range

from gym.spaces import Box, Discrete
import numpy as np
import math
from scipy.spatial.transform import Rotation
import cv2

################### Working Environment 5/4/2023 ###################
class SimpleVehicleSupervisor(RobotSupervisorEnv):
    """
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Lidar Ray 0                  0               4
        .   .
        .   .
        .   .
        59	Lidar Ray 59                 0               4
        60	Binary Pixel 0               0               1
        .   .
        .   .
        .   .
        1031 Binary Pixel 1031           0               1

    Actions:
        Type: Discrete(1)
        20 Discrete Actions

        Note: The first 10 discrete actions include all the steering angles and half speed. The last 10 discrete actions include all the steering angles and full speed
    """

    def __init__(self, loop=True, num_lidar_rays_in_obs = 60, num_discrete_turn_angles = 10, num_checkpoints = 5, num_wp_per_side = 100, steps_per_episode = 1000000000):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """

        super().__init__()

        #### Robot initialization ####
        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods

        # extract fields
        self.trackFront = 0.254
        self.wheelbase = 0.2921
        # self.trackFront = self.robot.getField("trackFront")
        # self.wheelbase = self.robot.getField("wheelbase")
        # Initialize Sensors and Motors
        self.camera = self.getDevice('camera')
        self.lidar = self.getDevice('LDS-01')
        self.steer_left_motor = self.getDevice('left_steer')
        self.steer_right_motor = self.getDevice('right_steer')
        l_motor = self.getDevice('rwd_motor_left')
        r_motor = self.getDevice('rwd_motor_right')
        self.motors = [l_motor,r_motor]
        # motor setup 
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)

        # enable
        self.lidar.enable(self.timestep) # 100ms LIDAR Readings
        self.lidar.enablePointCloud()
        self.camera.enable(self.timestep)

        # Thresholds for red color detection in the HSV color space
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([7, 255, 255 ])

        #### Observation Variables ####
        max_lidar_ray_indices = 60
        self.num_lidar_rays_in_obs = min(num_lidar_rays_in_obs, max_lidar_ray_indices) # max is 60
        self.lidar_obs_indices = (np.rint(np.linspace(0, max_lidar_ray_indices - 1, num=self.num_lidar_rays_in_obs))).astype(int)
        # print(f"Lidar Indices: {self.lidar_obs_indices}")

        #### Done Variables ####
        self.loop_episode_finish = False

        #### Episodes ####
        self.steps_per_episode = steps_per_episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

        #### Gym Environment ####
        # observation space lists
        pixel_scaling_factor = 3.5
        self.obs_pixel_width = int(self.camera.getWidth()/pixel_scaling_factor)
        self.obs_pixel_height = int(self.camera.getHeight()/pixel_scaling_factor)
        # print(f"Reduced X: {self.obs_pixel_width}, Reduced Y: {self.obs_pixel_height}")
        obs_space_low_list = [0 for _ in range(self.num_lidar_rays_in_obs)] + [0 for _ in range(self.obs_pixel_width*self.obs_pixel_height)]
        obs_space_high_list = [4 for _ in range(self.num_lidar_rays_in_obs)] + [1 for _ in range(self.obs_pixel_width*self.obs_pixel_height)]
        # Set up gym spaces
        self.observation_space = Box(low=np.array(obs_space_low_list),
                                     high=np.array(obs_space_high_list),
                                     dtype=np.float64)
        # self.action_space = Box(low=np.array([10, -1]), high=np.array([50,1]), dtype=np.float64)
        self.num_discrete_speeds = 2 # remeber to change the correct if statements in the apply actions
        self.num_discrete_turn_angles = num_discrete_turn_angles # remeber to change the correct if statements in the apply actions
        self.action_space = Discrete(self.num_discrete_speeds*self.num_discrete_turn_angles)

    def reset_reward_variables(self):
        """
        Reset the variables that are specific to each episode
        """
        self.episode_score = 0
        self.previous_waypoint_index = 0
        self.current_checkpoint_to_complete = 1
        self.loop_episode_finish = False

    def get_observations(self):
        """
        Returns the current state of the robot as described in the __init__
        Uses the supervisor
        """

        # Image
        red_object, small_gray_image = self.detect_red_object() # return scaled down image
        image_array_binary = small_gray_image.flatten() # turn into a 1D array
        image_array_binary[image_array_binary != 0] = 1 # all pixels that contain the red object are turned to 1s
        image_array_binary = image_array_binary.tolist() # convert to list


        # print(f"Detected: {red_object}, Flattened Image: {image_array_binary}")
        # print(f"Camera Heigh and Width: {self.obs_pixel_height}, {self.obs_pixel_width}")

        # Lidar Rays
        lidar_rays = np.round_(np.array(self.lidar.getRangeImage())[self.lidar_obs_indices],2)
        lidar_rays = np.clip(lidar_rays,0,4)

        return lidar_rays.tolist() + image_array_binary

    
    def detect_red_object(self):
        """
        Determines if the dark red rectangle visual cue is contained in the RGB pixels of the front camera
        """

        # gather image
        image = self.camera.getImage()
        image = np.frombuffer(image, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to detect red objects
        mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)

        # Count the number of non-zero pixels in the mask
        num_red_pixels = cv2.countNonZero(mask)

        # Mask the image so that only the pixels related to the visual cue is recognized
        # masked_image = cv2.bitwise_and(image, image, mask=mask)
        result_image = np.zeros_like(image)
        result_image[np.where(mask != 0)] = [255, 255, 255, 255] # turn it to white so pixel gets recognized in pixelation
        gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY) # convert from RGB to greyscale to reduce dimensions
        small_gray_image = cv2.resize(gray_image, (self.obs_pixel_height, self.obs_pixel_width)) # reduce the pixels in image

        # print(f"Gray Scale Image Array: {np.array(gray_image, dtype=np.uint8).tolist()}")
        # print(f"Small Gray Scale Image Array: {np.array(small_gray_image, dtype=np.uint8).tolist()}")
        # cv2.imshow("Masked Image", result_image)
        # cv2.imshow("Gray Image", gray_image)
        # cv2.imshow("Small Gray Image", small_gray_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # If there are more than a threshold number of dark red pixels
        red_object = False
        threshold = 0  # Adjust this threshold as needed
        if num_red_pixels > threshold:
            # print("Red object detected!")
            red_object = True

        return red_object, small_gray_image


    def get_reward(self, action):
        """
        Reward is based on if the front facing camera can see the object and if the robot is not in a corner (front facing LiDAR rays detect a wall)
        """

        reward = 0

        red_object, _ = self.detect_red_object() # extract if the red object is in view
        lidar_rays = np.round_(np.array(self.lidar.getRangeImage()),2) # extract the lidar rays
        if (not np.any(lidar_rays[25:35] < 4) ) and red_object: # if none of the front facing lidar rays are reading a non infinity value and there is a red object
            reward = 1

        return reward

    def is_done(self):
        """
        An episode ends if the robot reaches the goal or if the robot collides with a wall
        """

        # if loop is false, end the episode if the robot finished the course
        # print(f"Loop Episode Finish: {self.loop_episode_finish}, Current checkpoint to complete: {self.current_checkpoint_to_complete}")
        if self.loop_episode_finish:
            print(f"REACHED GOAL, TERMINAL STATE")
            self.loop_episode_finish = False
            return True

        # collision
        lidar_rays = np.round_(np.array(self.lidar.getRangeImage()),2)
        if np.any(lidar_rays < 0.2):
            return True

        return False

    def solved(self):
        """
        Method checks if the last 100 episodes average more than the solved reward condition
        """
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            solved_reward = 10000
            if (not self.loop):
                solved_reward = 900
            if np.mean(self.episode_score_list[-100:]) > solved_reward:  # Last 100 episode scores average value
                return True
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return np.array([0.0 for _ in range(self.observation_space.shape[0])])

    def apply_action(self, action):
        """
        Based on the action executed, apply the action to the actual simulated environment
        """
        # Extract the action
        action = int(action[0])

        # 2 discrete speeds
        motor_speed = 40
        if action > self.num_discrete_turn_angles - 1:
            motor_speed = 80
        
        # apply a steering angle based on the action index
        steering_angle = normalize_to_range(action % self.num_discrete_turn_angles, 0, self.num_discrete_turn_angles-1, -0.73, 0.73)
        # print(f"Action: {action}, Steering Angle: {steering_angle} \n")

        # apply the action
        self.set_velocity(motor_speed)
        self.set_steering_angle(steering_angle)

    def set_velocity(self,v: float):
        '''
        Sets rotational velocity of the rear wheel drive motors to v radians/second.
        '''
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(v)
     
    def set_steering_angle(self, angle_rad: float):
        '''
        Sets front wheel directions to appropriate angles given their horizontal wheel distances
        for an Ackermann vehicle.
        '''
        # trackFront = 0.254
        # wheelbase = 0.2921
        angle_right = 0
        angle_left = 0
        if math.fabs(angle_rad) > 1e-5:   
            angle_right = math.atan(1. / (1./math.tan(angle_rad) - self.trackFront / (2 * self.wheelbase)));
            angle_left = math.atan(1. / (1./math.tan(angle_rad) + self.trackFront / (2 * self.wheelbase)));
        self.steer_right_motor.setPosition(angle_right)
        self.steer_left_motor.setPosition(angle_left)

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")