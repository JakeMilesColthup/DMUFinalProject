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
        0	Vehicle Position x axis      -Inf            Inf
        1	Vehicle Position y axis      -Inf            Inf
        2	Vehicle Heading              -pi             pi
        3	Vehicle Speed                -Inf            Inf
        4	Lidar Ray 0                  0               4
        .   .
        .   .
        .   .
        63	Lidar Ray 59                 0               4

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

        self.camera.enable(self.timestep)
        self.lidar.enable(self.timestep) # 100ms LIDAR Readings
        self.lidar.enablePointCloud()

        #### Observation Variables ####
        max_lidar_ray_indices = 60
        self.num_lidar_rays_in_obs = min(num_lidar_rays_in_obs, max_lidar_ray_indices) # max is 60
        self.lidar_obs_indices = (np.rint(np.linspace(0, max_lidar_ray_indices - 1, num=self.num_lidar_rays_in_obs))).astype(int)
        # print(f"Lidar Indices: {self.lidar_obs_indices}")

        #### Reward Variables ####
        # waypoints
        self.waypoints = self.setup_waypoints(num_wp_per_side)
        self.previous_waypoint_index = 0
        # goal
        self.num_checkpoints = num_checkpoints
        self.checkpoints_wp_indices = (np.rint(np.linspace(0, len(self.waypoints) - 1, num=self.num_checkpoints))).astype(int)
        self.current_checkpoint_to_complete = 1
        self.endpoint_tol = int(num_wp_per_side/10)
        self.loop = loop
        self.loop_episode_finish = False
        # print(f"Checkpoint WP Indices: {self.checkpoints_wp_indices}")

        # print(f"Waypoints: {self.waypoints}")

        #### Episodes ####
        self.steps_per_episode = steps_per_episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

        #### Gym Environment ####
        # observation space lists
        obs_space_low_list = [-15, -15, -np.pi, 0] + [0 for _ in range(self.num_lidar_rays_in_obs)]
        obs_space_high_list = [15, 15, np.pi, np.inf] + [4 for _ in range(self.num_lidar_rays_in_obs)]
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
        # resets the required variables that affect the reward
        self.episode_score = 0
        self.previous_waypoint_index = 0
        self.current_checkpoint_to_complete = 1
        self.loop_episode_finish = False

    def setup_waypoints(self, num_waypoint_per_side):
        """
        Determine the waypoints of the course that the robot wishes to achieve
        """
        north2northeast = np.linspace((-1,11),(10.95,11),int(num_waypoint_per_side/2))
        northeast2southeast = np.linspace((11,11),(11,-10.95),num_waypoint_per_side)
        southeast2southwest = np.linspace((11,-11),(-10.95,-11),num_waypoint_per_side)
        southwest2northwest = np.linspace((-11,-11),(-11,10.95),num_waypoint_per_side)
        northwest2north = np.linspace((-11,11),(-1.5,11),int(num_waypoint_per_side/2))
        waypoints = np.concatenate((north2northeast, northeast2southeast, southeast2southwest, southwest2northwest, northwest2north), axis=0)
        
        return waypoints

    def get_observations(self):
        """
        Returns the current state of the robot as described in the __init__
        Uses the supervisor
        """
        
        # Position
        vehicle_position = self.robot.getPosition()
        # Velocity/Speed
        vehicle_velocity = self.robot.getVelocity()
        vehicle_speed = np.linalg.norm(vehicle_velocity)
        # Orientation
        heading_angle = self.get_robot_heading()

        # print(f"X: {vehicle_position[0]}, Y: {vehicle_position[1]}, Heading: {math.degrees(heading_angle)}, Speed: {vehicle_speed}")
        # print(f"Trackfront: {self.trackFront}, Wheelbase: {self.wheelbase}  \n")

        # Lidar Rays
        lidar_rays = np.round_(np.array(self.lidar.getRangeImage())[self.lidar_obs_indices],2)
        lidar_rays = np.clip(lidar_rays,0,4)

        # print(f"Observation: {[vehicle_position[0], vehicle_position[1], heading_angle, vehicle_speed] + lidar_rays.tolist()}")
        return [vehicle_position[0], vehicle_position[1], heading_angle, vehicle_speed] + lidar_rays.tolist()
    
    def get_robot_heading(self):
        """
        Determine the heading of the robot using the supervisor
        """
        vehicle_orientation = self.robot.getOrientation()
        # print(f"Vehicle Orientation: {vehicle_orientation}")
        # r12 = vehicle_orientation[1]   # Coefficient m[1][0]
        # r22 = vehicle_orientation[4]   # Coefficient m[1][1]
        # Calculate the heading angle
        heading_angle = -math.atan2(vehicle_orientation[1], vehicle_orientation[4])
        return heading_angle


    def get_reward(self, action):
        """
        Reward is based on progress, heading, and goal completion
        """

        ####### waypoint progress reward #######
        # Position
        vehicle_position = self.robot.getPosition()
        vehicle_position2D = np.array([vehicle_position[0],vehicle_position[1]])

        # find closest waypoint
        closest_waypoint_index = self.find_closest_waypoint(vehicle_position2D)
        wp_progress_reward = closest_waypoint_index - self.previous_waypoint_index

        # in case at the end of the track and not reached goal (the track wrapping around)
        if self.current_checkpoint_to_complete == self.num_checkpoints and closest_waypoint_index >= 0 and  (self.previous_waypoint_index <= len(self.waypoints) and self.previous_waypoint_index >= len(self.waypoints) - self.endpoint_tol):
            # wp_progress_reward = closest_waypoint_index - (self.previous_waypoint_index - len(self.waypoints))
            wp_progress_reward = 1

        # update checkpoints
        if self.current_checkpoint_to_complete != (self.num_checkpoints):
            if closest_waypoint_index <= self.checkpoints_wp_indices[self.current_checkpoint_to_complete] and closest_waypoint_index >= self.checkpoints_wp_indices[self.current_checkpoint_to_complete-1]:
                self.current_checkpoint_to_complete += 1

        # print(f"Closest Waypoint: {closest_waypoint_index}, Last Waypoint: {self.previous_waypoint_index}")

        # if the car is at the end of the course, do not decrement the reward in an absurd fashion and actually reward the car for completing the course. Reset the self.current_checkpoint_to_complete variable
        if vehicle_position[0] > 0 and self.current_checkpoint_to_complete == self.num_checkpoints:
            wp_progress_reward = int(len(self.waypoints)) # set the reward to be drastic, for completing the course
            self.current_checkpoint_to_complete = 1
            if not self.loop:
                self.loop_episode_finish = True
                # print(f"Setting episode to end")
            print("-------------------- REACHED LAST WAYPOINT OF THE COURSE IN A SEQUENTIAL MANNER! -----------------------")

        # update previous waypoint index
        self.previous_waypoint_index = closest_waypoint_index

        ####### waypoint heading reward #######
        # waypoint heading
        wp_heading_angle = 0
        if closest_waypoint_index != len(self.waypoints) - 1:
            wp_close = self.waypoints[closest_waypoint_index]
            # wp_previous = self.waypoints[closest_waypoint_index-1]
            # wp_heading_angle = math.atan2(wp_close[1]-wp_previous[1],wp_close[0]-wp_previous[0])
            wp_next = self.waypoints[closest_waypoint_index+1]
            wp_heading_angle = math.atan2(wp_next[1]-wp_close[1],wp_next[0]-wp_close[0])
        else: # currently is the last waypoint, set heading to the first waypoint heading
            wp_close = self.waypoints[0]
            wp_next = self.waypoints[1]
            wp_heading_angle = math.atan2(wp_next[1]-wp_close[1],wp_next[0]-wp_close[0])

            

        # robot heading
        robot_heading_angle = self.get_robot_heading()

        # determine reward proportionality
        heading_difference = abs(wp_heading_angle - robot_heading_angle)
        if heading_difference > np.pi:
            heading_difference = abs(2*np.pi - heading_difference)

        # determine heading multiplier (0-1)
        wp_heading_multiplier = normalize_to_range(heading_difference, 0.0, np.pi, 1.0, 0.0)
        # print(f"WP Heading Multiplier: {wp_heading_multiplier}, Heading Diff: {heading_difference}, Robot Heading: {math.degrees(robot_heading_angle)}, WP Heading: {math.degrees(wp_heading_angle)}")
        

        ####### collision negative reward #######
        collision_reward = 0
        # lidar_rays = np.array(self.lidar.getRangeImage())
        # if np.any(lidar_rays < 0.7):
        #     collision_reward = -1

        time_reward = 0

        return np.round_(wp_progress_reward*wp_heading_multiplier + time_reward + collision_reward,3)
        

    def find_closest_waypoint(self, current_position):
        """
        Using the robot's position, find the closest waypoint
        """
        # Calculate Euclidean distances between current position and all waypoints
        distances = np.linalg.norm(self.waypoints - current_position, axis=1)
        
        # Find the index of the closest waypoint
        closest_index = np.argmin(distances)
        
        return closest_index

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
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        """
        print("render() is not used")
