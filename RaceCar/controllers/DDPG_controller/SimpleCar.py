##########
# Final Project Robot Environment
# Author: Jake Miles-Colthup
# Date Created: 04/19/2023
# Date Modified: 
# Purpose: 
##########

##########
# Import Libraries
##########
import gym
from gym import spaces
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
import numpy as np
import math
from utilities import normalize_to_range

##########
# Class Definitions
##########
class SimpleCar(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        '''
        Observation and Action Space of the Robot
        Observation Space:
            Robot's X position in the track
            Robot's Y position in the track
            Robot's Orientation (angle of forwards direction with respect to x axis)
            Robot's Speed
            N LiDAR beam distances
        Action Space:
            Steering angle of front two wheels in radians
            Velocity of rear two wheels in m/s
        '''
        self.n_sensors = 9
        self.observation_space = spaces.Box(low=np.array(self.n_sensors*[0]+[-15, -15, 0, 0]), high=np.array(self.n_sensors*[4]+[15, 15, 5, 2*np.pi]), dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([-0.6, 25.0]), high=np.array([0.6, 80]), dtype=np.float64)

        self.robot = self.getSelf()

        '''
        Initialize sensors and motors
        '''
        self.camera = self.getDevice('camera')
        self.lidar = self.getDevice('LDS-01')
        self.steer_left_motor = self.getDevice('left_steer')
        self.steer_right_motor = self.getDevice('right_steer')
        self.l_motor = self.getDevice('rwd_motor_left')
        self.r_motor = self.getDevice('rwd_motor_right')

        for motor in [self.l_motor, self.r_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)

        self.camera.enable(self.timestep)
        self.lidar.enable(self.timestep) # 100ms LIDAR Readings
        self.lidar.enablePointCloud()

        '''
        Training params
        '''
        self.steps_per_episode = 3000
        self.episode_score = 0.0
        self.episode_score_list = []
        self.waypoints = self.generate_waypoints(240)
        self.current_waypoint_idx = 0
        self.closest_waypoint_idx = 0
        self.speed_history = []

    def get_observations(self):
        '''
        Create a reading of the observation space for the given timestep
        Normalize all values to [-1, 1] for RL agent
        '''
        car_position = self.robot.getPosition()
        x_pos = normalize_to_range(car_position[0], -15, 15, -1, 1)
        y_pos = normalize_to_range(car_position[1], -15, 15, -1, 1)
        speed = normalize_to_range(self.get_speed(), 0, 5, -1, 1)
        orientation = normalize_to_range(self.get_heading(), 0, 2*np.pi, -1, 1)
        sensor_reading = self.get_lidar_reading(self.n_sensors)
        sensor_obs = [normalize_to_range(reading, 0, 4, -1, 1) for reading in sensor_reading]

        return sensor_obs + [x_pos, y_pos, speed, orientation]

    def get_default_observation(self):
        '''
        Used internally by deepbots when a new training episode starts
        '''
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action):
        '''
        Reward function that agent trains on, called at every step of the simulation based on the robots state and action
        Positive reward for being in the same orientation of the track and progressing through waypoints
        Negative reward for hitting walls
        Positive reward at the goal state
        Default negative reward to incentivize speed
        '''

        # Initialize rewards
        time_penalty = -0.04
        waypoint_progress_reward = 0.0
        heading_scalar = 1.0
        goal_reward = 0.0
        wall_penalty = 0.0

        # Compute which waypoint is nearest to the robot
        self.closest_waypoint_idx = self.get_closest_waypoint(self.robot.getPosition()[0:2])
        # Check that the robot is progressing through the track the correct direction (mainly needed at very beginning of track)
        if self.closest_waypoint_idx - self.current_waypoint_idx < 50:
            # Waypoint progress reward for moving through waypoints
            waypoint_progress_reward = self.closest_waypoint_idx - self.current_waypoint_idx
            # Determine the angle between the waypoints and the car angle, scale progress by difference
            waypoint_heading = self.get_track_angle(self.closest_waypoint_idx)
            robot_heading = self.get_heading()
            diff = abs(waypoint_heading - robot_heading)
            if diff > np.pi:
                diff = abs(2*np.pi - diff)
            if diff > np.pi/2:
                heading_scalar = normalize_to_range(diff, np.pi/2, np.pi, 0.0, 0.5)
            else:
                heading_scalar = normalize_to_range(diff, 0.0, np.pi/2, 1.0, 0.0)
            # Update current waypoint of robot
            self.current_waypoint_idx = self.closest_waypoint_idx

        # If robot is progressing the wrong direction, give negative reward
        else:
            waypoint_progress_reward = -5.0
            heading_scalar = 1.0

        # If robot is at the end of the waypoint list give large reward
        if self.current_waypoint_idx == self.waypoints.shape[0] - 3: # -3 due to waypoint formulation
            goal_reward = 100.0
        
        # If robot is in a collision, give negative reward
        if self.get_collision():
            wall_penalty = -2.0

        # Return reward for a given timestep
        reward = np.round_(waypoint_progress_reward*heading_scalar + time_penalty + goal_reward + wall_penalty, 3)
        return reward


    def is_done(self):
        '''
        Function called at every timestep to check whether the agent has reached a terminal state
        Terminal if stopped, in collision or at goal
        '''
        # Check if robot is in collision
        if self.get_collision():
            return True
        
        # Look at previous 100 steps to determiine if robot is stopped
        if len(self.speed_history) > 100:
            if np.mean(self.speed_history) < 0.1:
                return True
            else:
                self.speed_history.pop(0)
                self.speed_history.append(self.get_speed())
        else:
            self.speed_history.append(self.get_speed())

        # Check if robot has reeached the goal state
        if self.current_waypoint_idx == self.waypoints.shape[0] - 3:
            return True

        return False
    
    def solved(self):
        '''
        Function that checks if a the RL problem has been solved
        Solved if over 100 episodes with mean score greater than 190
        '''
        if len(self.episode_score_list) > 100:
            if np.mean(self.episode_score_list[-100:]) > 190.0:
                return True
        return False

    def get_info(self):
        '''
        Dummy function required by deepbots framework and gym
        '''
        return None

    def render(self, mode='human'):
        '''
        Dummy function required by deepbots framework and gym
        '''
        pass

    def apply_action(self, action):
        '''
        Function that applies the selected action to the actual robot
        '''

        # Convert the [-1, 1] normalized action to the action space
        action = [normalize_to_range(action, -1, 1, self.action_space.low[i], self.action_space.high[i]) for (i, action) in enumerate(action)]
        action = np.clip(action, self.action_space.low ,self.action_space.high)
        # Extract the steering angle and wheel speed
        steering_angle = action[0]
        rear_wheel_speed = action[1]
        # Set the steering angle and wheel speed
        self.set_steering_angle(steering_angle)
        self.set_velocity(rear_wheel_speed)


    def generate_waypoints(self, num_points):
        '''
        Function that generates num_points number of waypoints around the track
        TODO: Losing waypoints there are num_points - 3 total waypoints created
        '''
        # Initialize waypoints array and determine needed spacing
        waypoints = []
        length = 22.0
        width = 22.0
        x_spacing = length / (num_points // 4)
        y_spacing = width / (num_points // 4)

        # Starting point
        waypoints.append((0.0, 11.0))

        # First segment
        for i in range(num_points // 8):
            x = x_spacing * (i + 1)
            y = 11.0
            waypoints.append((x, y))

        # Second segment
        for i in range(num_points // 4 - 1):
            x = 11.0
            y = 11.0 - y_spacing * (i + 1)
            waypoints.append((x, y))

        # Third segment
        for i in range(num_points // 4 - 1):
            x = 11.0 - x_spacing * (i + 1)
            y = -11.0
            waypoints.append((x, y))

        # Fourth segment
        for i in range(num_points // 4 - 1):
            x = -11.0
            y = -11.0 + y_spacing * (i + 1)
            waypoints.append((x, y))

        # Fifth segment
        for i in range(num_points // 8):
            x = -(11 - x_spacing * (i + 1))
            y = 11.0
            waypoints.append((x, y))

        # Back to starting point
        waypoints.append((0.0, 11.0))

        return np.array(waypoints)
    
    def get_collision(self):
        '''
        Function checks if cone of lidar readings around the front of vehicle are less than determined collision threshold
        '''
        # Get the lidar readings
        lidar_readings = self.lidar.getRangeImage()
        # Check for collision
        if any(reading < 0.4 for reading in lidar_readings[27:34]):
            return True
        return False
    
    def set_velocity(self, v: float):
        '''
        Sets rotational velocity of the rear wheel drive motors to v radians/second.
        '''
        for motor in [self.l_motor, self.r_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(v)

    def set_steering_angle(self, angle_rad: float):
        '''
        Sets front wheel directions to appropriate angles given their horizontal wheel distances
        for an Ackermann vehicle.
        '''
        trackFront = 0.254
        wheelbase = 0.2921
        angle_right = 0
        angle_left = 0
        if math.fabs(angle_rad) > 1e-5:   
            angle_right = math.atan(1. / (1./math.tan(angle_rad) - trackFront / (2 * wheelbase)))
            angle_left = math.atan(1. / (1./math.tan(angle_rad) + trackFront / (2 * wheelbase)))
        self.steer_right_motor.setPosition(angle_right)
        self.steer_left_motor.setPosition(angle_left)

    def get_heading(self):
        '''
        Function computes the angle of the vehicle with respect to the positive x world axis
        '''
        orientation = self.robot.getOrientation()
        heading = -math.atan2(orientation[1], orientation[4])
        return heading
    
    def get_lidar_reading(self, n):
        '''
        Function gets n number of evenly distributed LiDAR beam readings for the observation space
        '''
        lidar_readings = self.lidar.getRangeImage()
        spacing = len(lidar_readings)/n
        readings = [lidar_readings[int(i*spacing)] for i in range(n)]
        # Sensor has a max distance of 4, replace 'inf' with 4
        return [4 if math.isinf(x) else x for x in readings]
    
    def get_speed(self):
        '''
        Function determines the 2D speed of the robot within the environment
        '''
        return np.linalg.norm(self.robot.getVelocity()[0:2])
    
    def get_closest_waypoint(self, pos):
        '''
        Function computes which waypoint is closest to the vehicles current position
        '''
        distances = np.linalg.norm(self.waypoints - np.array(pos), axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def get_track_angle(self, wp_idx):
        '''
        Function to get the track angle ahead of a given waypoint
        '''
        # if waypoint is the last in the array, compute angle between waypoint and previous waypoint
        if wp_idx == self.waypoints.shape[0] - 1:
            wp_1 = self.waypoints[wp_idx-1]
            wp_2 = self.waypoints[wp_idx]
        # if waypoint is second last in the array, compute angle between waypoint and next waypoint
        elif wp_idx == self.waypoints.shape[0] - 2:
            wp_1 = self.waypoints[wp_idx]
            wp_2 = self.waypoints[wp_idx + 1]
        # Generally, compute the track angle of the next waypoint and 2 waypoints ahead
        else:
            wp_1 = self.waypoints[wp_idx+1]
            wp_2 = self.waypoints[wp_idx + 2]
        

        track_angle = math.atan2(wp_2[1] - wp_1[1], wp_2[0] - wp_1[0])
        return track_angle
    
    def get_sim_time(self):
        '''
        Function returns the current simulation time
        '''
        return self.getTime()