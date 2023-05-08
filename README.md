# Reinforcement Learning on an Ackermann Car
## Introduction
The purpose of this project is to implement and evaluate the performance of a reinforcement learning (RL) agent on a simulated track environment using an Ackermann car representation. Specifically, we explore three different scenarios with varying levels of supervisor access to the car's position, velocity, and orientation as well as onboard sensor data. Our objectives are to evaluate the performance of RL algorithms on these scenarios, and to compare the results of using different algorithms and levels of supervision.
## Ackermann Car Representation
An Ackermann car is a type of vehicle with a steering mechanism that allows the wheels on the left and right sides to turn at different angles, allowing the car to make turns without skidding. In this project, we represent the Ackermann car using a continuous state space, which allows us to apply any value between a minimum and maximum action input. We also explore both discrete and continuous action spaces.
## Webots Environment
We use the Webots environment, which is a full physics and visualization simulation engine, to simulate the Ackermann car on a rectangular track with protruding walls and other obstacles. The environment allows us to simulate various scenarios and to test the performance of the RL agent under different conditions.
## Reinforcement Learning Algorithms
We use two different RL algorithms in this project: Deep Deterministic Policy Gradient (DDPG) and Proximal Policy Optimization (PPO). We use DDPG for the scenario where the Ackermann car has access to continuous action space, and we use PPO for the scenarios where the Ackermann car has access to a discrete action spaces at two levels of supervisor access.
## Results
All policies converged to their optimal states, with the first scenario converging after 19500 episodes, the second scenario converging after 1200 episodes, and the third scenario converging after 3700 episodes.
## Directory Structure
- RaceCar/: Contains the Webots environment used to simulate the Ackermann car.
  - controllers/: Contains the controllers used in the simulation.
    - DDPG_controller/: Contains the code for the DDPG algorithm.
    - PPO_controller/: Contains the code for the PPO algorithm.
    - PPO_supervisor_controller/: Contains the code for the PPO algorithm with supervisor access.
  - worlds/: Contains the .wbt files defining the simulated environments.
  - protos/: Contains protos for robot and objects in the world.
  - plugins/: Contains any additional plugins required for the simulation.
  - libraries/: Contains any additional libraries required for the simulation.
- Report/: Contains the PDF report of the project.
- Results/: Contains the videos and learning curves of the RL agent's performance.

In addition to the controllers/ directory, the RaceCar/ directory includes four additional subdirectories: worlds/, protos/, plugins/, and libraries/. The worlds/ directory contains the .wbt files defining the simulated environments used in the project. The protos/ directory contains protos for robot and objects in the world. The plugins/ and libraries/ directories contain any additional plugins or libraries required for the simulation.

The Report/ directory contains the PDF report summarizing the project's results, while the Results/ directory includes the videos and learning curves of the RL agent's performance.
## Reproducing the Results
To reproduce the results of this project, users can navigate to the controllers directory and run the appropriate controller for the scenario they wish to replicate. They can also refer to the report and the results directory for further information and visualizations of the project's results.

Note that in order to run the project, users will need to install any necessary dependencies and set up the appropriate environment. Detailed setup instructions can be found in the report.