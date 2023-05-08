##########
# Final Project Robot Controller
# Author: Jake Miles-Colthup
# Date Created: 04/19/2023
# Date Modified: 
# Purpose: 
##########

##########
# Import Libraries
##########
from SimpleCar import SimpleCar
from utilities import plot_data
import numpy as np
from DDPG_agent import DDPGAgent
import copy
from bayes_opt import BayesianOptimization
import time

##########
# Functions
##########
def update_top_networks(top_episodes, top_scores, top_networks, episode, episode_score, episode_network):
    '''
    Function updates the running list of top performing agents in the environment for saving checkpoints
    '''
    data = list(zip(episode, episode_score, episode_network))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    # Update the top scores and networks if necessary
    for i, (episode, score, network) in enumerate(sorted_data):
        if score >= min(top_scores):
            lowest_score_index = top_scores.index(min(top_scores))
            top_scores[lowest_score_index] = score
            top_networks[lowest_score_index] = network
            top_episodes[lowest_score_index] = episode
            top_episodes, top_scores, top_networks = zip(*sorted(zip(top_episodes, top_scores, top_networks), reverse=True))

    return list(top_episodes[:3]), list(top_scores[:3]), list(top_networks[:3])

def trainDDPG(env, agent):
    '''
    Function that runs training loop for DDPG agent adding exploration noise
    '''
    # Initialize parameters
    episode_limit = 12000
    episode_count = 0
    solved = False  # Whether the solved requirement is met
    top_scores = [-float('inf'), -float('inf'), -float('inf')]
    top_agents = [[None], [None], [None]]
    top_episodes = [-1, -1, -1]

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < episode_limit:
        # Reset robot and get starting observation
        state = env.reset()
        env.episode_score = 0
        env.current_waypoint_idx = 0
        env.closest_waypoint_distance = 0
        env_steps = 0
        env.speed_history = []

        # Inner loop is the episode loop
        for step in range(env.steps_per_episode):
            # In training mode the agent returns the action plus OU noise for exploration
            action = agent.choose_action_train(state)
            # Step the supervisor to get the current selected_action reward, the new state and whether we reached
            # the done condition
            new_state, reward, done, info = env.step(action)
            # Save the current state transition in agent's memory
            agent.remember(state, action, reward, new_state, int(done))
            # Accumulate episode reward
            env.episode_score += reward
            # Perform a learning step
            agent.learn()
            if done or step == env.steps_per_episode - 1:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                solved = env.solved()  # Check whether the task is solved
                break
            # state for next step is current step's new_state
            state = new_state  
            env_steps += 1

        # Update networks after episode conclusion and print results
        top_episodes, top_scores, top_agents = update_top_networks(top_episodes, top_scores, top_agents, [episode_count], [env.episode_score], [copy.deepcopy(agent)])
        print(f"Episode: {episode_count} terminated with score: {env.episode_score} after {env_steps} steps. Best Episodes: {list(zip(top_episodes, top_scores))}")
        episode_count += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # this is done to smooth out the plots
    moving_avg_n = 10
    plot_data(np.convolve(env.episode_score_list, np.ones((moving_avg_n,)) / moving_avg_n, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    return top_agents

def optimize_ddpg(env, lr_actor, lr_critic, theta, sigma, actor_units, critic_units, batch_size):
    '''
    Function runs 500 training loop episodes with an agent created with the defined hyperparameters
    '''
    # Create an instance of the DDPG agent with the specified hyperparameters
    batch_size = int(batch_size)
    actor_units = int(actor_units)
    critic_units = int(critic_units)
    agent = DDPGAgent(env.observation_space.shape, env.action_space.shape, lr_actor=lr_actor, lr_critic=lr_critic, tau=0.01, gamma=0.99,
                      max_size=1000000, batch_size=batch_size, theta=theta, sigma=sigma, actor_units=actor_units, critic_units=critic_units)
    num_episodes = 500
    
    # Train the agent on the environment for a fixed number of episodes
    for episode in range(num_episodes):
        # Run outer loop until the episodes limit is reached or the task is solved
        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0
        env.current_waypoint_idx = 0
        env.closest_waypoint_distance = 0
        env_steps = 0
        env.speed_history = []

        # Inner loop is the episode loop
        for step in range(env.steps_per_episode):
            # In training mode the agent returns the action plus OU noise for exploration
            action = agent.choose_action_train(state)
            # Step the supervisor to get the current selected_action reward, the new state and whether we reached
            # the done condition
            new_state, reward, done, info = env.step(action)
            # Save the current state transition in agent's memory
            agent.remember(state, action, reward, new_state, int(done))
            # Accumulate episode reward
            env.episode_score += reward
            # Perform a learning step
            agent.learn()
            if done or step == env.steps_per_episode - 1:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                solved = env.solved()  # Check whether the task is solved
                break
            # state for next step is current step's new_state
            state = new_state
            env_steps += 1

        print(f"Episode: {episode} terminated with score: {env.episode_score} after {env_steps} steps.")
    
    # Return the average reward obtained by the agent over the last few episodes
    return np.mean(env.episode_score_list[-15])


def testDDPG(env, agent):
    '''
    Function runs n number of episodes with a specified agent with no noise
    '''
    n_episodes = 100
    score_arr = []
    for episode in range(n_episodes):
        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0
        env.current_waypoint_idx = 0
        env.closest_waypoint_distance = 0
        env_steps = 0
        env.speed_history = []
        while True:
            selected_action = agent.choose_action_test(state)
            state, reward, done, _ = env.step(selected_action)
            env.episode_score += reward  # Accumulate episode reward

            if done:
                print("Reward accumulated =", env.episode_score)
                score_arr.append(env.episode_score)
                break

    print("Average reward over {n_episodes}: {np.mean(score_arr)}")

def race(env, agent):
    '''
    Function that executes three race laps and records time and std_dev
    '''
    n_laps = 3
    time_arr = []
    for lap in range(n_laps):
        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0
        env.current_waypoint_idx = 0
        env.closest_waypoint_distance = 0
        env_steps = 0
        env.speed_history = []
        t_0 = env.get_sim_time()
        while True:
            selected_action = agent.choose_action_test(state)
            state, reward, done, _ = env.step(selected_action)
            env.episode_score += reward  # Accumulate episode reward

            if done:
                t_end = env.get_sim_time()
                lap_time = t_end - t_0
                print(f"Lap {lap} time: {np.round_(lap_time, 2)} seconds")
                time_arr.append(lap_time)
                break

    print(f"Race finished with average lap time: {np.round_(np.mean(time_arr), 2)} seconds and standard deviation {np.std(time_arr)} seconds")



def main():
    '''
    Main function to train and test agent
    Set flags to 1 depending on if trying to optimize, train, or test agents
    Optimizer Output:
        {'target': 204.349999999999, 'params': {'actor_units': 17.314604473500456, 'batch_size': 96.36488097712662, 'critic_units': 144.12252205448837, 'lr_actor': 0.0006028208456011765, 'lr_critic': 0.0002263482447357104, 'sigma': 0.17924059563395153, 'theta': 0.4202978274702147}}
    '''
    OPT_FLAG = 0
    TRAIN_FLAG = 0
    TEST_FLAG = 0
    RACE_FLAG = 1
    env = SimpleCar()

    if OPT_FLAG == 1:
        # Define the hyperparameter search space
        pbounds = {'lr_actor': (0.0001, 0.001),
                'lr_critic': (0.0001, 0.001),
                'batch_size': (32, 128),
                'sigma': (0.1, 0.5),
                'theta': (0.1, 0.5),
                'actor_units': (16, 64),
                'critic_units': (64, 256)}
        
        # Wrapper function that enables env var to be passed into optimizer
        def optimize_ddpg_wrapper(lr_actor, lr_critic, theta, sigma, actor_units, critic_units, batch_size):
            return optimize_ddpg(env, lr_actor, lr_critic, theta, sigma, actor_units, critic_units, batch_size)

        # Initialize the Bayesian optimization algorithm
        optimizer = BayesianOptimization(
            f=optimize_ddpg_wrapper,
            pbounds=pbounds,
            verbose=2,  # verbose=1 prints only when a maximum is observed, verbose=0 is silent
            random_state=1,
        )

        # Run the Bayesian optimization algorithm
        optimizer.maximize(
            init_points=5,
            n_iter=25,
        )

        # Print the best set of hyperparameters found
        print(optimizer.max)

    if TRAIN_FLAG == 1:
        # Initialize agent
        agent = DDPGAgent(env.observation_space.shape, env.action_space.shape, lr_actor=0.0006, lr_critic=0.00023, tau=0.01, gamma=0.99,
                        max_size=1000000, batch_size=96, theta=0.42, sigma=0.17, actor_units=17, critic_units=144)
        top_three_agents = trainDDPG(env, agent)
        # Specify filename extension for saved agents, saves top 3 agents and final agent
        top_three_agents[0].save_models('_1_3')
        top_three_agents[1].save_models('_2_3')
        top_three_agents[2].save_models('_3_3')
        agent.save_models('_last')

    if TEST_FLAG == 1:
        # Initialize agent - make sure parameters are SAME as saved agent
        agent = DDPGAgent(env.observation_space.shape, env.action_space.shape, lr_actor=0.0006, lr_critic=0.00023, tau=0.01, gamma=0.99,
                        max_size=1000000, batch_size=96, theta=0.42, sigma=0.17, actor_units=17, critic_units=144)
        # Load desired checkpoint network
        agent.load_models('_last')
        testDDPG(env, agent)

    if RACE_FLAG == 1:
        # Initialize agent - make sure parameters are SAME as saved agent
        agent = DDPGAgent(env.observation_space.shape, env.action_space.shape, lr_actor=0.0006, lr_critic=0.00023, tau=0.01, gamma=0.99,
                        max_size=1000000, batch_size=96, theta=0.42, sigma=0.17, actor_units=17, critic_units=144)
        # Load desired checkpoint network
        agent.load_models('_race')
        race(env, agent)


if __name__ == "__main__":
    main()