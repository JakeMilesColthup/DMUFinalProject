from numpy import convolve, ones, mean
import numpy as np

from robot_supervisor import SimpleVehicleSupervisor
from agent.PPO_agent import PPOAgent, Transition
from utilities import plot_data
import copy

from torch import from_numpy, no_grad, save, load, tensor, clamp


def run(episode_limit = 5000, path2actor=None, path2critic=None, loop = True):
    """
    Trains the PPO agent, given the environment
    """
    # Initialize supervisor object
    env = SimpleVehicleSupervisor(loop = loop)

    # Initialize the agent
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, batch_size=64)
    if((path2actor is not None) and (path2critic is not None)):
        actor_state_dict = load(path2actor)
        critic_state_dict = load(path2critic)
        agent.actor_net.load_state_dict(actor_state_dict)
        agent.critic_net.load_state_dict(critic_state_dict)

    # Initialize history variables
    episode_count = 0
    solved = False  # Whether the solved requirement is met
    average_episode_action_probs = []  # Save average episode taken actions probability to plot later
    best_model_results = [{"score": -np.inf, "episode": -1, "actor_net": copy.deepcopy(agent.actor_net.state_dict()), "critic_net": copy.deepcopy(agent.critic_net.state_dict())} for _ in range(10)] # Save the top 5 best results 
    minimum_best_result = {"index": 0, "score": -100000}
    # print(f"Best Model Results: {len(best_model_results)}")
    # Starting print statements
    print(f"Starting Discrete PPO. Episode Limit: {episode_limit}")

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < episode_limit:
        state = env.reset()  # Reset robot and get starting observation
        env.reset_reward_variables() # reset the episode_score, previous_waypoint_index, and current_checkpoint_to_complete
        action_probs = []  # This list holds the probability of each chosen action

        # Episode loop
        for step in range(env.steps_per_episode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selected_action, action_prob = agent.work(state, type_="selectAction")
            # Save the current selected_action's probability
            action_probs.append(action_prob)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
            # done condition
            new_state, reward, done, info = env.step([selected_action])

            # Save the current state transition in agent's memory
            trans = Transition(state, selected_action, action_prob, reward, new_state)
            agent.store_transition(trans)

            env.episode_score += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                agent.train_step(batch_size=step + 1) # train the networks
                solved = env.solved()  # Check whether the task is solved
                break

            state = new_state  # state for next step is current step's new_state

        # update the best episode list
        if env.episode_score >= minimum_best_result["score"]:
            # replace the minimum best with the current episode
            best_model_results[minimum_best_result["index"]] = {"score": env.episode_score, "episode": episode_count, "actor_net": copy.deepcopy(agent.actor_net.state_dict()), "critic_net": copy.deepcopy(agent.critic_net.state_dict())}
            # find the new minimum best result
            min_index = np.argmin(np.array([object["score"] for object in best_model_results]))
            min_score = best_model_results[min_index]["score"]
            minimum_best_result["score"] = min_score
            minimum_best_result["index"] = min_index

        # Progress print statement
        print("Episode #", episode_count,"/", episode_limit, " score:", env.episode_score, " Checkpoints Completed: ",env.current_checkpoint_to_complete-1,"-----Best Episodes",[object["episode"] for object in best_model_results], "Best associated scores:", ([object["score"] for object in best_model_results]))

        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avg_action_prob = mean(action_probs)
        average_episode_action_probs.append(avg_action_prob)
        print("Avg action prob:", avg_action_prob)

        episode_count += 1  # Increment episode counter

    ##### Save Agents #####
    # save the best results
    path = "SavedAgents/"
    for model in best_model_results:
        save(model["actor_net"], path + f"Ep{model['episode']}_Score{model['score']}" + '_actor.pkl')
        save(model["critic_net"], path + f"Ep{model['episode']}_Score{model['score']}" + '_critic.pkl')
    # save the last result
    save(agent.actor_net.state_dict(), path + f"Ep{episode_count}_Score{env.episode_score}" + '_actor.pkl')
    save(agent.critic_net.state_dict(), path + f"Ep{episode_count}_Score{env.episode_score}" + '_critic.pkl')


    ##### Plotting #####
    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    moving_avg_n = 10
    plot_data(convolve(env.episode_score_list, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plot_data(convolve(average_episode_action_probs, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             "episode", "average episode action probability", "Average episode action probability over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")
    
    # Run the simulation using the last learned policy
    state = env.reset()
    env.episode_score = 0
    while True:
        # choose action
        selected_action, action_prob = agent.work(state, type_="selectActionMax") # only choose the best action based on NNs

        # step environment
        state, reward, done, _ = env.step([selected_action])
        env.episode_score += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            state = env.reset()


def run_model(path2actor, path2critic, loop = True):
    """
    Runs a simulation given a previously saved model
    """
    # Initialize supervisor object
    env = SimpleVehicleSupervisor(loop = loop)

    # Initialize the agent
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, batch_size=64)

    # load in the actor and critic dictionaries
    actor_state_dict = load(path2actor)
    critic_state_dict = load(path2critic)
    agent.actor_net.load_state_dict(actor_state_dict)
    agent.critic_net.load_state_dict(critic_state_dict)

    # run the simulation with this model
    state = env.reset()
    env.episode_score = 0
    timestep = env.timestep
    time_until_terminal = 0
    while True:
        # choose action
        selected_action, action_prob = agent.work(state, type_="selectActionMax") # only choose the best action based on NNs

        # step environment
        state, reward, done, _ = env.step([selected_action])

        # track performance
        env.episode_score += reward  # Accumulate episode reward
        time_until_terminal += timestep # Track time

        if done:
            print(f"Reward accumulated = {env.episode_score}. Time until terminal state = {time_until_terminal/1000.0}")
            env.episode_score = 0
            time_until_terminal = 0
            state = env.reset()
