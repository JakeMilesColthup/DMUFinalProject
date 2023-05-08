"""
More runners for discrete RL algorithms can be added here.
"""
import PPO_runner

if __name__ == '__main__':

    ###### _2Speed_10Steering_60Lidar_20000Epi ######
    # 21.584 Seconds
    # model_path2actor = "SavedAgents\_2Speed_10Steering_60Lidar_20000Epi\Ep19419_Score806.4870000000002_actor.pkl"
    # model_path2critic = "SavedAgents\_2Speed_10Steering_60Lidar_20000Epi\Ep19419_Score806.4870000000002_critic.pkl"

    # 21.632 Seconds
    # model_path2actor = "SavedAgents\_2Speed_10Steering_60Lidar_20000Epi\Ep19392_Score806.707000000001_actor.pkl"
    # model_path2critic = "SavedAgents\_2Speed_10Steering_60Lidar_20000Epi\Ep19392_Score806.707000000001_critic.pkl"

    ###### _2Speed_10Steering_60Lidar_10000Epi_PostOptimize ######

    # 20.224 Seconds
    # model_path2actor = "SavedAgents\_2Speed_10Steering_60Lidar_10000Epi_PostOptimize\Ep8943_Score807.3880000000001_actor.pkl"
    # model_path2critic = "SavedAgents\_2Speed_10Steering_60Lidar_10000Epi_PostOptimize\Ep8943_Score807.3880000000001_critic.pkl"

    # 20.096 Seconds
    # model_path2actor = "SavedAgents\_2Speed_10Steering_60Lidar_10000Epi_PostOptimize\Ep9209_Score807.1840000000004_actor.pkl"
    # model_path2critic = "SavedAgents\_2Speed_10Steering_60Lidar_10000Epi_PostOptimize\Ep9209_Score807.1840000000004_critic.pkl"

    ##### Change file path structure to run on mac vs windows
    model_path2actor = "SavedAgents/_2Speed_10Steering_60Lidar_10000Epi_PostOptimize/Ep9209_Score807.1840000000004_actor.pkl"
    model_path2critic = "SavedAgents/_2Speed_10Steering_60Lidar_10000Epi_PostOptimize/Ep9209_Score807.1840000000004_critic.pkl"

    ######## Run Reinforcement Learning ########
    # PPO_runner.run(episode_limit=25000, loop = False) # from scratch
    # PPO_runner.run(episode_limit=10000, path2actor=model_path2actor, path2critic=model_path2critic, loop = False) # using a previously learned network

    ######## Run Already Learned Model ########
    PPO_runner.run_model(model_path2actor, model_path2critic, loop = False)
