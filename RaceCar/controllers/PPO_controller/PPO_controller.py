"""
More runners for discrete RL algorithms can be added here.
"""
import PPO_runner

if __name__ == '__main__':


    ###### _2Speed_10Steering_60Lidar_10000Epi_PostOptimize ######

    # model_path2actor = "SavedAgents\_5000Epi_2Speed_10Steering\Ep3788_Score2443_actor.pkl"
    # model_path2critic = "SavedAgents\_5000Epi_2Speed_10Steering\Ep3788_Score2443_critic.pkl"

    # model_path2actor = "SavedAgents\_5000Epi_2Speed_10Steering\Ep3799_Score2052_actor.pkl"
    # model_path2critic = "SavedAgents\_5000Epi_2Speed_10Steering\Ep3799_Score2052_critic.pkl"

    ##### Change file path structure to run on mac vs windows
    model_path2actor = "SavedAgents/_5000Epi_2Speed_10Steering/Ep3799_Score2052_actor.pkl"
    model_path2critic = "SavedAgents/_5000Epi_2Speed_10Steering/Ep3799_Score2052_critic.pkl"

    ######## Run Reinforcement Learning ########
    # PPO_runner.run(episode_limit=10000, loop = False) # from scratch
    # PPO_runner.run(episode_limit=10000, path2actor=model_path2actor, path2critic=model_path2critic, loop = False) # using a previously learned network

    ######## Run Already Learned Model ########
    PPO_runner.run_model(model_path2actor, model_path2critic, loop = False)
