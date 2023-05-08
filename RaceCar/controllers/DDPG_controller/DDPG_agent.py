import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OUActionNoise(object):
    """
    Ornstein–Uhlenbeck noise, https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    Exploration/Exploitation balance defined by sigma (size of random deviation) and theta (attraction to mean)
    """

    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    '''
    Replay buffer stores training transition data to sample from
    '''
    def __init__(self, max_size, input_shape, output_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, *output_shape))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, reward, new_states, terminal


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, lr, fc1_dims, fc2_dims, fc3_dims, name,
                 chkpt_dir='./models/saved/default_ddpg/', use_cuda=False):
        super(CriticNetwork, self).__init__()

        # initialize network parameters
        self.input_shape = input_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.output_shape = output_shape
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        # Build network structure
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(*self.output_shape, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims + fc2_dims, fc3_dims)
        self.q = nn.Linear(fc3_dims, 1)

        # Initialize network weights
        self.initialization()
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Computation device
        if use_cuda:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.to(self.device)

    def initialization(self):
        '''
        Use xavier initialization to initialize network weights
        '''
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.action_value.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state, action):
        '''
        Forward pass through network
        '''
        state_value = self.fc1(state)
        state_value = F.leaky_relu(state_value)
        # state_value = self.bn1(state_value)

        state_value = self.fc2(state_value)
        state_value = F.leaky_relu(state_value)
        # state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        action_value = F.leaky_relu(action_value)

        state_action_value = T.cat((action_value, state_value), dim=1)
        state_action_value = self.fc3(state_action_value)
        state_action_value = F.relu(state_action_value)

        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, name):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file + name)

    def load_checkpoint(self, name):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file + name))


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, lr, fc1_dims, fc2_dims, fc3_dims, name,
                 chkpt_dir='./models/saved/default_ddpg/', use_cuda=False):
        super(ActorNetwork, self).__init__()
        # Initialize network parameters
        self.lr = lr
        self.input_shape = input_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.output_shape = output_shape
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")

        # Build network structure
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.mu = nn.Linear(self.fc3_dims, *self.output_shape)

        # Initialize network weights
        self.initialization()
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Computation device
        if use_cuda:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.to(self.device)

    def initialization(self):
        '''
        Initialize network weights with xavier initialization
        '''
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.mu.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, state):
        '''
        Forward pass through network
        '''
        x = self.fc1(state)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        x = F.leaky_relu(x)

        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self, name):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file + name)

    def load_checkpoint(self, name):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file + name))


class DDPGAgent(object):
    def __init__(self, input_shape, output_shape, lr_actor=0.00005, lr_critic=0.0005, tau=0.01, gamma=0.99,
                 max_size=1000000, batch_size=32, sigma=0.2, theta=0.15, actor_units=16, critic_units=64):
        # Initialize agent hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_shape, output_shape)
        self.output_shape = output_shape
        # Create actor, actor_target, critic, critic_target networks
        self.actor = ActorNetwork(input_shape, output_shape, lr_actor, actor_units, actor_units, actor_units,
                                  name="Actor")

        self.target_actor = ActorNetwork(input_shape, output_shape, lr_actor, actor_units, actor_units, actor_units,
                                         name="TargetActor")

        self.critic = CriticNetwork(input_shape, output_shape, lr_critic, critic_units, critic_units, critic_units,
                                    name="Critic")

        self.target_critic = CriticNetwork(input_shape, output_shape, lr_critic, critic_units, critic_units, critic_units,
                                           name="TargetCritic")
        # Initialize exploration noise
        self.noise = OUActionNoise(np.zeros(output_shape), sigma=sigma, theta=theta)
        # Update params
        self.update_network_parameters(tau=tau)

    def choose_action_train(self, observation):
        '''
        Chooses an action for the agent given an observation
        Exploration noise is incorporated
        '''
        if observation is not None:
            # set actor to evaluate
            self.actor.eval()
            # coonvert observation to tensor and send to actor device
            observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
            # get action probabilities
            mu = self.actor(observation).to(self.actor.device)
            # add exploration noise
            noise = T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            mu_prime = mu + noise
            # set to train
            self.actor.train()
            # return action from network
            return mu_prime.cpu().detach().numpy()
        return np.zeros(self.output_shape)

    def choose_action_test(self, observation):
        '''
        Chooses an action for the agent given an observation
        No exploration noise
        '''
        if observation is not None:
            # set actor to evaluate
            self.actor.eval()
            # Convert observation to tensor
            observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
            # Get action probability
            mu = self.target_actor(observation).to(self.target_actor.device)

            return mu.cpu().detach().numpy()
        return np.zeros(self.output_shape)

    def remember(self, state, action, reward, new_state, done):
        '''
        Store the experience tuple in the replay buffer
        '''
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, batch_size=None):
        '''
        Train the agent by sampling batch from replay buffer and updating the actor and the critic networks
        '''
        if batch_size is None:
            if self.memory.mem_cntr < self.batch_size:
                return
            batch_size = self.batch_size

        # Sample transitions from replay buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)

        # Convert experience tuple to tensor
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # Set networks to evaluate mode
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # Forward pass the new_state tensor and target actions through the target critic network to get the Q-values of the next state-action pairs.
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        # Forward pass the state tensor and action tensor through the critic network to get the Q-values of the current state-action pairs.
        critic_value = self.critic.forward(state, action)

        # Calculate target Q-values with bellmen equation
        target = []
        for j in range(batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])

        target = T.tensor(target).to(self.critic.device)
        target = target.view(batch_size, 1)

        # Set critic to train mode
        self.critic.train()
        # Zero the gradient 
        self.critic.optimizer.zero_grad()
        # Compute critic loss
        critic_loss = F.mse_loss(target, critic_value)
        # Backpropogate the loss
        critic_loss.backward()
        # update critic weights
        self.critic.optimizer.step()
        # Return to evaluate mode
        self.critic.eval()

        # Zero gradient
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        # set actor to train mode
        self.actor.train()
        # forward pass state to get loss
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        # backpropogate loss
        actor_loss.backward()
        # update actor weights
        self.actor.optimizer.step()

        self.update_network_parameters()

    def work(self):
        '''
        Sets networks to evaluation mode
        '''
        self.target_actor.eval()
        self.target_critic.eval()

    def update_network_parameters(self, tau=None):
        '''
        Updates target networks by interpolating parameters of current network and target networks
        '''
        if tau is None:
            tau = self.tau

        # create iterators for network parameters
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # Create dictionaries to store network params
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        # Update the target critic dictionary by interpolating the current critic network parameters and target critic network parameters using the "tau" value.
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()
        # Update the target critic network
        self.target_critic.load_state_dict(critic_state_dict)

        # Update the target actor dictionary by interpolating the current actor network parameters and target actor network parameters using the "tau" value.
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()
        # Update the target actor network
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self, name):
        self.actor.save_checkpoint(name)
        self.critic.save_checkpoint(name)
        self.target_actor.save_checkpoint(name)
        self.target_critic.save_checkpoint(name)

    def load_models(self, name):
        self.actor.load_checkpoint(name)
        self.critic.load_checkpoint(name)
        self.target_actor.load_checkpoint(name)
        self.target_critic.load_checkpoint(name)
