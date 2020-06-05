import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FCQAgent(object):
    """Agent for interacting and learning from the environment."""
    def __init__(self, state_size, action_size, gamma=0.99):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            gamma (float): gamma hyperparameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model = FCQNetwork(state_size, action_size)
        self.gamma = gamma
        self.memory = []
        self.k_max = 1000
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

    def step(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_values = self.model(state)
        # Epsilon-greedy action selection
        if random.random() > eps:  # If eps=1 --> always random choice
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = list(map(list, zip(*self.memory)))
        states = torch.FloatTensor(states).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE).view(-1, 1)
        actions = torch.LongTensor(actions).to(DEVICE).view(-1, 1)
        dones = torch.IntTensor(dones).to(DEVICE).view(-1, 1)

        max_a_q_sp = self.model(next_states).detach().max(1)[0].unsqueeze(1) * (1 - dones)
        target_q_s = rewards + max_a_q_sp
        q_sa = self.model(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(target_q_s, q_sa)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []  # Throw away past experience after learning


class FCQNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCQNetwork, self).__init__()
        self.input_layer = nn.Linear(state_size, fc1_units)
        self.hidden_layer = nn.Linear(fc1_units, fc2_units)
        self.output_layer = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)