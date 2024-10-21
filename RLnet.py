import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



class ActorNetwork(nn.Module):
    """
    Actor Network for the A3C algorithm.
    Supports both discrete and continuous action spaces.
    """
    def __init__(self, state_dim, action_dim, continuous=False):
        super(ActorNetwork, self).__init__()
        self.continuous = continuous
        self.action_dim = action_dim

        # Define network architecture
        self.l1_neurons, self.l2_neurons, self.l3_neurons, self.l4_neurons = 100, 150, 200, 100

        # Create layers
        self.fc1 = self._create_linear_layer(state_dim, self.l1_neurons)
        self.fc2 = self._create_linear_layer(self.l1_neurons, self.l2_neurons)
        self.fc3 = self._create_linear_layer(self.l2_neurons, self.l3_neurons)
        self.fc4 = self._create_linear_layer(self.l3_neurons, self.l4_neurons)

        # Output layer depends on action space type
        if not self.continuous:
            self.outputLayer = nn.Linear(self.l4_neurons, action_dim)
            nn.init.kaiming_uniform_(self.outputLayer.weight)
        else:
            self.mu_net = nn.Linear(self.l4_neurons, 1)
            self.sigma_net = nn.Linear(self.l4_neurons, 1)
            nn.init.kaiming_uniform_(self.mu_net.weight)
            nn.init.kaiming_uniform_(self.sigma_net.weight)
            self.act_mu = nn.Tanh()
            self.act_sigma = nn.Softplus()

    def _create_linear_layer(self, in_features, out_features):
        """Helper method to create a linear layer with Xavier initialization."""
        layer = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(layer.weight)
        return layer

    def forward(self, x):
        """Forward pass through the network."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = F.sigmoid(layer(x))

        if not self.continuous:
            return F.softmax(self.outputLayer(x), dim=-1)
        else:
            mu = self.act_mu(self.mu_net(x)) * self.action_dim
            sigma = self.act_sigma(self.sigma_net(x)) + 1e-5
            return mu, sigma





class CriticNetwork(nn.Module):
    """
    Critic Network for the A3C algorithm.
    Estimates the value function V(s).
    """
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.l1_neurons, self.l2_neurons, self.l3_neurons, self.l4_neurons = 100, 150, 200, 100

        # Create layers
        self.fc1 = self._create_linear_layer(state_dim, self.l1_neurons)
        self.fc2 = self._create_linear_layer(self.l1_neurons, self.l2_neurons)
        self.fc3 = self._create_linear_layer(self.l2_neurons, self.l3_neurons)
        self.fc4 = self._create_linear_layer(self.l3_neurons, self.l4_neurons)
        self.outputLayer = nn.Linear(self.l4_neurons, 1)
        nn.init.kaiming_uniform_(self.outputLayer.weight)

    def _create_linear_layer(self, in_features, out_features):
        """Helper method to create a linear layer with Xavier initialization."""
        layer = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(layer.weight)
        return layer

    def forward(self, x):
        """Forward pass through the network."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = F.sigmoid(layer(x))
        return self.outputLayer(x)

if __name__ == '__main__':
    # Test code for the networks
    S_INFO, S_LEN, AGENT_NUM, ACTION_DIM = 6, 6, 3, 6
    discount = 0.9

    c_net = CriticNetwork(S_LEN)
    t_c_net = CriticNetwork(S_LEN)
    a_net = ActorNetwork(S_LEN, ACTION_DIM)

    a_optim = torch.optim.Adam(a_net.parameters(), lr=0.001)
    c_optim = torch.optim.Adam(c_net.parameters(), lr=0.005)
    loss_func = nn.MSELoss()

    for i in range(100):
        npState = torch.randn(1, S_LEN)
        next_npState = torch.randn(1, S_LEN)
        print('state is', npState)

        reward = torch.randn(AGENT_NUM)
        print('reward is', reward)

        action = a_net.forward(npState)
        t_action = a_net.forward(next_npState)

        q = c_net.forward(npState)
        t_q_out = t_c_net.forward(next_npState)

        updateCriticLoss = loss_func(reward, q)

        c_net.zero_grad()
        updateCriticLoss.backward()
        c_optim.step()





