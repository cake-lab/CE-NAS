import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from RLnet import ActorNetwork, CriticNetwork
from a3c_params import *

PATH = './results/'
RAND_RANGE = 1000

class A3C(object):
    """
    Asynchronous Advantage Actor-Critic (A3C) algorithm implementation.
    """
    def __init__(self, is_central, model_type, s_dim, action_dim, actor_lr=1e-4, critic_lr=1e-1, 
                 load_checkpoint=False, continuous_action=False):
        self.s_dim = s_dim
        self.a_dim = action_dim
        self.discount = 0.99
        self.entropy_weight = 0.5
        self.entropy_eps = 1e-6
        self.model_type = model_type
        self.continuous_action = continuous_action
        self.is_central = is_central
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor network
        self.actorNetwork = ActorNetwork(self.s_dim, self.a_dim, continuous=self.continuous_action).to(self.device)
        if load_checkpoint:
            self.actorNetwork.load_state_dict(torch.load(actor_checkpoint))

        if self.is_central:
            self.actorOptim = torch.optim.RMSprop(self.actorNetwork.parameters(), lr=actor_lr, alpha=0.9, eps=1e-10)
            self.actorOptim.zero_grad()
            if model_type < 2:
                '''
                model==0 mean original
                model==1 mean critic_td
                model==2 mean only actor
                '''
                self.criticNetwork = CriticNetwork(self.s_dim).to(self.device)
                if load_checkpoint:
                    self.criticNetwork.load_state_dict(torch.load(critic_checkpoint))
                self.criticOptim = torch.optim.RMSprop(self.criticNetwork.parameters(), lr=critic_lr, alpha=0.9, eps=1e-10)
                self.criticOptim.zero_grad()
        else:
            self.actorNetwork.eval()

        self.loss_function = nn.MSELoss()

    def getNetworkGradient(self, s_batch, a_batch, r_batch):
        """Calculate gradients for actor and critic networks."""
        s_batch = torch.cat(s_batch).to(self.device)
        a_batch = torch.LongTensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)

        # Calculate discounted rewards
        R_batch = self._calculate_discounted_rewards(r_batch)

        if self.model_type < 2:
            with torch.no_grad():
                v_batch = self.criticNetwork(s_batch).squeeze().to(self.device)
            td_batch = R_batch - v_batch
        else:
            td_batch = R_batch

        # Calculate actor loss
        actor_loss, entropy_loss = self._calculate_actor_loss(s_batch, a_batch, td_batch)
        total_loss = actor_loss + entropy_loss
        total_loss.backward()

        # Calculate critic loss if applicable
        if self.model_type < 2:
            critic_loss = self._calculate_critic_loss(s_batch, R_batch)
            critic_loss.backward()

    def _calculate_discounted_rewards(self, r_batch):
        """Calculate discounted rewards."""
        R_batch = torch.zeros(r_batch.shape).to(self.device)
        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0]-1)):
            R_batch[t] = r_batch[t] + self.discount * R_batch[t+1]
        return R_batch

    def _calculate_actor_loss(self, s_batch, a_batch, td_batch):
        """Calculate actor loss and entropy loss."""
        if not self.continuous_action:
            probability = self.actorNetwork(s_batch)
            m_probs = Categorical(probability)
            log_probs = m_probs.log_prob(a_batch)
            actor_loss = torch.sum(log_probs * (-td_batch))
            entropy_loss = -self.entropy_weight * torch.sum(m_probs.entropy())
        else:
            mu, sigma = self.actorNetwork(s_batch)
            normal_dist = Normal(mu, sigma)
            log_probs = normal_dist.log_prob(a_batch)
            actor_loss = torch.sum(log_probs * (-td_batch))
            entropy_loss = -self.entropy_weight * torch.sum(normal_dist.entropy())
        return actor_loss, entropy_loss

    def _calculate_critic_loss(self, s_batch, R_batch):
        """Calculate critic loss based on model type."""
        if self.model_type == 0:
            return self.loss_function(R_batch, self.criticNetwork(s_batch).squeeze())
        else:
            v_batch = self.criticNetwork(s_batch[:-1]).squeeze()
            next_v_batch = self.criticNetwork(s_batch[1:]).squeeze().detach()
            return self.loss_function(R_batch[:-1] + self.discount * next_v_batch, v_batch)

    def actionSelect(self, stateInputs):
        """Select an action based on the current state."""
        if not self.is_central:
            with torch.no_grad():
                if not self.continuous_action:
                    probability = self.actorNetwork(stateInputs)
                    m = Categorical(probability)
                    return m.sample().item()
                else:
                    mu, sigma = self.actorNetwork(stateInputs)
                    normal_dist = Normal(mu, sigma)
                    action = torch.clamp(normal_dist.sample([1]).squeeze(0), min=0, max=self.a_dim)
                    return action.item()

    def hardUpdateActorNetwork(self, actor_net_params):
        """Perform a hard update of the actor network parameters."""
        for target_param, source_param in zip(self.actorNetwork.parameters(), actor_net_params):
            target_param.data.copy_(source_param.data)

    def updateNetwork(self):
        """Update the network parameters."""
        if self.is_central:
            self.actorOptim.step()
            self.actorOptim.zero_grad()
            if self.model_type < 2:
                self.criticOptim.step()
                self.criticOptim.zero_grad()

    def getActorParam(self):
        """Get the parameters of the actor network."""
        return list(self.actorNetwork.parameters())

    def getCriticParam(self):
        """Get the parameters of the critic network."""
        return list(self.criticNetwork.parameters())

if __name__ == '__main__':
    # Test code for A3C
    rl_model = A3C(IS_CENTRAL, model_type, state_len, action_len, continuous_action=continuous)
    
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_batch = torch.load('s_batch.pt', map_location=device)
    a_batch = torch.load('a_batch.pt', map_location=device)
    r_batch = torch.load('rFunc.pt', map_location=device)

    batch_size = 6
    for i in range(0, len(r_batch), batch_size):
        cur_s_batch = s_batch[i: i + batch_size]
        cur_a_batch = a_batch[i: i + batch_size]
        cur_r_batch = r_batch[i: i + batch_size]

        print('this is iteration ', i)

        rl_model.getNetworkGradient(s_batch[:len(r_batch)], a_batch[:len(r_batch)], r_batch)
        if i % 30 == 0:
            rl_model.updateNetwork()

    # Save model parameters
    torch.save(rl_model.actorNetwork.state_dict(), actor_checkpoint)
    torch.save(rl_model.criticNetwork.state_dict(), critic_checkpoint)