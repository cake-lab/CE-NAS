import torch
from carbon_reader import max_carbon, min_carbon
import math

# Global parameters
IS_CENTRAL = True
model_type = 0
budget = 40
max_hv = 4150.7236328125
continuous = False

# Checkpoint file names
actor_checkpoint = 'actor_params.pth' if not continuous else 'actor_continuous.pth'
critic_checkpoint = 'critic_params.pth' if not continuous else 'critic_continuous.pth'

# State and action dimensions
state_len = 4
action_len = 10

def set_state(remaining_budget, carbon_trace, cur_hv, t_samples):
    """
    Set the state for the A3C algorithm.
    
    Args:
        remaining_budget (float): Remaining time budget
        carbon_trace (float): Next carbon trace
        cur_hv (float): Current hypervolume
        t_samples (int): Total number of samples

    Returns:
        torch.Tensor: State tensor
    """
    relative_carbon_rate = (carbon_trace - min_carbon) / (max_carbon - min_carbon)
    hv_rate = cur_hv / max_hv
    remaining_budget_rate = remaining_budget / budget
    state = torch.tensor([[remaining_budget_rate, relative_carbon_rate - 0.5, hv_rate, t_samples]])
    return state.cuda() if torch.cuda.is_available() else state

def get_action(action, action_len=action_len):
    """
    Convert the raw action to a rate for Vanilla NAS GPU allocation.

    Args:
        action (float): Raw action value
        action_len (int): Action space dimension

    Returns:
        float: Normalized action value between 0 and 1
    """
    return action / action_len

# Reward hyperparameters
alpha = 20  # Hypervolume improvement weight
beta = 2 / max_carbon  # Carbon cost weight
theta = 0.05  # Unused in current implementation
gamma = 0.05  # Number of samples weight

def get_reward(prev_hv, cur_hv, carbon_cost, n_samples, remaining_budget, alpha=alpha, beta=beta, theta=theta):
    """
    Part 1: Hypervolume improvement
    Part 2: Carbon cost of this step
    Part 3: difference between current HV and Max HV

    Args:
        prev_hv (float): Previous hypervolume
        cur_hv (float): Current hypervolume
        carbon_cost (float): Carbon cost of the current step
        n_samples (int): Number of finished samples in this step
        remaining_budget (float): Remaining time budget
        alpha (float): Hypervolume improvement weight
        beta (float): Carbon cost weight
        theta (float): Unused in current implementation

    Returns:
        tuple: Reward tensor and a tensor of individual reward components
    """
    cur_hv = cur_hv / max_hv
    prev_hv = prev_hv / max_hv
    hv_improvement = cur_hv - prev_hv
    cur_step = budget - remaining_budget
    step_ratio = (cur_step / budget) * 0.8 + 0.8

    hv_reward = hv_improvement * alpha * step_ratio
    carbon_reward = -carbon_cost * beta
    samples_reward = n_samples * gamma

    print(f'hv_improvement reward is {hv_reward}')
    print(f'carbon_cost reward is {carbon_reward}')
    print(f'n_samples reward is {samples_reward}')

    reward = hv_reward + carbon_reward + samples_reward
    print(f'total reward is {reward}')

    reward_tensor = torch.tensor([reward])
    components_tensor = torch.tensor([hv_improvement, -carbon_cost, cur_hv, n_samples, cur_step, budget])

    if torch.cuda.is_available():
        return reward_tensor.cuda(), components_tensor.cuda()
    else:
        return reward_tensor, components_tensor
