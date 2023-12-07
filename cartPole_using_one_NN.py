import argparse
import gymnasium as gym 
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn # To make weight and Bias part of the nn
import torch.nn.functional as F # for activation function
import torch.optim as optim # 
from torch.distributions import Categorical


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

#env is a high level implementation of MDPs (environments)

env = gym.make('CartPole-v1', render_mode="rgb_array")  # Initializing gym env
env.reset(seed=543)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) # creating a class(SavedAction) with parameters log_prob and value 


class Policy(nn.Module): # NN class, inheriting from nn.module 
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__() # initialization  
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = [] 
        # will store information about the actions taken
        self.rewards = [] 
        # will store the rewards 

    def forward(self, x):
        """
        forward of both actor and critic
        """
    
        x = F.relu(self.affine1(x))
    
        # returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: estimate of the state value
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy() # instantiating the policy 
optimizer = optim.Adam(model.parameters(), lr=3e-2) # Adam optimizer for updating the weights with learning rate = 0.03
eps = np.finfo(np.float32).eps.item() # for some sort of numerical stability (even I didn't get it)


def select_action(state):
    state = torch.from_numpy(state).float() # converting state values to pytorch tensor and casting it to float data type
    probs, state_value = model(state) # model returning probability of actions and state values 

    # create a categorical distribution over the list of probabilities of actions (since our actions are discrete)
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)

    # print(action.item())
    return action.item()


# The official implementation of cartpole using actor critic method in pytorch example is incorrect as per Sutton Barto Book
# As they didn't calculate the TD(0) error rather they used Monte Carlo method due to which we've to wait for each episode to end
# In order to get the returns ( which is nothing but Policy Gradient with baseline using Monte Carlo approach)
# In order to solve this issue I've made changes to the finish_episode funtion.

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss

    for t in range(len(model.rewards)-1):
        reward = model.rewards[t]
        next_value = model.saved_actions[t+1].value
        value = model.saved_actions[t].value
        log_prob = model.saved_actions[t].log_prob

        # TD Target
        td_target = reward + args.gamma * next_value.item()
        td_error = td_target - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * td_error)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([td_target])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

print("Environment Reward Threshold:", env.spec.reward_threshold)


################### Pytorch implementation of finish_episode ########################

# def finish_episode():
#     """
#     Training code. Calculates actor and critic loss and performs backprop.
#     """
#     R = 0 # counter for accumulated rewards
#     saved_actions = model.saved_actions #List of named tuples (SavedAction) containing the log probability of the action and the state value.
#     policy_losses = [] # list to save actor (policy) loss
#     value_losses = [] # list to save critic (value) loss
#     returns = [] # list to save the true values

#     # calculate the true value using rewards returned from the environment
#     for r in model.rewards[::-1]: # Loops through the rewards in reverse order (from the last time step to the first).
#         # calculate the discounted value
#         R = r + 0.99 * R
#         returns.insert(0, R)

#     returns = torch.tensor(returns)
#     returns = (returns - returns.mean()) / (returns.std() + eps)

#     for (log_prob, value), R in zip(saved_actions, returns):
#         advantage = R - value.item()

#         # calculate actor (policy) loss
#         policy_losses.append(-log_prob * advantage)

#         # calculate critic (value) loss using L1 smooth loss
#         value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

#     # reset gradients
#     optimizer.zero_grad()

#     # sum up all the values of policy_losses and value_losses
#     loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

#     # perform backprop
#     loss.backward()
#     optimizer.step()

#     # reset rewards and action buffer
#     del model.rewards[:]
#     del model.saved_actions[:]


def main():
    running_reward = 1

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        #print(running_reward)
        
        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()