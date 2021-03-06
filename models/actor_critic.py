# Actor-Critic Model
# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()


class ActorCritic(nn.Module):
    def __init__(self, agent_index, obs_shape, action_shape):
        super(ActorCritic, self).__init__()
        self.affine1 = nn.Linear(obs_shape, 128)
        self.action_head = nn.Linear(128, action_shape)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

        self.losses = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(
            SavedAction(m.log_prob(action), state_value))
        return action.item()

    def finish_episode1(self, optimizer, gamma):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]

    def finish_episode(self, optimizer, gamma):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        # optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()
        self.losses.append(loss)

    def update(self, optimizer, inner_updates):
        optimizer.zero_grad()
        try:
            loss = torch.cat(self.losses).sum() / inner_updates
        except:
            loss = torch.stack(self.losses, dim=0).sum() / inner_updates
        loss.backward()
        optimizer.step()
        self.losses = []
        del self.rewards[:]
        del self.saved_actions[:]
