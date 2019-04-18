# Simple REINFORCE policy gradient
# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class Reinforce(nn.Module):
    def __init__(self, env, agent_index, obs_shape, action_shape):
        super(Reinforce, self).__init__()
        self.linear1 = nn.Linear(obs_shape, 64)
        self.dropout = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(64, action_shape)

        self.saved_log_probs = []
        self.rewards = []
        self.policy_losses = []

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=1)

    def action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode1(self, optimizer, gamma):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def finish_episode(self, optimizer, gamma):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)  # Not collecting average?
        # optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        self.policy_losses.append(policy_loss)

    def update(self, optimizer, inner_updates):
        optimizer.zero_grad()
        try:
            policy_loss = torch.cat(self.policy_losses).sum() / inner_updates
        except:
            policy_loss = torch.stack(
                self.policy_losses, dim=0).sum() / inner_updates
        policy_loss.backward()
        optimizer.step()
        self.policy_losses = []
        del self.rewards[:]
        del self.saved_log_probs[:]
