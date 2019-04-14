# Simple REINFORCE policy gradient
# Inspired by https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Reinforce(nn.Module):
    def __init__(self, env, agent_index, obs_shape, action_shape):
        super(Reinforce, self).__init__()
        self.linear1 = nn.Linear(obs_shape, 64)
        self.dropout = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(64, action_shape)

        self.saved_log_probs = []
        self.rewards = []

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
