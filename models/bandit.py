import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .reinforce import Reinforce

from os import listdir
from os.path import isfile, join

eps = np.finfo(np.float32).eps.item()


class ThompsonSampler():
    def __init__(self, models_path):
        # Initialize with pre-trained models as bandits
        super(ThompsonSampler, self).__init__()
        self.models = self.load_models()
        self.parameters = []
        self.init_probs()
        self.current_model_ix = None
        self.rewards = []

    def load_models(self, dir_name):
        path = '../trained_models/{}'.format(dir_name)
        trained_models = [join(fpath, f)
                          for f in listdir(fpath) if (isfile(join(fpath, f))) and ('.pt' in f)]
        policies = [Reinforce(None, 2, 5) for m in trained_models]
        for ix, m in enumerate(trained_models):
            policies[ix].load_state_dict(torch.load(m))

    def init_probs(self):
        for model in self.models:
            self.parameters.append([1, 1])  # Uniform Prior

    def action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        samples = [np.random.beta(param[0], param[1])
                   for param in self.parameters]
        model_ix = max(range(len(self.models)), key=lambda x: samples[x])
        self.current_model_ix = model_ix
        return self.models[model_ix].action(state)

    def update_params(self, reward):
        try:
            if reward >= self.rewards[-1]:
                self.parameters[self.current_model_ix[0]] += 1
            else:
                self.parameters[self.current_model_ix[1]] += 1
        except:
            pass  # Auto-accept first reward
