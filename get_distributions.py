# Calculate policy distributions by sampling from state space by checkpoint episode
# Can then use KL / JS divergence to measure distances

import gym
import csv
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.distributions import kl

import particles
import particles.scenarios as scenarios
from particles.environment import PersonalAgentEnv

from os import listdir
from os.path import isfile, join

import pickle

from models.reinforce import Reinforce

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument('--trained_models', default='./trained_models/',
                    help='Trained model name for visualization')
parser.add_argument('--state_sampler', default='./episode_states.p')
parser.add_argument('--num_states', default=100, type=int,
                    help='Number of states to collect')
parser.add_argument('--save_results', default='divergences.p')
args = parser.parse_args()

fpath = args.trained_models

action = [['State_x_pos', 'State_y_pos', 'Action', 'Model', 'Time']]

checkpoints = [0, 10, 20, 30, 40, 50]  # Multiply by 10 for actual episode


def check_valid_model(model):
    if '_cp_' not in model:
        return False
    num = int(model.split('_cp_')[-1].split('.')[0])
    if num in checkpoints:
        return True
    return False


# Load models
trained_models = [join(fpath, f) for f in listdir(
    fpath) if (isfile(join(fpath, f))) and ('.pt' in f) and check_valid_model(f)]


# Load state samplers
with open(args.state_sampler, 'rb') as f:
    state_kdes = pickle.load(f)

models = []

# Generate sample states
for ix in range(len(checkpoints)):
    # For variant where we just load the already generated data
    states = state_kdes[ix]
    for model in trained_models:
        # Filter for same checkpoint, so go through reps and diff types at same time
        if int(model.split('_cp_')[-1].split('.')[0]) == checkpoints[ix]:
            # Load model
            name = model.split('_cp_')[0]
            agent_type = int(name.split('_')[-1].split('-')[0])
            rep = int(name.split('_')[-1].split('-')[1])

            policy = Reinforce(None, 2, 5)
            policy.load_state_dict(torch.load(model))
            # Calculate categorical action distributions
            model_info = {'name': name, 'type': agent_type, 'rep': rep,
                          'state_p': []}
            for state in states:
                a, p = policy.action(state, distribution=True)
                model_info['state_p'].append(p)
            models.append(model_info)

pairwise_dists = []

for m1 in models:
    for m2 in models:
        sym_kls = []
        kls = []
        for x in range(len(m1['state_p'])):  # get symmetrized KL
            sym_kl = (kl.kl_divergence(m1['state_p'][x], m2['state_p'][x]) +
                      kl.kl_divergence(m2['state_p'][x], m1['state_p'][x]))
            kl = kl.kl_divergence(m1['state_p'][x], m2['state_p'][x])
            sym_kls.append(sym_kl)
            kls.append(kl)
        pairwise_dists.append({'kl_symmetric': sym_kls,
                               'kl': kls,
                               'p': m1['name'],
                               'p_type': m1['type'],
                               'p_rep': m1['rep'],
                               'q': m2['name'],
                               'q_type': m2['type'],
                               'q_rep': m2['rep']})

with open(args.save_results, 'wb') as f:
    pickle.dump(pairwise_dists, file=f)


# python get_distributions.py --num_states 100 --trained_models './trained_models/'
