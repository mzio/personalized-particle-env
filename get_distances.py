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

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import scipy

import pickle

from models.reinforce import Reinforce

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--seed', default=42, type=int,
                    help='Randomization seed')
parser.add_argument('--trained_models', default='./trained_models/',
                    help='Trained model name for visualization')
parser.add_argument('--num_states', default=100, type=int,
                    help='Number of states to collect')
parser.add_argument('--save_results', default='distances.p')
parser.add_argument('--num_models', type=int)
args = parser.parse_args()


# KDE
def get_kde(states, bw=np.logspace(-1, 1, 20), cv=5):
    """:states: is list of states observed"""
    params = {'bandwidth': bw}
    grid = GridSearchCV(KernelDensity(), params, cv=cv)
    grid.fit(states)

    kde = grid.best_estimator_
    return kde


def sample_kde(kde, num_samples=1000, seed=0):
    sample = kde.sample(num_samples, random_state=seed)
    return sample


# JS Divergence
def jsd(p, q, base=np.e):
    # convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    # normalize p, q to probabilities
    p, q = p / p.sum(), q / q.sum()
    m = 1. / 2 * (p + q)
    return scipy.stats.entropy(p, m, base=base) / 2. + scipy.stats.entropy(q, m, base=base) / 2.


action = [['State_x_pos', 'State_y_pos', 'Action', 'Model', 'Time']]

checkpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Load model paths
fpath = args.trained_models

model_cp = {}

for cp in checkpoints:
    model_cp[cp] = [join(fpath, f) for f in listdir(
        fpath) if (isfile(join(fpath, f))) and ('.pt' in f) and ('1_cp_{}.pt'.format(cp) in f)]

model_checkpoints = {}

for c in checkpoints:
    for cp in model_cp[c]:
        entity = cp.split('model_ppe_simple_reinforce_')[-1].split('-')[0]
        try:
            model_checkpoints[int(entity)][c] = cp
        except:
            model_checkpoints[int(entity)] = {c: cp} 

print('Checkpoints loaded!')

# Get trajectories
with open('states.p', 'rb') as f:
    loaded_trajectories = pickle.load(f)

print('Trajectories loaded!')

checkpoint_divergences = {}

for cp in checkpoints:
    print('Processing checkpoint {}...'.format(cp))
    # Go through each model type
    trajectories = []
    policies = []
    kde_n = []  # Collect KDEs for each type, but unsure if necessary
    for t in range(args.num_models):
        trajectories.extend(loaded_trajectories[t][cp])
        kde_n.append(get_kde(loaded_trajectories[t][cp]))
        policy = Reinforce(None, 2, 5)
        policy.load_state_dict(torch.load(model_checkpoints[t][cp]))
        policies.append(policy)
    print('Policies specified!')
    print('Calculating aggregate state KDE...')
    kde = get_kde(trajectories)
    samples = sample_kde(kde, args.num_states)
    sample_divergences = []

    print('Calculating state statistics')
    for sample in samples:
        sample_divergence = np.zeros(
            [args.num_models, args.num_models])  # 24 x 24
        probs_n = []
        for t in range(args.num_models):
            _, probs = policies[t].action(
                sample, distribution=True)  # 4d vector
            # print(probs.detach())
            state_freq = np.exp(kde_n[t].score(sample.reshape(1, -1)))
            probs_n.append(probs.detach() * state_freq)
        # Compute pairwise KL divergence
        # pairwise JS divergence
        print('Computing JS divergence...')
        for a in range(args.num_models):
            for b in range(args.num_models):
                # print(jsd(probs_n[a].squeeze(), probs_n[b].squeeze()))
                sample_divergence[a][b] = jsd(probs_n[a].squeeze(), probs_n[b].squeeze())
        # Check this part
        sample_prob = np.exp(kde.score(sample.reshape(1, -1)))
        sample_divergences.append(sample_divergence * sample_prob)
    # Compute average divergence over entirety
    divergence = np.sum(sample_divergences, axis=0)  # or sum?
    checkpoint_divergences[cp] = divergence
    # print(divergence)

with open('distances_om-1.p', 'wb') as f:
    pickle.dump(checkpoint_divergences, f)

# python get_distances.py --num_states 100 --save_results 'distances.p'  --trained_models './trained_models/simple_reinforce_cp/' --num_models 24

