# Implementation of Meta-Learner based on divergent trajectories
# Relation to prototypical / matching networks applied to RL
# Relation to MAML / Reptile shortest descent algorithms

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import entropy


class MetaLearner(object):
    def __init__(self, policy, obs_shape, action_shape, K=10, num_iters=100,
                 initialize_size=5, look_back='all', lr=1e-2):
        self.algo = policy(0, obs_shape, action_shape)
        # Used for initializing new policies
        self.policy = policy
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        # Update with list of dict objects keyed by 'trajectory' and 'policy', keyed by training iteration
        self.memory = {}
        self.K = K  # how few can we go?
        self.num_iters = num_iters  # how many training iters per episode
        # Number of tasks to do regular policy updates with
        self.initialize_size = initialize_size
        self.look_back = look_back  # number of previous episodes to look back
        self.episode = 0  # current episode
        self.KDEs = {}  # KDE for entirety of this update iteration

        self.env = None  # environment
        self.episode_len = 100
        self.optimizer = None  # specify Adam or SGD
        self.gamma = lr

        self.current_policy = None
        self.meta_divergences = {}  # Also start at 1 (should match memory)
        self.meta_samples = {}
        self.meta_occupancies = {}

    def update_meta_property(self, property, item, iteration):
        """Due to indexing, helper method to help update meta properties"""
        try:
            property[iteration].append(item)
        except:
            property[iteration] = [item]

    def sample(self, policy):  # tasks are entity types encountered
        obs_n = self.env.reset()
        observations = []
        for _ in range(self.episode_len):
            ep_reward = 0
            act_n = []
            act_n.append(policy.action(obs_n[0]))
            obs_n, reward_n, done_n, _ = self.env.step(act_n)
            policy.rewards.append(reward_n[0])
            ep_reward += reward_n[0]
            observations.append(obs_n[0])
        policy.finish_episode(self.optimizer, self.gamma)
        return observations, ep_reward

    def adapt(self, policy, optimizer, K):
        # After sampling multiple times, get updates
        policy.update(optimizer, K)
        # self.current_policy.update(self.optimizer, self.K)

    def train(self, K, iter_num):
        """
        Run k updates (K or 1), adapt, and run another, saving this trajectory and the adapted model
        :iter_num: the current iteration (used for indexing)
        """
        if K > 1:  # K + 1 shots overall?
            for _ in range(K):
                self.sample(self.current_policy)
            self.adapt(self.current_policy, self.optimizer,
                       self.K)  # policy.update(optimizer, K)
            # new trajectory with updated policy
            trajectory, ep_reward = self.sample(self.current_policy)
            kde = self.calculate_KDE(trajectory)
            update = {'trajectory': trajectory,
                      'policy': self.current_policy,
                      'kde': kde,
                      'reward': ep_reward,
                      'optimizer': self.optimizer,
                      'update': iter_num}
            try:
                self.memory[iter_num].append(update)  # could really be a list
            except:
                self.memory[iter_num] = [update]
        else:
            trajectory = self.sample(policy)
        return trajectory

    def update(self):
        """Update and find nearest neighbor from before to initialize at"""
        new_trajectory = self.train(policy)

    def calculate_KDE(self, states, bw=np.logspace(-1, 1, 20), cv=5):
        """Given state trajectories, calculate KDE"""
        params = {'bandwidth': bw}
        grid = GridSearchCV(KernelDensity(), params, cv=cv)
        grid.fit(states)
        kde = grid.best_estimator_
        return kde

    def sample_kde(self, kde, num_samples=100, seed=0):
        """Sample datapoints from given kde"""
        return kde.sample(num_samples, random_state=seed)

    def calculate_JSD(self, p, q, base=np.e):
        """Calculate Jensen-Shannon divergence"""
        p, q = np.asarray(p), np.asarray(q)
        p, q = p / p.sum(), q / q.sum()  # normalize to probabilities
        m = 1. / 2 * (p + q)
        return entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.

    def calculate_occupancy(self, kde, policy, sample):
        state_freq = np.exp(kde.score(sample.reshape(1, -1)))
        _, action_freq = policy.action(sample, distribution=True)
        return action_freq.detach() * state_freq

    def calculate_KNN(self, K, policy, trajectory, iteration, sample_num=100):
        """Return K nearest policies for a given policy and training iteration"""
        kde_policy = self.calculate_KDE(trajectory)
        kde_all = self.KDEs[iteration]

        saved_policies = [i['policy'] for i in self.memory[iteration]]
        samples = self.meta_samples[iteration]

        sample_divergences = []
        for sample in samples:
            sample_key = '-'.join(map(str, sample))
            om = self.calculate_occupancy(kde_policy, policy, sample)
            sample_divergence = np.zeros(len(saved_policies))
            for n in range(len(saved_policies)):
                print('iter: {}'.format(iteration))
                print('n: {}'.format(n))
                print(self.meta_occupancies)
                saved_om = self.meta_occupancies[iteration][n][sample_key]
                jsd = self.calculate_JSD(om.squeeze(), saved_om.squeeze())
                sample_divergence[n] = jsd

            sample_probs = np.exp(kde_all.score(sample.reshape(1, -1)))
            sample_divergences.append(sample_divergence * sample_probs)

        divergence = np.sum(sample_divergences, axis=0)
        closest_ixs = divergence.argsort()[:K]
        return closest_ixs

    def get_updated_policies(self, K, policy, trajectory, iteration,
                             sample_num=100):
        closest_ixs = self.calculate_KNN(
            K, policy, trajectory, iteration, sample_num)
        policies = []
        new_iteration = len(self.memory)
        updated_policies = self.memory[new_iteration]
        for ix in range(len(updated_policies)):
            if ix in closest_ixs:
                policies.append(updated_policies[ix])
        return policies  # pick the one with the best reward?

    def calculate_distances(self, iterations=[1]):
        """
        Given training iterations, calculate nearby policies for all policies
        :iterations: list of iterations, default consider initial
        """
        for i in iterations:
            saved_policies = self.memory[i]
            # Load relevant saved experiences
            trajectories = []
            policies = []
            kdes = []

            for p in saved_policies:
                trajectories.extend(p['trajectory'])
                policies.append(p['policy'])
                kdes.append(p['kde'])

            print(trajectories)
            num_policies = len(policies)
            
            kde_all = self.calculate_KDE(trajectories)
            samples = self.sample_kde(kde_all, 100)

            self.KDEs[i] = kde_all
            self.meta_samples[i] = samples

            sample_divergences = []

            for sample in samples:
                probs_n = []
                sample_divergence = np.zeros([num_policies, num_policies])
                for ix in range(num_policies):
                    occupancy_measure = self.calculate_occupancy(
                        kdes[ix], policies[ix], sample)
                    probs_n.append(occupancy_measure)
                    # Save occupancy measures for comparison later
                    key = '-'.join(map(str, sample))
                    # om_stat = {'-'.join(map(str, sample)): occupancy_measure}

                    try:
                        self.meta_occupancies[i][ix][key] = occupancy_measure
                    except:
                        pass
                    try:
                        self.meta_occupancies[i][ix] = {key: occupancy_measure}
                    except:
                        self.meta_occupancies[i] = {
                            ix: {key: occupancy_measure}}

                for a in range(num_policies):
                    for b in range(num_policies):
                        sample_divergence[a][b] = self.calculate_JSD(
                            probs_n[a].squeeze(), probs_n[b].squeeze())

                sample_prob = np.exp(kde_all.score(sample.reshape(1, -1)))
                sample_divergences.append(sample_divergence * sample_prob)
            # Compute average divergence over entirety
            divergence = np.sum(sample_divergences, axis=0)  # or sum?
            self.meta_divergences[i] = divergence

    def update_nearest_neighbors(self, K, iteration):
        """Update adjacency list of K nearest neighbors, given prior policies"""
        divergences = self.meta_divergences[iteration]
